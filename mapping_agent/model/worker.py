import random
import numpy as np
import torch
import torch.nn.functional as F

from  model.actor_critic import ActorCriticLSTM

# Function to calculate the discounted cumulative rewards
def compute_discounted_rewards(rewards, gamma):
    discounted_rewards = []
    cumulative_reward = 0
    for reward in reversed(rewards):
        cumulative_reward = reward + gamma * cumulative_reward
        discounted_rewards.insert(0, cumulative_reward)
    return discounted_rewards

# Worker process for asynchronous training
def worker( global_model, optimizer, global_episode_counter, lock, env , gamma=0.99, max_steps=45):

    # Set a fixed seed value
    seed = 42

    # Set seeds for various libraries
    random.seed(seed)             # Python random module
    np.random.seed(seed)          # NumPy random module
    torch.manual_seed(seed)       # PyTorch CPU random numbers
    torch.cuda.manual_seed(seed)  # PyTorch GPU random numbers (if using CUDA)
    torch.cuda.manual_seed_all(seed)  # PyTorch all GPU random numbers (for multi-GPU setups)

    # Ensure deterministic behavior for PyTorch on GPU (if needed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    local_model = ActorCriticLSTM(input_size=env.observation_space.shape[0], # 6
                                  hidden_size=256,
                                  action_size=env.action_space.n)

    local_model.load_state_dict(global_model.state_dict())
    hx, cx = local_model.init_hidden()  # Initialize hidden and cell states

    while global_episode_counter.value < 5000:  # Some stopping condition based on episodes
        state = env.reset()

        done = False
        episode_reward = 0
        hx, cx = local_model.init_hidden()  # Reset LSTM hidden state at the beginning of each episode

        states, actions, rewards, log_probs, values = [], [], [], [], []

        for step in range(max_steps):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Shape (batch_size=1, seq_len=1, input_size)
     
            policy, value, hx, cx = local_model(state_tensor, hx, cx)

            action_prob = F.softmax(policy, dim=-1)

            action = torch.multinomial(action_prob, 1).item()

            log_prob = torch.log(action_prob.squeeze(0)[action])  # Log probability of the chosen action

            next_state, reward, done, _ = env.step(action)
            episode_reward += reward

            # Store state, action, reward, log_prob, and value
            states.append(state_tensor)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value.squeeze(0))

            state = next_state

            if done:
                break

        # Calculate discounted rewards
        discounted_rewards = compute_discounted_rewards(rewards, gamma)
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).unsqueeze(1)

        # Calculate advantages
        values = torch.stack(values)
        advantages = discounted_rewards - values

        # Loss calculation
        policy_loss = sum(-log_prob * advantage.detach() for log_prob, advantage in zip(log_probs, advantages))
        value_loss = F.mse_loss(values, discounted_rewards)
        entropy_loss = -sum(torch.sum(torch.exp(log_prob) * log_prob) for log_prob in log_probs)

        # Total loss
        total_loss = policy_loss + 0.5 * value_loss -  1 * entropy_loss

        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(local_model.parameters(), 50)

        # Update global model
        with lock:
            for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
                global_param._grad = local_param.grad
            optimizer.step()

        # Sync local model with global model
        local_model.load_state_dict(global_model.state_dict())

        # Update global episode counter
        with lock:
            global_episode_counter.value += 1
            print(f"Episode: {global_episode_counter.value}, Loss: {global_param._grad.item()}")
            if float(global_param._grad.item()) >= 0.0 and float(global_param._grad.item()) < 1:
              break

            
            