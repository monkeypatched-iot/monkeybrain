import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

# Define the Actor-Critic network with LSTM
class ActorCriticLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, action_size):
        super(ActorCriticLSTM, self).__init__()

        self.lstm = nn.LSTM(10, hidden_size, batch_first=True)
        self.actor = nn.Linear(hidden_size, action_size)
        self.critic = nn.Linear(hidden_size, 1)

        self.hidden_size = hidden_size

    def forward(self, x, hx, cx):
        x, (hx, cx) = self.lstm(x, (hx, cx))
        x = x[:, -1, :]  # Use the last output of the LSTM
        policy = self.actor(x)
        value = self.critic(x)
        return policy, value, hx, cx

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_size), torch.zeros(1, 1, self.hidden_size))