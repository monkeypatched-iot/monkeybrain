import numpy as np
import torch
from torch.distributions import Categorical
from PIL import Image
import os
from database.spatial import SpatialDatabase
import logging

logging.basicConfig(level=logging.INFO)


# connect to postgres
db_name = os.getenv("SPATIAL_DBNAME")
user = os.getenv("SPATIAL_DB_USER")
password = os.getenv("SPATIAL_DB_PASSWORD")
host = os.getenv("SPATIAL_DB_HOST")
port = os.getenv("SPATIAL_DB_PORT")

# Function to select action from policy network
def select_action(model, state):

    # Add batch and sequence dimensions
    state = torch.from_numpy(state).float().unsqueeze(0)

    # Initialize hidden and cell states
    hx, cx = model.init_hidden()

    # Forward pass through the model
    probs, state_value, _, _ = model(state, hx, cx)  

    # Normalize output to probabilities
    probs = torch.softmax(probs, dim=-1)

    m = Categorical(probs)

    action = m.sample()

    # Log probability of the chosen action
    return action.item()

def save_path(env,global_model):

     # List to store frames
    frames = [] 

    # List to save the pathe cordinates
    path_cords = []

    # create spaial database connection
    spatial_db = SpatialDatabase(db_name=db_name, user=user, password=password, host=host, port=port)

    # reset the enviornment
    state = env.reset()

    for t in range(1000):
        # Infer the next action using the policy network
        action = select_action(global_model, state)

        # Take the action in the environment
        next_state, reward, done, _ = env.step(action)

        # Capture the rendered frame (Assuming env.render() returns the frame)
        frame = env.render(mode='rgb_array')
        frames.append(Image.fromarray(frame))

        if done:
            print(f"Episode finished after {t+1} timesteps")
            break

        # Logic for finding positions of '2' and '4' in the game state (if applicable)
        positions_2 = np.argwhere(state == 2)
        positions_4 = np.argwhere(state == 4)

        # Flag to break both loops
        found = False

        # Check if 2 is adjacent to 4
        for pos_2 in positions_2:
            for pos_4 in positions_4:
                if (pos_2[0] == pos_4[0] and abs(pos_2[1] - pos_4[1]) == 1) or (pos_2[1] == pos_4[1] and abs(pos_2[0] - pos_4[0]) == 1):
                    print(f"Breaking: '2' at {pos_2} is adjacent to '4' at {pos_4}")
                    tuple_result = tuple(positions_2.flatten())
                    logging.info(positions_2)
                    path_cords.append(tuple_result)
                    found = True
                    break
        if found:
            break

        # Update state for the next iteration
        state = next_state  

        tuple_result = tuple(positions_2.flatten())
        path_cords.append(tuple_result)

    # Filter out empty tuples before unpacking
    filtered_path_coords = [cord for cord in path_cords if len(cord) >= 2]

    path = list(set(filtered_path_coords))

    # save the path in the database
    spatial_db.insert_path(list(set(filtered_path_coords)))

    return path
