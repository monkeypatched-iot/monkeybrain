from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from learning.env import GridworldEnv  # Assuming this is a custom Gym environment
import json
import numpy as np
import gym
import torch
import torch.optim as optim
import torch.multiprocessing as mp
import dill
from utils.keyframes_utils import convert_to_segments, find_keyframes_along_path
from model.worker import worker  # Assuming you have a worker script for RL tasks
import gnwrapper  # Assuming this is some custom library related to the environment
from model.actor_critic import ActorCriticLSTM  # Assuming this is your RL model
from gym.envs.registration import register
from database.path import save_path
from database.spatial import SpatialDatabase
import boto3
import numpy as np
import io
import os
import logging

logging.basicConfig(level=logging.INFO)

# connect to postgres
# Reading database credentials from environment variables
db_name = os.getenv("SPATIAL_DBNAME ")
user = os.getenv("SPATIAL_DB_USER")
password = os.getenv("SPATIAL_DB_PASSWORD")
host = os.getenv("SPATIAL_DB_HOST")
port = os.getenv("SPATIAL_DB_PORT")

# get all keyframe
spatial = SpatialDatabase(db_name=db_name, user=user, password=password, host=host, port=port)

# Function to generate a map based on start and stop positions
def generate_map(flipped_rotated_bw_map, start_position, stop_position):
    map = flipped_rotated_bw_map.copy()

    if map[start_position[0], start_position[1]] == 0:
        map[start_position[0], start_position[1]] = 2  # Mark start position with 2
    else:
        print("Start position is already occupied.")
        map[start_position[0], start_position[1]] = 1  # Occupied, mark with 1
    
    if map[stop_position[0], stop_position[1]] == 0:
        map[stop_position[0], stop_position[1]] = 4  # Mark stop position with 4
    else:
        print("Stop position is already occupied.")
        map[stop_position[0], stop_position[1]] = 1  # Occupied, mark with 1

    return map

# Function to get the optimal path from a given prompt
def get_map(prompt_text):
    # Define the prompt template
    template = """Given the following text extract the starting and ending coordinates and return 
                them as a JSON object.The output should include "starting_cordinates" and "ending_cordinates" 
                as the keys,with their corresponding values and return the response as JSON. The text is {text}"""

    # Create the prompt and model instances
    prompt = ChatPromptTemplate.from_template(template)
    model = OllamaLLM(model="mistral")  # Assuming "grid_layout=mapmistral" is a model identifier for your LLM

    # Chain the prompt with the model
    chain = prompt | model

    # Invoke the model with the input text
    response = chain.invoke({"text": prompt_text})

    # Attempt to parse the response as JSON
    try:
        output = json.loads(response)
    except json.JSONDecodeError:
        logging.error("Error: The response is not in valid JSON format.")
        return None

    # Print the full output for debugging
    logging.info("Model Response:", output)

    # Safely extract starting and ending coordinates
    starting_coordinates = output.get('starting_coordinates') or output.get('starting_coordinate')
    ending_coordinates = output.get('ending_coordinates') or output.get('ending_coordinate')

    if starting_coordinates and ending_coordinates:
        try:
            # Parse coordinates from string format '(x,y)' to tuple (x, y)
            starting_coordinates = tuple(map(int, starting_coordinates.strip('()').split(',')))
            ending_coordinates = tuple(map(int, ending_coordinates.strip('()').split(',')))
        except ValueError:
            logging.error("Error: Coordinates format is incorrect.")
            return None
    else:
        logging.error("Error: Starting or ending coordinates are missing.")
        return None

    # Initialize the S3 client
    s3 = boto3.client('s3')

    # Define S3 bucket and file key
    bucket_name = os.getenv("SIMULATION_BUCKET_NAME")  # Replace with your actual bucket name
    file_key = 'map/map.csv' # Replace 'folder_name/' with your desired path in the bucket

    # Download the CSV file as a stream
    response = s3.get_object(Bucket=bucket_name, Key=file_key)

    # Read the content of the file into a numpy array
    matrix = np.genfromtxt(io.StringIO(response['Body'].read().decode('utf-8')), delimiter=',', skip_header=1)

    # Generate the environment map with the provided coordinates
    env_map = generate_map(matrix, starting_coordinates, ending_coordinates)
 
    return env_map

def create_gridworld_env_with_map(map=None, size=6, reward_type='sparse'):
    return GridworldEnv(map=map, size=size, reward_type=reward_type)

def register_custom_env(id='Gridworld-v0', entry_point=create_gridworld_env_with_map, map=None, max_episode_steps=200):
    # Register the custom environment with the factory function as entry_point
    register(
        id=id,
        entry_point=entry_point,  # Now we pass the factory function as entry point
        max_episode_steps=max_episode_steps,  # Set maximum steps per episode if desired
    )
    print(f"Custom environment '{id}' registered with entry point '{entry_point.__name__}' and map={map} and max_episode_steps={max_episode_steps}!")


def calculate_path(prompt_text):
    # Example prompt text to provide the model
 
    # Get the optimal path (starting and ending coordinates)
    env_map = get_map(prompt_text)

    # If the map generation was successful, initialize the custom environment
    if env_map is not None:
        register_custom_env(id='CustomGridworld-v1', entry_point=create_gridworld_env_with_map, map=env_map, max_episode_steps=250)

        # Create the Gym environment
        env = create_gridworld_env_with_map(map=env_map, size=6, reward_type='dense')

        # global model for experience learning
        global_model = ActorCriticLSTM(input_size=env.observation_space.shape[0], hidden_size=256, action_size=env.action_space.n)
        
        # back prop optimizer for global model
        optimizer = optim.Adam(global_model.parameters(), lr=0.001)
        
        # Create shared variables
        global_episode_counter = mp.Value('i', 0)
        lock = mp.Lock()

        # Run workers without multithreading
        worker(global_model, optimizer, global_episode_counter, lock, env)

        # calculate and save path
        path = save_path(env,global_model)

        keyframes = spatial.get_all_keyframes()

        path_segments = convert_to_segments(path) # Paths as segments

        # find the keyframes on the optimal path
        path = find_keyframes_along_path(keyframes, path_segments, max_distance=5.0)

        nearby_points = sorted(list(set(path)))

        env.close()

        # Convert each tuple to a dictionary
        keys = ["id", "sequence", "timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"]
        json_ready_data = [dict(zip(keys, entry)) for entry in nearby_points]

        # Convert to JSON string
        json_output = json.dumps(json_ready_data, indent=4)
        logging.info(json_output)

        return {"path":json_output}
    else:
        print("Failed to generate the environment map.")