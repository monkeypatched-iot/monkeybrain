import copy
import numpy as np
import gym
from learning.state import State
from learning.grid import GridRenderer

# Grid cell state and color mapping
EMPTY = BLACK = 0  # Assuming white for empty space
WALL = WHITE = 1    # Assuming grey for a wall
AGENT = RED = 2    # Assuming red for the agent
SHELF = BLUE = 3   # Assuming blue for the shelf
GOAL = GREEN = 4   # Assuming green for the goal

# Action mapping
START = 1
STOP = 2
LEFT = 3
RIGHT = 4
FORWARD = 5
BACKWARD = 6
NOOP = 7

class GridworldEnv(gym.Env):
    def __init__(self, map=None, size=10, reward_type='sparse'):
        super(GridworldEnv, self).__init__()
        
        # Default map if none is provided
        self.map = map if map is not None else np.zeros((size, size), dtype=int)
        self.size = size
        self.reward_type = reward_type
   
        # map is to be pased to state ?

        self.state = State(grid_layout=self.map)
        self.observation_space = gym.spaces.Box(
            low=0, high=size, shape=(size,size), dtype=np.uint8
        )
        self.metadata = {"render.modes": ["human"]}

        # Actions
        self.actions = [NOOP, FORWARD, BACKWARD, LEFT, RIGHT]
        self.action_space = gym.spaces.Discrete(len(self.actions))
        self.viewer = None
        self.renderer = GridRenderer([256, 256, 3], self.state.get_shape())
        # Initialize the random number generator with a seed for reproducibility
        self.rng = np.random.default_rng(seed=42)

    def step(self, action):
        """Return next observation, reward, done, info"""
        action = self.actions[action]
        reward, done, info = self.state.execute_action(action)
        if done:
            terminal_state = copy.deepcopy(self.state)
            _ = self.reset()
            return terminal_state.grid_state, reward, done, info
        return self.state.grid_state, reward, done, info

    def reset(self):
        self.state.reset()
        return self.state.grid_state

    def render(self, mode="human", close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
            self.viewer = None
            return

        grid_state = self.state.grid_state
        img = self.renderer.render(grid_state)

        if mode == "rgb_array":
            return img
        elif mode == "human":
            from gym.envs.classic_control import rendering

            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)

    def close(self):
        self.render(close=True)

    def __getstate__(self):  # Method to control object state when pickling
        state = self.__dict__.copy()
        # Remove rng before pickling to avoid the error
        del state['rng']
        return state

    def __setstate__(self, state):  # Method to restore object state when unpickling
        self.__dict__.update(state)
        # Re-initialize rng after unpickling
        self.rng = np.random.default_rng(seed=42)

    @staticmethod
    def get_action_meanings():
        return ["NOOP", "FORWARD", "BACKWARD", "LEFT", "RIGHT"]