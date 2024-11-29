from collections import defaultdict
import copy
import numpy as np

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


class State:
    def __init__(self, grid_layout=None, max_steps=1000):

        # generate the grid layout
        self.grid_layout = grid_layout
        self.initial_grid_state = np.array(self.grid_layout)
        self.grid_state = copy.deepcopy(self.initial_grid_state)

        # dictionary for possible actions
        self.action_pos_dict = defaultdict(
            lambda: [0, 0],
            {
                NOOP: [0, 0],
                FORWARD: [-1, 0],
                BACKWARD: [1, 0],
                LEFT: [0, -1],
                RIGHT: [0, 1],
            },
        )

        # get current state
        (self.agent_state, self.goal_state) = self.get_state()

        self.done = False
        self.info = {"status": "Live"}
        self.step_num = 0  # To keep track of number of steps
        self.max_steps = max_steps
        self.reward = 0

    def get_shape(self):
        return self.grid_state.shape

    def get_state(self):
        """
        Get the agent's position, as well as the goal's position
        """
        start_state = np.where(self.grid_state == AGENT)
        goal_state = np.where(self.grid_state == GOAL)

        start_or_goal_not_found = not (len(start_state[0]) and len(goal_state[0]))
        if start_or_goal_not_found:
            print(
                "Start and/or Goal state not present in the Gridworld. "
                "Check the Grid layout"
            )
            return None, None
        start_state = (start_state[0][0], start_state[1][0])
        goal_state = (goal_state[0][0], goal_state[1][0])

        return start_state, goal_state

    def reset(self):
        """Reset the grid state
        """
        self.grid_state = copy.deepcopy(self.initial_grid_state)
        self.done = False
        self.info["status"] = "Live"
        self.step_num = 0
        self.reward = 0
        self.agent_state, self.goal_state = self.get_state()
        return self.grid_state

    def is_next_state_invalid(self, next_state):
        """
        Check if the next agent move is invalid or not
        """
        next_state_invalid = (
            next_state[0] < 0 or next_state[0] >= self.grid_state.shape[0]
        ) or (next_state[1] < 0 or next_state[1] >= self.grid_state.shape[1])
        return next_state_invalid

    def execute_action(self, action):
        """Execute the given action and calculate the reward
        """

        # get the next state
        next_state = (
            self.agent_state[0] + self.action_pos_dict[action][0],
            self.agent_state[1] + self.action_pos_dict[action][1],
        )

        # check if the next state is valid
        next_state_invalid = self.is_next_state_invalid(next_state)

        # Leave the agent state unchanged
        if next_state_invalid:
            next_state = self.agent_state
            self.info["status"] = "Next state is invalid"

        # Get the next agent state
        next_agent_state = self.grid_state[next_state[0], next_state[1]]

        # Calculate reward
        if next_agent_state == 0:
            # Move agent from previous state to the next state on the grid
            self.info["status"] = "Agent moved to a new cell"
            # update the agent state
            self.grid_state[next_state[0], next_state[1]] = AGENT
            self.grid_state[self.agent_state[0], self.agent_state[1]] = EMPTY
            self.agent_state = copy.deepcopy(next_state)
            self.reward = self.reward + 1

        elif next_agent_state == 1:
            # Agent hit a wall with this move
            self.info["status"] = "Agent bumped into a wall"
            self.reward = -0.1
        elif next_agent_state == 4:
            # Terminal state: goal reached
            self.info["status"] = "Agent reached the GOAL "
            self.done = True
            self.reward = 1
        else:
            # NOOP or next state is invalid
            self.done = False

        self.step_num += 1

        self.grid_state[self.agent_state[0], self.agent_state[1]] = self.grid_state[next_state[0], next_state[1]]

        # if the reward drops below 0
        # Check if max steps per episode has been reached
        if self.step_num >= self.max_steps:
            self.done = True
            self.info["status"] = "Max steps reached"

        return self.reward, self.done, self.info