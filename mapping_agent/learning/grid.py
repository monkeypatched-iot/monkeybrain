
import numpy as np

# Grid cell state and color mapping
EMPTY = BLACK = 0  # Assuming white for empty space
WALL = WHITE = 1    # Assuming grey for a wall
AGENT = RED = 2    # Assuming red for the agent
SHELF = BLUE = 3   # Assuming blue for the shelf
GOAL = GREEN = 4   # Assuming green for the goal

# RGB color value table (optional, for visualization)
COLOR_MAP = {
    WHITE:  [1.0, 1.0, 1.0],  # Assuming white for empty space
    BLUE:   [0.6, 0.4, 0.2],
    GREEN:  [0.0, 1.0, 0.0],
    RED:    [0.0, 0.0, 1.0],  # Optional for agent visualization
    BLACK:  [0.0, 0.0, 0.0],
}

class GridRenderer:
    def __init__(self, img_shape, grid_shape):
        self.img_shape = img_shape
        self.grid_shape = grid_shape

    def render(self, grid_state):
        observation = np.zeros(self.img_shape, dtype=np.uint8)
        scale_x = int(observation.shape[0] / self.grid_shape[0])
        scale_y = int(observation.shape[1] / self.grid_shape[1])

        for i in range(self.grid_shape[0]):
            for j in range(self.grid_shape[1]):
                cell_value = grid_state[i, j]
                cell_color = COLOR_MAP[cell_value]

                observation[
                    i * scale_x : (i + 1) * scale_x,
                    j * scale_y : (j + 1) * scale_y,
                    :,
                ] = np.array(cell_color) * 255

        return observation