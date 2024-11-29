from  helpers.cordinates import get_cordinates
from  helpers.calculate_path import calculate_path

import logging

logging.basicConfig(level=logging.INFO)


def GetCordinates(parameters):
    """Turns on the robot and updates the LLM."""
    # Call the function to process the coordinates
    logging.info(f"Parameters: {parameters}")
    starting_location, ending_location = get_cordinates(parameters)
    return [starting_location, ending_location]


def GetOptimalPath(parameters):
    """ get the optimal path """
    starting_coordinates, ending_coordinates = calculate_path(parameters)
    return starting_coordinates, ending_coordinates