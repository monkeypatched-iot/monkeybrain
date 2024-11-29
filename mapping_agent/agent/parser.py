import re
import json

import logging

logging.basicConfig(level=logging.INFO)

def parse(input_text):


    pattern = r"step:\s*(\d+)\s*action:\s*([A-Za-z0-9_]+)\s*\(parameters:\s*(\{[^}]+\})\)"

    matches = re.findall(pattern, input_text)
    steps = []

    # Output the matches
    for match in matches:
        step_number, action, parameters = match
        logging.info(f"Step: {step_number}, Action: {action}, Parameters: {parameters}")
        # Extracting data
        step = {
            "step": step_number,
            "action": action
        }
        if parameters:
            step["parameters"] = parameters
            logging.info(step)
            steps.append(step)

    return steps
