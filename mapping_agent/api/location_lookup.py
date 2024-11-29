from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from helpers.initialization import initialize
import logging
from helpers.calculate_path import calculate_path

logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app
app = FastAPI()

# Pydantic model for the input prompt
class Prompt(BaseModel):
    prompt: str  # Input field that contains the prompt text

def generate_prompt() -> str:
    """
    Generates a formatted prompt based on the input text.

    Args:
        prompt_text (str): The input text to be formatted into the template.

    Returns:
        str: The formatted prompt.
    """
    PROMPT_TEMPLATE = """Imagine that you are a robot navigating a warehouse. Your goal is to find the optimal path between the starting and ending locations. Follow the steps below to find the optimal path, and always respond in the format specified.

    Steps:

        Get the coordinates for the given locations.
            Step: 1
            Action: GetCordinates (parameters: {prompt})

        Calculate the optimal path between these locations.
            Step: 2
            Action: GetOptimalPath

    Response Format: For each step, provide the response in the exact format below:

        step: [Step Number]
        action: [Action Name] (parameters: {prompt})

    Always ensure that each step is clearly labeled with "step:" and "action:" as shown."""

    # Inject the value of 'prompt' into the template
    return PROMPT_TEMPLATE


@app.post("/prompt")
async def read_item(prompt: Prompt):
    """
    Endpoint to take a prompt from the UI and generate a response.

    Args:
        prompt (Prompt): The input prompt as a Pydantic model.

    Returns:
        dict: The generated prompt in JSON format.
    """
    try:
        template = generate_prompt()
        locations = initialize(template,prompt)
        path = calculate_path(locations)
        return path
    
    except Exception as e:
        # Handle any unexpected errors and return an HTTP 400 response
        logging.error(f"Error generating prompt: {e}")
        raise HTTPException(status_code=400, detail="Failed to generate the prompt.")
