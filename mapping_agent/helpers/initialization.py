from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from agent.invoker import invoke
from agent.parser import parse
import json
import logging

logging.basicConfig(level=logging.INFO)

def initialize(prompt_text,request):
    # Log the input prompt_text
    logging.info(f"Prompt Text: {prompt_text}")

    # Use the prompt template for the input prompt text
    template = prompt_text
    prompt = ChatPromptTemplate.from_template(template)

    # Initialize the model using Ollama
    model = OllamaLLM(model="mistral")

    # Create the chain from prompt to model
    chain = prompt | model

    # Invoke the model chain and get the response
    response = chain.invoke({"prompt": json.dumps({"prompt": request.prompt})})

    # Log the model response
    logging.info(f"Model Response: {response}")

    # Parse the response to extract steps
    steps = parse(response)

    logging.info(steps)

    # Iterate over each step in the parsed response
    for step in steps:
        # Check if the step contains parameters
        if "parameters" in step.keys():
            # Extract the parameters and function name
            arguments = json.loads(step["parameters"])
            if arguments['prompt']:
                arguments = arguments['prompt']
                
            function_name = step["action"]

            # If the arguments are a dictionary, pass them as keyword arguments
            if isinstance(arguments, dict):
                result = invoke(function_name, **arguments)
            if isinstance(arguments, str):
                result = invoke(function_name, arguments)
            else:
                # Otherwise pass them as positional arguments
                result = invoke(function_name, *arguments)
            logging.info(f"Result from invoking {function_name} with parameters: {result}")
            return result  # Assuming we return the first result
        else:
            # Call function without parameters
            function_name = step["action"]
            result = invoke(function_name)
            logging.info(f"Result from invoking {function_name} without parameters: {result}")
            return result  # Assuming we return the first result
