from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from utils.location_utils import get_cordinates_for_location
import json

import logging

logging.basicConfig(level=logging.INFO)

DEFAULT = ""

def get_cordinates(prompt_text=DEFAULT):
    template = """
    Extract the 'from' and 'to' locations from the following text
            {text}. Return the result in JSON format with keys 'from' and 'to'.
    """

    prompt = ChatPromptTemplate.from_template(template)
    model = OllamaLLM(model="mistral")

    chain = prompt | model

    response = chain.invoke({"text": prompt_text})

    logging.info(response)

    response = json.loads(response)
    
    from_location = get_cordinates_for_location(response['from'])
    to_location = get_cordinates_for_location(response['to'])

    return from_location,to_location

