import json
import ast
from langchain_community.embeddings import HuggingFaceEmbeddings
from database.vector import QdrantDB
from sentence_transformers import SentenceTransformer

model_name = "all-MiniLM-L6-v2"  # A small, efficient model from SentenceTransformers
model = SentenceTransformer(model_name)

local_embeddings = HuggingFaceEmbeddings(model_name=model_name)
vector_db = QdrantDB()

def get_cordinates_for_location(text):
  vector = local_embeddings.embed_query(text)
  data = vector_db.get("places",vector,1)
  location = json.loads(data['result'])['payload']['location']
  location = ast.literal_eval(location.replace("x", "'x'").replace("y", "'y'"))
  location = tuple(location.values())
  return location