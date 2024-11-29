from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams
import logging as logger
import os

URL = os.getenv("QDRANT_URL")
API_KEY = os.getenv("QDRANT_API_KEY")

class QdrantDB():

    def __init__(self) -> None:
        ''' connects to the qdrant vector database'''
        try:
            self.client = QdrantClient(url=URL,api_key=API_KEY)
        except ConnectionError as e:
            logger.error(e)

    def create_collection(self,name):
        ''' create a collection in the vector database '''
        try:
            try:
                self.client.get_collection(collection_name=name)
                logger.info(f'Collection {name} already exists, skipping creation.')
                return {"status": "ok"} # Return success even if collection already exists
            except Exception as e:
                pass

            logger.info(f'creating a collection with name {name}')
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            return {"status": "ok"}
        except RuntimeError as e:
            logger.error(e)
            logger.error(f'can not create collection with {name}')

    def get(self,name,vector,top_k=5):
        ''' get a result from the vector databse using the given query'''
        try:
            logger.info(f'getting from collection with name {name} using query')
            search_result = self.client.search(
                collection_name=name, query_vector=vector, limit=top_k
            )
            logger.info("successfully got results from collection")
            return {"status":"ok","result":search_result[0].json()}
        except RuntimeError as e:
            logger.error(f'error executing query on collection {name}')
            logger.error(e)

    def add(self,id,payload,vector,name):
        ''' adds if the id does not exists or else uptates the existing
            recordd for the given collection'''
        try:
            logger.info(f'upserting {str(payload)} into vector database')
            self.client.upsert(
                collection_name=name,
                points=[
                    models.PointStruct(
                        id=id,
                        payload=payload,
                        vector=vector,
                    )
                ],
            )
            logger.info("upsert successfull")
        except RuntimeError as e:
            logger.error(f'error upserting {str(payload)} into vector database')
            logger.error(e)