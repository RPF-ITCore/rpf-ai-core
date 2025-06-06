from qdrant_client import models, QdrantClient
from ..VectorDBInterface import VectorDBInterface
from ..VectorDBEnums import VectorDBEnums, DistanceMethodEnums
from schemas import RetrievedDocumentSchema
import logging
from typing import List
from core.config import Settings, get_settings

class QdrantDBProvider(VectorDBInterface):
    def __init__(self, db_path : str, distance_method : str):

        self.db_path = db_path
        self.client = None  # We will initialize it in the Connect Method
        self.distance_method = None 

        self.logger = logging.getLogger(__name__)

        if distance_method == DistanceMethodEnums.COSINE.value:
            self.distance_method = models.Distance.COSINE

        elif distance_method == DistanceMethodEnums.DOT.value:
            self.distance_method = models.Distance.DOT
        
        self.config = get_settings()
        
    
    def connect(self, url : str, api_key : str):
        self.client = QdrantClient(
                        url=url, 
                        api_key=api_key,
                            )
        self.logger.info("Qdrant Provider : Connected")
    
    def disconnect(self):
        self.client = None
        self.logger.info("Qdrant Provider : Disconnected")
    
    def is_collection_exist(self, collection_name : str) -> bool:
        return self.client.collection_exists(collection_name = collection_name)
    
    def list_all_collections(self) -> List:
        return self.client.get_collections()
    
    def get_collection_info(self, collection_name : str) -> dict:
        return self.client.get_collection(collection_name = collection_name)
    
    def delete_collection(self, collection_name : str):
        if self.is_collection_exist(collection_name = collection_name):
            return self.client.delete_collection(collection_name = collection_name)
        else:
            self.logger.info(f"Qdrant Provider (delete_collection) : Collection '{collection_name}' does not exist already.")
    
    def create_collection(self, collection_name : str, embedding_size : int, do_reset : bool = False):
        try:
            if do_reset:
                _ = self.delete_collection(collection_name=collection_name)
            
            if not self.is_collection_exist(collection_name=collection_name):
                self.logger.info(f"Creating collection '{collection_name}' with embedding size: {embedding_size}")
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=embedding_size,
                        distance=self.distance_method,
                    )
                )
                return True
            else:
                # If collection exists, verify its configuration
                self.logger.info(f"Collection '{collection_name}' exists, checking its configuration")
                collection_info = self.get_collection_info(collection_name)
                self.logger.info(f"Collection info: {collection_info}")
                
                # Access vector config using object attributes instead of dictionary access
                if collection_info and hasattr(collection_info, 'config') and hasattr(collection_info.config, 'params'):
                    vector_size = collection_info.config.params.size
                    if vector_size != embedding_size:
                        self.logger.warning(f"Collection '{collection_name}' exists with different embedding size ({vector_size} vs {embedding_size}). Deleting and recreating.")
                        _ = self.delete_collection(collection_name=collection_name)
                        return self.create_collection(collection_name, embedding_size, do_reset=False)
                return False
        except Exception as e:
            self.logger.error(f"Error creating collection: {str(e)}")
            raise e
        

    def insert_one(self, collection_name : str, text : str, vector : list,
                   metadata : dict = None,
                   record_id : int = None): # in Qdrant DB we don't need to use record_id
        
        if not self.is_collection_exist(collection_name = collection_name):
            self.logger.error(f"Qdrant Provider (Insert One) : Collection '{collection_name}' does not exist.")
            return False
        
        try:
            _ = self.client.upload_records(
                collection_name = collection_name,
                records = [
                    models.Record(
                        id = record_id,
                        vector = vector,
                        payload={
                            "text" : text,
                            "metadata" : metadata,
                        }
                    )
                ],
            )
        
        except Exception as e:
            self.logger.error(f"Qdrant Provider (Insert One) : Error inserting record: {str(e)}")
            return False
        

        return True
    

    def insert_many(self, collection_name : str, texts : list, vectors : list,
                   metadatas :  list = None,
                   record_ids : list = None, batch_size : int = 50):
        
        if not self.is_collection_exist(collection_name = collection_name):
            self.logger.error(f"Qdrant Provider (Insert Many) : Collection '{collection_name}' does not exist.")
            return False
        
        if metadatas is None:
            metadatas = [None] * len(texts)
        
        if record_ids is None:
            record_ids = list(range(0, len(texts)))
        
        for i in range(0, len(texts), batch_size):
            batch_end = i + batch_size

            batch_texts = texts[i:batch_end]
            batch_vectors = vectors[i:batch_end]
            batch_metadatas = metadatas[i:batch_end]
            batch_record_ids = record_ids[i:batch_end]

            batch_records = [

                models.Record(
                    id = batch_record_ids[x],
                    vector = batch_vectors[x],
                    payload={
                        "text" : batch_texts[x],
                        "metadata" : batch_metadatas[x],
                    }
                )

                for x in range(len(batch_texts))
            ]

            try:
                _ = self.client.upload_records(
                    collection_name = collection_name,
                    records = batch_records,
                )
            
            except Exception as e:
                self.logger.error(f"Qdrant Provider (Insert Many) : Error during batch insertion: {str(e)}")
                return False

        return True


    def search_by_vector(self, collection_name : str, vector : list, limit : int = 5):
        
        if not self.is_collection_exist(collection_name = collection_name):
            self.logger.error(f"Qdrant Provider (Search by Vector) : Collection '{collection_name}' does not exist.")
            return []
        
        
        results = self.client.search(
            collection_name = collection_name,
            query_vector = vector,
            limit = limit
        )

        if results is None or len(results) == 0:
            self.logger.error(f"(Search by Vector) returned no results")
            return []

        return [
            RetrievedDocumentSchema(**{
                "text" : result.payload["text"],
                "score" : result.score
                }
            )
            for result in results
        ]

