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
        
    
    def connect(self, url : str, api_key : str, timeout : float = 60.0):
        try:
            self.client = QdrantClient(
                            url=url, 
                            api_key=api_key,
                            timeout=timeout,  # Increase timeout for cloud instances
                                )
            # Test the connection by trying to get collections
            collections = self.client.get_collections()
            self.logger.info(f"Qdrant Provider : Connected successfully. Found {len(collections.collections)} collections.")
        except Exception as e:
            self.logger.error(f"Qdrant Provider : Connection failed: {str(e)}")
            self.logger.error(f"URL: {url}")
            self.logger.error(f"API Key provided: {'Yes' if api_key else 'No'}")
            raise e
    
    def disconnect(self):
        self.client = None
        self.logger.info("Qdrant Provider : Disconnected")
    
    def is_collection_exist(self, collection_name : str) -> bool:
        try:
            return self.client.collection_exists(collection_name = collection_name)
        except Exception as e:
            self.logger.error(f"Error checking if collection exists: {str(e)}")
            # Try an alternative approach - list all collections and check
            try:
                collections = self.client.get_collections()
                collection_names = [col.name for col in collections.collections]
                exists = collection_name in collection_names
                self.logger.info(f"Alternative check - Collection '{collection_name}' exists: {exists}")
                return exists
            except Exception as e2:
                self.logger.error(f"Alternative collection check also failed: {str(e2)}")
                return False
    
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
                self.logger.info(f"Resetting collection '{collection_name}'")
                _ = self.delete_collection(collection_name=collection_name)
            
            collection_exists = self.is_collection_exist(collection_name=collection_name)
            self.logger.info(f"Collection '{collection_name}' exists: {collection_exists}")
            
            if not collection_exists:
                self.logger.info(f"Creating collection '{collection_name}' with embedding size: {embedding_size}")
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=embedding_size,
                        distance=self.distance_method,
                    )
                )
                self.logger.info(f"Successfully created collection '{collection_name}'")
                return True
            else:
                # If collection exists, verify its configuration
                self.logger.info(f"Collection '{collection_name}' exists, checking its configuration")
                try:
                    collection_info = self.get_collection_info(collection_name)
                    self.logger.info(f"Collection info retrieved successfully")
                    
                    # Access vector config - handle different Qdrant client versions
                    vector_size = None
                    if collection_info and hasattr(collection_info, 'config'):
                        config = collection_info.config
                        
                        # Try accessing via config.params.size (older versions)
                        if hasattr(config, 'params') and hasattr(config.params, 'size'):
                            vector_size = config.params.size
                        # Try accessing via config.vectors.size (newer versions)
                        elif hasattr(config, 'vectors'):
                            vectors_config = config.vectors
                            if isinstance(vectors_config, dict):
                                if 'size' in vectors_config:
                                    vector_size = vectors_config['size']
                                elif len(vectors_config) > 0:
                                    # For named vectors, get the first vector config
                                    first_vector_config = list(vectors_config.values())[0]
                                    if hasattr(first_vector_config, 'size'):
                                        vector_size = first_vector_config.size
                            elif hasattr(vectors_config, 'size'):
                                vector_size = vectors_config.size
                    
                    if vector_size is not None and vector_size != embedding_size:
                        self.logger.warning(f"Collection '{collection_name}' exists with different embedding size ({vector_size} vs {embedding_size}). Deleting and recreating.")
                        _ = self.delete_collection(collection_name=collection_name)
                        return self.create_collection(collection_name, embedding_size, do_reset=False)
                    
                    return False
                except Exception as info_error:
                    self.logger.error(f"Error getting collection info: {str(info_error)}")
                    # Collection exists but we can't get info, assume it's okay to use
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
                   record_ids : list = None, batch_size : int = 10, max_retries : int = 3):
        """
        Insert multiple records into Qdrant with retry logic and smaller batches.
        
        Args:
            collection_name: Name of the collection
            texts: List of text strings
            vectors: List of embedding vectors
            metadatas: List of metadata dictionaries
            record_ids: List of record IDs
            batch_size: Number of records per batch (default 10 for cloud instances)
            max_retries: Maximum number of retry attempts per batch (default 3)
        """
        import time
        
        if not self.is_collection_exist(collection_name = collection_name):
            self.logger.error(f"Qdrant Provider (Insert Many) : Collection '{collection_name}' does not exist.")
            return False
        
        if metadatas is None:
            metadatas = [None] * len(texts)
        
        if record_ids is None:
            record_ids = list(range(0, len(texts)))
        
        total_batches = (len(texts) + batch_size - 1) // batch_size
        self.logger.info(f"Uploading {len(texts)} records in {total_batches} batches of {batch_size}")
        
        for i in range(0, len(texts), batch_size):
            batch_end = min(i + batch_size, len(texts))
            batch_num = (i // batch_size) + 1

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

            # Retry logic with exponential backoff
            retry_count = 0
            success = False
            while retry_count < max_retries and not success:
                try:
                    self.logger.info(f"Uploading batch {batch_num}/{total_batches} (records {i}-{batch_end-1})...")
                    _ = self.client.upload_records(
                        collection_name = collection_name,
                        records = batch_records,
                    )
                    success = True
                    self.logger.info(f"✅ Successfully uploaded batch {batch_num}/{total_batches}")
                    
                except Exception as e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        self.logger.error(f"❌ Qdrant Provider (Insert Many) : Failed to upload batch {batch_num} after {max_retries} attempts: {str(e)}")
                        return False
                    else:
                        # Exponential backoff: wait 2^retry_count seconds
                        wait_time = 2 ** retry_count
                        self.logger.warning(f"⚠️ Batch {batch_num} upload failed (attempt {retry_count}/{max_retries}). Retrying in {wait_time}s... Error: {str(e)}")
                        time.sleep(wait_time)
            
            # Small delay between successful batches to avoid overwhelming the server
            if success and batch_num < total_batches:
                time.sleep(0.5)

        self.logger.info(f"✅ Successfully uploaded all {len(texts)} records to Qdrant")
        return True


    def search_by_vector(self, collection_name : str, vector : list, limit : int = 5):
        """
        Search for similar vectors in the collection.
        
        Args:
            collection_name: Name of the collection to search
            vector: Query vector (embedding)
            limit: Maximum number of results to return
            
        Returns:
            List of RetrievedDocumentSchema objects
        """
        if not self.is_collection_exist(collection_name = collection_name):
            self.logger.error(f"Qdrant Provider (Search by Vector) : Collection '{collection_name}' does not exist.")
            return []
        
        # Validate vector dimension
        if not vector or len(vector) == 0:
            self.logger.error("Empty vector provided for search.")
            return []
        
        try:
            # Get collection info to verify vector dimension
            # Note: We attempt dimension validation but proceed with search even if validation fails
            # This allows for different Qdrant client versions with varying collection info structures
            try:
                collection_info = self.get_collection_info(collection_name)
                expected_dim = None
                
                # Try different ways to access the vector size depending on Qdrant client version
                if collection_info and hasattr(collection_info, 'config'):
                    config = collection_info.config
                    
                    # Try accessing via config.params.size (older versions)
                    if hasattr(config, 'params') and hasattr(config.params, 'size'):
                        expected_dim = config.params.size
                    # Try accessing via config.vectors.size (newer versions)
                    elif hasattr(config, 'vectors'):
                        vectors_config = config.vectors
                        # Handle both dict and object access
                        if isinstance(vectors_config, dict):
                            if 'size' in vectors_config:
                                expected_dim = vectors_config['size']
                        elif hasattr(vectors_config, 'size'):
                            expected_dim = vectors_config.size
                        # Handle named vectors case
                        elif isinstance(vectors_config, dict) and len(vectors_config) > 0:
                            # For named vectors, get the first vector config
                            first_vector_config = list(vectors_config.values())[0]
                            if hasattr(first_vector_config, 'size'):
                                expected_dim = first_vector_config.size
                
                if expected_dim is not None:
                    actual_dim = len(vector)
                    if actual_dim != expected_dim:
                        self.logger.error(
                            f"Vector dimension mismatch: collection expects {expected_dim}, got {actual_dim}. "
                            f"This usually means the embedding model doesn't match the collection configuration."
                        )
                        return []
                    else:
                        self.logger.debug(f"Vector dimension validated: {actual_dim}")
            except Exception as validation_error:
                # Dimension validation failed, but we'll still attempt the search
                # This allows for cases where collection info structure differs
                self.logger.warning(f"Could not validate vector dimension: {str(validation_error)}. Proceeding with search anyway.")
            
            results = self.client.search(
                collection_name = collection_name,
                query_vector = vector,
                limit = limit
            )
        except Exception as e:
            self.logger.error(f"Error during vector search: {str(e)}")
            return []

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

