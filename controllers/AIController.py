import logging
from typing import List, Dict, Optional, Tuple
from schemas.utils import RetrievedDocumentSchema
from llm.llm_config import get_generation_client, get_embedding_client
from llm.prompt_templates.template_parser import TemplateParser
from core.config import get_settings

logger = logging.getLogger(__name__)


class AIController:
    """
    AI Controller for RAG (Retrieval-Augmented Generation) operations.
    Handles retrieval of context from knowledge base and generation of RAG-enhanced responses.
    """
    
    def __init__(
        self,
        vectordb_client,
        embedding_client=None,
        generation_client=None,
        settings=None
    ):
        """
        Initialize AIController with required dependencies.
        
        Args:
            vectordb_client: Vector database client for semantic search
            embedding_client: Client for generating embeddings (optional, uses singleton if not provided)
            generation_client: Client for generating responses (optional, uses singleton if not provided)
            settings: Application settings (optional, uses get_settings() if not provided)
        """
        self.vectordb_client = vectordb_client
        self.embedding_client = embedding_client or get_embedding_client()
        self.generation_client = generation_client or get_generation_client()
        self.settings = settings or get_settings()
        self.template_parser = TemplateParser(language=self.settings.DEFAULT_LANGUAGE)
        self.logger = logging.getLogger(__name__)
    
    def get_embedding_size(self) -> Optional[int]:
        """
        Get the configured embedding size from the embedding client.
        
        Returns:
            Embedding dimension size, or None if not available
        """
        try:
            if hasattr(self.embedding_client, 'embedding_size'):
                return self.embedding_client.embedding_size
            elif hasattr(self.embedding_client, 'model_config'):
                config = self.embedding_client.model_config
                if isinstance(config, dict) and 'embedding_size' in config:
                    return config['embedding_size']
            # Fallback to settings
            return int(self.settings.EMBEDDING_MODEL_SIZE) if self.settings.EMBEDDING_MODEL_SIZE else None
        except Exception as e:
            self.logger.warning(f"Could not determine embedding size: {str(e)}")
            return None
    
    def validate_collection_dimension(self, collection_name: str) -> Tuple[bool, Optional[int], Optional[int]]:
        """
        Validate that the collection dimension matches the embedding client dimension.
        
        Args:
            collection_name: Name of the collection to validate
            
        Returns:
            Tuple of (is_valid, collection_dim, embedding_dim)
        """
        try:
            # Get embedding client dimension
            embedding_dim = self.get_embedding_size()
            if embedding_dim is None:
                self.logger.warning("Could not determine embedding client dimension")
                return False, None, None
            
            # Get collection dimension
            collection_info = self.vectordb_client.get_collection_info(collection_name)
            collection_dim = None
            
            if collection_info:
                # Log the type and available attributes for debugging
                self.logger.debug(f"Collection info type: {type(collection_info)}")
                self.logger.debug(f"Collection info attributes: {dir(collection_info)}")
                
                # Try multiple ways to access the vector size based on Qdrant client version
                try:
                    # Method 1: Direct access via config.params.size (older Qdrant versions)
                    if hasattr(collection_info, 'config'):
                        config = collection_info.config
                        self.logger.debug(f"Config type: {type(config)}, attributes: {dir(config)}")
                        
                        if hasattr(config, 'params'):
                            params = config.params
                            self.logger.debug(f"Params type: {type(params)}, attributes: {dir(params)}")
                            if hasattr(params, 'size'):
                                collection_dim = params.size
                                self.logger.debug(f"Found dimension via config.params.size: {collection_dim}")
                        
                        # Method 2: Access via config.vectors (newer Qdrant versions)
                        if collection_dim is None and hasattr(config, 'vectors'):
                            vectors_config = config.vectors
                            self.logger.debug(f"Vectors config type: {type(vectors_config)}")
                            
                            # Handle VectorParams object
                            if hasattr(vectors_config, 'size'):
                                collection_dim = vectors_config.size
                                self.logger.debug(f"Found dimension via vectors.size: {collection_dim}")
                            # Handle dict format
                            elif isinstance(vectors_config, dict):
                                if 'size' in vectors_config:
                                    collection_dim = vectors_config['size']
                                    self.logger.debug(f"Found dimension via vectors dict size: {collection_dim}")
                                elif len(vectors_config) > 0:
                                    # Named vectors case - get first vector config
                                    first_key = list(vectors_config.keys())[0]
                                    first_vector_config = vectors_config[first_key]
                                    if hasattr(first_vector_config, 'size'):
                                        collection_dim = first_vector_config.size
                                        self.logger.debug(f"Found dimension via named vector {first_key}: {collection_dim}")
                            # Handle tuple or other formats
                            elif isinstance(vectors_config, tuple) and len(vectors_config) > 0:
                                if hasattr(vectors_config[0], 'size'):
                                    collection_dim = vectors_config[0].size
                                    self.logger.debug(f"Found dimension via vectors tuple: {collection_dim}")
                    
                    # Method 3: Try direct attribute access (some versions)
                    if collection_dim is None:
                        if hasattr(collection_info, 'vectors_config'):
                            vc = collection_info.vectors_config
                            if hasattr(vc, 'size'):
                                collection_dim = vc.size
                                self.logger.debug(f"Found dimension via vectors_config.size: {collection_dim}")
                        
                        if collection_dim is None and hasattr(collection_info, 'vector_size'):
                            collection_dim = collection_info.vector_size
                            self.logger.debug(f"Found dimension via vector_size: {collection_dim}")
                    
                except Exception as extract_error:
                    self.logger.debug(f"Error extracting dimension: {str(extract_error)}")
            
            if collection_dim is None:
                self.logger.warning(
                    f"Could not determine collection dimension for '{collection_name}'. "
                    f"Collection info: {type(collection_info).__name__ if collection_info else 'None'}. "
                    f"Will attempt search anyway, but dimension mismatch errors may occur."
                )
                # Return True to allow search to proceed - let Qdrant API return the actual error
                return True, None, embedding_dim
            
            is_valid = collection_dim == embedding_dim
            if not is_valid:
                self.logger.error(
                    f"Dimension mismatch detected: Collection '{collection_name}' expects {collection_dim} dimensions, "
                    f"but embedding client produces {embedding_dim} dimensions. "
                    f"Please update EMBEDDING_MODEL_SIZE in settings to {collection_dim} or recreate the collection with dimension {embedding_dim}."
                )
            else:
                self.logger.debug(f"Dimension validation passed: {collection_dim} == {embedding_dim}")
            
            return is_valid, collection_dim, embedding_dim
            
        except Exception as e:
            self.logger.error(f"Error validating collection dimension: {str(e)}")
            # Return True to allow search attempt - dimension check will happen during actual search
            embedding_dim = self.get_embedding_size()
            return True, None, embedding_dim
    
    def retrieve_context(
        self,
        query: str,
        collection_name: str,
        max_results: int = 5
    ) -> List[RetrievedDocumentSchema]:
        """
        Retrieve relevant context documents from the knowledge base using semantic search.
        
        Args:
            query: User query string
            collection_name: Name of the Qdrant collection to search
            max_results: Maximum number of results to return
            
        Returns:
            List of RetrievedDocumentSchema objects containing retrieved documents
        """
        try:
            # Check if collection exists
            if not self.vectordb_client.is_collection_exist(collection_name=collection_name):
                self.logger.warning(f"Collection '{collection_name}' does not exist")
                return []
            
            # Validate dimensions before attempting search
            is_valid, collection_dim, embedding_dim = self.validate_collection_dimension(collection_name)
            
            # If we couldn't determine collection dimension, proceed with search
            # The Qdrant API will return a clear error if there's a dimension mismatch
            if not is_valid and collection_dim is not None and embedding_dim is not None:
                self.logger.error(
                    f"Cannot search collection '{collection_name}': dimension mismatch "
                    f"(collection: {collection_dim}, embedding: {embedding_dim}). "
                    f"Please update EMBEDDING_MODEL_SIZE setting to {collection_dim} or recreate the collection with dimension {embedding_dim}."
                )
                return []
            
            # Generate embedding for the query
            query_embedding = self.embedding_client.embed_text(query)
            if not query_embedding:
                self.logger.error("Failed to generate query embedding")
                return []
            
            # Validate actual embedding dimension matches expected
            actual_dim = len(query_embedding)
            if embedding_dim and actual_dim != embedding_dim:
                self.logger.error(
                    f"Embedding dimension mismatch: Expected {embedding_dim}, got {actual_dim}. "
                    f"Model '{self.embedding_client.embedding_model_id}' may produce different dimensions than configured."
                )
                return []
            
            # Search in vector database
            try:
                search_results = self.vectordb_client.search_by_vector(
                    collection_name=collection_name,
                    vector=query_embedding,
                    limit=max_results
                )
                
                if not search_results:
                    self.logger.info(f"No results found for query in collection '{collection_name}'")
                    return []
                
                self.logger.info(f"Retrieved {len(search_results)} documents from collection '{collection_name}'")
                return search_results
                
            except Exception as search_error:
                error_msg = str(search_error)
                # Try to extract dimension from error message
                # Qdrant error format: "expected dim: 768, got 1536" or in raw response: "expected dim: 768, got 1536"
                if "expected dim" in error_msg.lower() or "dimension error" in error_msg.lower():
                    import re
                    # Try to extract both expected and actual dimensions from error
                    expected_match = re.search(r'expected\s+dim[:\s]+(\d+)', error_msg, re.IGNORECASE)
                    got_match = re.search(r'got\s+(\d+)', error_msg, re.IGNORECASE)
                    
                    if expected_match:
                        expected_dim = int(expected_match.group(1))
                        got_dim = int(got_match.group(1)) if got_match else actual_dim
                        self.logger.error(
                            f"Dimension mismatch during search: Collection '{collection_name}' expects {expected_dim} dimensions, "
                            f"but embedding produced {got_dim} dimensions. "
                            f"Please set EMBEDDING_MODEL_SIZE={expected_dim} in your .env file to match the collection, "
                            f"or recreate the collection with dimension {got_dim}."
                        )
                    return []
                else:
                    # Re-raise if it's not a dimension error
                    raise
            
        except Exception as e:
            self.logger.error(f"Error retrieving context: {str(e)}")
            return []
    
    def build_rag_prompt(
        self,
        query: str,
        context_documents: List[RetrievedDocumentSchema],
        conversation_history: List[Dict[str, str]] = None
    ) -> List[Dict[str, str]]:
        """
        Build RAG-enhanced prompt with retrieved context and conversation history.
        
        Args:
            query: Current user query
            context_documents: List of retrieved documents from knowledge base
            conversation_history: Previous conversation messages (optional)
            
        Returns:
            List of message dictionaries formatted for chat API
        """
        messages = []
        conversation_history = conversation_history or []
        
        # Build context from retrieved documents
        context_parts = []
        if context_documents:
            for idx, doc in enumerate(context_documents, start=1):
                doc_content = self.template_parser.get_template(
                    group="rag",
                    key="document_prompt",
                    vars={
                        "doc_num": idx,
                        "chunk_text": doc.text
                    }
                )
                context_parts.append(doc_content)
        
        # Get system prompt
        system_content = self.template_parser.get_template(
            group="rag",
            key="system_prompt"
        )
        
        # Build user message with context
        if context_parts:
            context_section = "\n\n".join(context_parts)
            
            # Get footer prompt with question
            footer_content = self.template_parser.get_template(
                group="rag",
                key="footer_prompt",
                vars={"query": query}
            )
            
            user_content = f"{context_section}\n\n{footer_content}"
        else:
            # No context found, use query directly
            user_content = query
        
        # Add system message
        messages.append({
            "role": "system",
            "content": system_content
        })
        
        # Add conversation history (excluding system messages to avoid duplicates)
        for msg in conversation_history:
            if msg.get("role") != "system":
                messages.append(msg)
        
        # Add current user query with context
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        return messages
    
    async def generate_rag_response(
        self,
        query: str,
        collection_name: str,
        conversation_history: List[Dict[str, str]] = None,
        max_results: int = 5,
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> Tuple[str, List[RetrievedDocumentSchema]]:
        """
        Generate RAG-enhanced response by retrieving context and generating answer.
        
        Args:
            query: User query string
            collection_name: Name of the Qdrant collection to search
            conversation_history: Previous conversation messages (optional)
            max_results: Maximum number of documents to retrieve
            max_tokens: Maximum tokens for response generation
            temperature: Temperature for response generation
            
        Returns:
            Tuple of (response_text, retrieved_documents)
        """
        try:
            # Retrieve relevant context
            retrieved_docs = self.retrieve_context(
                query=query,
                collection_name=collection_name,
                max_results=max_results
            )
            
            # Build RAG prompt with context
            messages = self.build_rag_prompt(
                query=query,
                context_documents=retrieved_docs,
                conversation_history=conversation_history
            )
            
            # Generate response (async)
            response = await self.generation_client.chat(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            if not response:
                self.logger.error("Failed to generate response")
                return "I apologize, but I'm having trouble generating a response right now. Please try again.", retrieved_docs
            
            return response, retrieved_docs
            
        except Exception as e:
            self.logger.error(f"Error generating RAG response: {str(e)}")
            return "I apologize, but I'm having trouble generating a response right now. Please try again.", []
    
    def should_use_rag(self, collection_name: str) -> bool:
        """
        Check if RAG should be used based on collection existence and availability.
        
        Args:
            collection_name: Name of the collection to check
            
        Returns:
            True if RAG should be used, False otherwise
        """
        try:
            if not self.vectordb_client:
                return False
            return self.vectordb_client.is_collection_exist(collection_name=collection_name)
        except Exception as e:
            self.logger.warning(f"Error checking RAG availability: {str(e)}")
            return False

