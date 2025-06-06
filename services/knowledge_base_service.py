import os
import logging
from typing import List
from fastapi import HTTPException
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llm.llm_config import get_generation_client, get_embedding_client
from core.config import get_settings

class KnowledgeBaseService:
    def __init__(self, vectordb_client):
        self.vectordb_client = vectordb_client
        self.generation_client = get_generation_client()
        self.embedding_client = get_embedding_client()
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)

    def ingest_file(self, file_path: str, collection_name: str, 
                   chunk_size: int = 1200, overlap_size: int = 200, 
                   reset_collection: bool = False) -> dict:
        """
        Ingest a file into the knowledge base
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

            # Load the document
            loader = TextLoader(file_path=file_path)
            documents = loader.load()
            
            if not documents:
                raise HTTPException(status_code=400, detail="No content found in the file")

            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap_size,
                length_function=len
            )
            
            chunks = text_splitter.split_documents(documents)
            self.logger.info(f"Split document into {len(chunks)} chunks")

            # Get embedding size from the embedding client
            embedding_size = int(self.settings.EMBEDDING_MODEL_SIZE)
            
            # Create collection
            collection_created = self.vectordb_client.create_collection(
                collection_name=collection_name,
                embedding_size=embedding_size,
                do_reset=reset_collection
            )
            
            if collection_created:
                self.logger.info(f"Created new collection: {collection_name}")
            else:
                self.logger.info(f"Using existing collection: {collection_name}")

            # Generate embeddings and store chunks
            texts = [chunk.page_content for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            
            # Generate embeddings in batches
            batch_size = 10
            total_chunks = len(texts)
            
            for i in range(0, total_chunks, batch_size):
                batch_end = min(i + batch_size, total_chunks)
                batch_texts = texts[i:batch_end]
                batch_metadatas = metadatas[i:batch_end]
                
                # Generate embeddings for this batch
                embeddings = []
                for text in batch_texts:
                    embedding = self.embedding_client.get_embedding(text)
                    embeddings.append(embedding)
                
                # Generate record IDs
                record_ids = list(range(i, batch_end))
                
                # Insert batch into vector database
                success = self.vectordb_client.insert_many(
                    collection_name=collection_name,
                    texts=batch_texts,
                    vectors=embeddings,
                    metadatas=batch_metadatas,
                    record_ids=record_ids
                )
                
                if not success:
                    raise HTTPException(status_code=500, detail=f"Failed to insert batch {i}-{batch_end}")
                
                self.logger.info(f"Inserted batch {i}-{batch_end} of {total_chunks}")

            return {
                "message": "File ingested successfully",
                "collection_name": collection_name,
                "chunks_count": len(chunks),
                "file_path": file_path
            }

        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Error ingesting file: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error ingesting file: {str(e)}")

    def query_knowledge_base(self, query: str, collection_name: str, max_results: int = 5) -> dict:
        """
        Query the knowledge base and generate a response
        """
        try:
            # Generate embedding for the query
            query_embedding = self.embedding_client.get_embedding(query)
            
            # Search in vector database
            search_results = self.vectordb_client.search_by_vector(
                collection_name=collection_name,
                vector=query_embedding,
                limit=max_results
            )
            
            if not search_results:
                return {
                    "response": "I don't have information about that topic in my knowledge base.",
                    "sources": [],
                    "confidence_scores": []
                }

            # Extract context from search results
            context_texts = [result.text for result in search_results]
            confidence_scores = [result.score for result in search_results]
            
            # Create context for the generation model
            context = "\n\n".join(context_texts)
            
            # Create prompt for generation
            prompt = f"""Based on the following context about Syria, please answer the user's question accurately and comprehensively.

Context:
{context}

Question: {query}

Answer:"""

            # Generate response
            response = self.generation_client.generate_content(prompt)
            
            return {
                "response": response,
                "sources": context_texts[:3],  # Return top 3 sources
                "confidence_scores": confidence_scores[:3]
            }

        except Exception as e:
            self.logger.error(f"Error querying knowledge base: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error querying knowledge base: {str(e)}")

    def get_collection_info(self, collection_name: str) -> dict:
        """
        Get information about a collection
        """
        try:
            if not self.vectordb_client.is_collection_exist(collection_name):
                raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")
            
            info = self.vectordb_client.get_collection_info(collection_name)
            return {
                "collection_name": collection_name,
                "exists": True,
                "info": info
            }
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Error getting collection info: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error getting collection info: {str(e)}") 