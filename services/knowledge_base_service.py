import os
import logging
from typing import List, Optional
from fastapi import HTTPException
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llm.llm_config import get_generation_client, get_embedding_client
from core.config import get_settings
from schemas.knowledge_base_schemas import KnowledgeBase, Document, DataChunk
from dtos.knowledge_base import UploadDocumentRequest, UploadDocumentResponse, DataChunkDTO
from bson import ObjectId

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

class KnowledgeBaseMongoService:
    def __init__(self, db, vectordb_client, embedding_client):
        self.db = db
        self.vectordb_client = vectordb_client
        self.embedding_client = embedding_client
        self.knowledge_bases = db.knowledge_bases
        self.documents = db.documents
        self.data_chunks = db.data_chunks
        self.logger = logging.getLogger(__name__)

    async def get_knowledge_base(self, kb_id: str) -> Optional[KnowledgeBase]:
        doc = await self.knowledge_bases.find_one({"_id": ObjectId(kb_id)})
        return KnowledgeBase(**doc) if doc else None

    async def create_document(self, doc: Document) -> str:
        result = await self.documents.insert_one(doc.model_dump(by_alias=True, exclude={"id"}))
        return str(result.inserted_id)

    async def create_data_chunks(self, chunks: List[DataChunk]) -> int:
        if not chunks:
            return 0
        await self.data_chunks.insert_many([chunk.model_dump(by_alias=True, exclude={"id"}) for chunk in chunks])
        return len(chunks)

    async def ingest_document(self, kb_id: str, file_path: str, name: str, type_: str, description: Optional[str], chunk_size: int = 1200, overlap_size: int = 200) -> UploadDocumentResponse:
        # 1. Check KB exists
        kb = await self.get_knowledge_base(kb_id)
        if not kb:
            raise HTTPException(status_code=404, detail="Knowledge base not found")
        # 2. Create Document
        doc = Document(knowledge_base_id=kb_id, name=name, type=type_, description=description)
        doc_id = await self.create_document(doc)
        # 3. Extract and chunk
        from controllers.ProcessDataController import ProcessDataController
        process_ctrl = ProcessDataController(project_id=kb_id)  # project_id can be kb_id for now
        file_content = process_ctrl.get_file_content(file_id=file_path)
        if not file_content:
            raise HTTPException(status_code=400, detail="Failed to extract file content")
        chunks = process_ctrl.process_file_content(file_content, chunk_size=chunk_size, overlap_size=overlap_size)
        # 4. Store chunks in MongoDB
        data_chunks = []
        for idx, chunk in enumerate(chunks):
            data_chunks.append(DataChunk(document_id=doc_id, text_chunk=chunk.page_content, order=idx, metadata=chunk.metadata))
        await self.create_data_chunks(data_chunks)
        # 5. Embed and store in vector DB
        texts = [chunk.text_chunk for chunk in data_chunks]
        metadatas = [chunk.metadata for chunk in data_chunks]
        embeddings = [self.embedding_client.get_embedding(text) for text in texts]
        record_ids = list(range(len(texts)))
        collection_name = f"kb_{kb_id}"
        embedding_size = int(self.embedding_client.model_config.get('embedding_size', 1536))
        self.vectordb_client.create_collection(collection_name=collection_name, embedding_size=embedding_size, do_reset=False)
        success = self.vectordb_client.insert_many(collection_name=collection_name, texts=texts, vectors=embeddings, metadatas=metadatas, record_ids=record_ids)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to insert into vector DB")
        return UploadDocumentResponse(document_id=doc_id, knowledge_base_id=kb_id, chunk_count=len(data_chunks), vector_db_collection=collection_name, message="Document ingested successfully") 