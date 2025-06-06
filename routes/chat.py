from fastapi import APIRouter, HTTPException, status, Depends, Request
from fastapi.responses import JSONResponse
from services import KnowledgeBaseService
from schemas import ChatRequestSchema, ChatResponseSchema, KnowledgeBaseIngestSchema
import os
import logging

chat_router = APIRouter(prefix="/chat", tags=["Chat"])
logger = logging.getLogger(__name__)

def get_kb_service(request: Request) -> KnowledgeBaseService:
    """Dependency to get knowledge base service"""
    if not hasattr(request.app, 'vectordb_client'):
        raise HTTPException(status_code=500, detail="Vector database not initialized")
    return KnowledgeBaseService(request.app.vectordb_client)

@chat_router.post("/ingest-syria-kb", summary="Ingest Syria Knowledge Base", status_code=status.HTTP_200_OK)
async def ingest_syria_knowledge_base(
    request: KnowledgeBaseIngestSchema,
    kb_service: KnowledgeBaseService = Depends(get_kb_service)
):
    """
    Ingest the Syria knowledge base file into the vector database
    """
    try:
        # Path to the Syria knowledge base file
        file_path = os.path.join("assets", "Knowledge Base Report Syria.txt")
        
        result = kb_service.ingest_file(
            file_path=file_path,
            collection_name=request.collection_name,
            chunk_size=request.chunk_size,
            overlap_size=request.overlap_size,
            reset_collection=request.reset_collection
        )
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in ingest_syria_knowledge_base: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while ingesting the knowledge base: {str(e)}"
        )

@chat_router.post("/query", summary="Chat with Knowledge Base", response_model=ChatResponseSchema)
async def chat_with_knowledge_base(
    request: ChatRequestSchema,
    kb_service: KnowledgeBaseService = Depends(get_kb_service)
):
    """
    Chat with the Syria knowledge base
    """
    try:
        result = kb_service.query_knowledge_base(
            query=request.message,
            collection_name=request.collection_name,
            max_results=request.max_results
        )
        
        return ChatResponseSchema(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat_with_knowledge_base: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing your query: {str(e)}"
        )

@chat_router.get("/collection/{collection_name}/info", summary="Get Collection Info")
async def get_collection_info(
    collection_name: str,
    kb_service: KnowledgeBaseService = Depends(get_kb_service)
):
    """
    Get information about a collection
    """
    try:
        result = kb_service.get_collection_info(collection_name)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_collection_info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while getting collection info: {str(e)}"
        ) 