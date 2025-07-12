from fastapi import APIRouter, HTTPException, status, Depends, Request, Query
from fastapi.responses import JSONResponse
from typing import Optional, List
import logging

from controllers.ChatController import ChatController
from dtos.chat import (
    CreateSessionRequest, ChatRequest, SessionResponse, 
    ChatResponse, SessionWithMessagesResponse, SessionListResponse,
    MessageListResponse
)
from schemas.chat import SessionStatus

chat_session_router = APIRouter(prefix="/chat-session", tags=["Chat Sessions"])
logger = logging.getLogger(__name__)

def get_chat_controller(request: Request) -> ChatController:
    """Dependency to get chat controller"""
    if not hasattr(request.app, 'db_client'):
        raise HTTPException(status_code=500, detail="Database not initialized")
    
    controller = ChatController(request.app.db_client)
    return controller

@chat_session_router.on_event("startup")
async def initialize_chat_controller():
    """Initialize chat controller on startup"""
    logger.info("Initializing chat controller...")

# =============================================================================
# Session Management Endpoints
# =============================================================================

@chat_session_router.post("/sessions", summary="Create New Chat Session", response_model=SessionResponse)
async def create_session(
    request: CreateSessionRequest,
    controller: ChatController = Depends(get_chat_controller)
):
    """
    Create a new chat session
    """
    try:
        session = await controller.create_session(request)
        return session
        
    except Exception as e:
        logger.error(f"Error creating session: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating session: {str(e)}"
        )

@chat_session_router.get("/sessions/{session_id}", summary="Get Chat Session", response_model=SessionResponse)
async def get_session(
    session_id: str,
    controller: ChatController = Depends(get_chat_controller)
):
    """
    Get a specific chat session by ID
    """
    try:
        session = await controller.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session not found: {session_id}"
            )
        return session
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session {session_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting session: {str(e)}"
        )

@chat_session_router.get("/sessions/{session_id}/messages", summary="Get Session with Messages", response_model=SessionWithMessagesResponse)
async def get_session_with_messages(
    session_id: str,
    limit: Optional[int] = Query(None, description="Limit number of messages to return"),
    controller: ChatController = Depends(get_chat_controller)
):
    """
    Get a chat session with its messages
    """
    try:
        session_with_messages = await controller.get_session_with_messages(session_id, limit)
        if not session_with_messages:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session not found: {session_id}"
            )
        return session_with_messages
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session with messages {session_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting session with messages: {str(e)}"
        )

@chat_session_router.get("/sessions", summary="List Chat Sessions", response_model=SessionListResponse)
async def list_sessions(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
    controller: ChatController = Depends(get_chat_controller)
):
    """
    List chat sessions with pagination
    """
    try:
        sessions, total = await controller.list_sessions(user_id=user_id, page=page, page_size=page_size)
        
        has_next = (page * page_size) < total
        
        return SessionListResponse(
            sessions=sessions,
            total=total,
            page=page,
            page_size=page_size,
            has_next=has_next
        )
        
    except Exception as e:
        logger.error(f"Error listing sessions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing sessions: {str(e)}"
        )

@chat_session_router.delete("/sessions/{session_id}", summary="Delete Chat Session")
async def delete_session(
    session_id: str,
    controller: ChatController = Depends(get_chat_controller)
):
    """
    Delete a chat session and all its messages
    """
    try:
        success = await controller.delete_session(session_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session not found: {session_id}"
            )
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"message": f"Session {session_id} deleted successfully"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting session: {str(e)}"
        )

# =============================================================================
# Message Management Endpoints
# =============================================================================

@chat_session_router.get("/sessions/{session_id}/messages-list", summary="Get Session Messages", response_model=MessageListResponse)
async def get_session_messages(
    session_id: str,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Page size"),
    controller: ChatController = Depends(get_chat_controller)
):
    """
    Get messages for a session with pagination
    """
    try:
        # First check if session exists
        session = await controller.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session not found: {session_id}"
            )
        
        messages = await controller.get_session_messages(session_id, page, page_size)
        
        # For simplicity, we'll estimate if there are more pages
        # In a real application, you might want to count total messages
        has_next = len(messages) == page_size
        
        return MessageListResponse(
            messages=messages,
            total=len(messages),  # This is just the current page count
            page=page,
            page_size=page_size,
            has_next=has_next
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session messages {session_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting session messages: {str(e)}"
        )

# =============================================================================
# Main Chat Endpoint
# =============================================================================

@chat_session_router.post("/chat", summary="Send Message and Get AI Response", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    controller: ChatController = Depends(get_chat_controller)
):
    """
    Main chat endpoint - send a message and get an AI response
    This handles the core chat flow with session and message management
    """
    try:
        # Initialize controller (create indexes if needed)
        await controller.initialize()
        
        response = await controller.chat(request)
        
        if not response.success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=response.error or "Chat processing failed"
            )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat processing error: {str(e)}"
        )

# =============================================================================
# Health Check Endpoints
# =============================================================================

@chat_session_router.get("/health", summary="Health Check")
async def health_check(
    controller: ChatController = Depends(get_chat_controller)
):
    """
    Health check endpoint for chat session service
    """
    try:
        # Try to initialize controller to check database connectivity
        await controller.initialize()
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "healthy",
                "message": "Chat session service is running",
                "database": "connected"
            }
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy", 
                "message": f"Service error: {str(e)}",
                "database": "disconnected"
            }
        ) 