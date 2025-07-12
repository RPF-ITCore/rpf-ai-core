import logging
import uuid
from typing import Optional, List
from motor.motor_asyncio import AsyncIOMotorDatabase

from services.chat import ChatService
from llm.llm_config import get_generation_client
from schemas.chat import MessageRole, MessageType
from dtos.chat import (
    CreateSessionRequest, CreateMessageRequest, ChatRequest,
    SessionResponse, MessageResponse, ChatResponse, SessionWithMessagesResponse
)

logger = logging.getLogger(__name__)

class ChatController:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.chat_service = ChatService(db)
        self.generation_client = get_generation_client()
        
    async def initialize(self):
        """Initialize the controller and create necessary indexes"""
        await self.chat_service.create_indexes()
        
    async def create_session(self, request: CreateSessionRequest) -> SessionResponse:
        """Create a new chat session"""
        try:
            session = await self.chat_service.create_session(request)
            logger.info(f"Created new chat session: {session.id}")
            return session
        except Exception as e:
            logger.error(f"Error creating chat session: {str(e)}")
            raise
    
    async def get_session(self, session_id: str) -> Optional[SessionResponse]:
        """Get a chat session by ID"""
        try:
            session = await self.chat_service.get_session(session_id)
            if not session:
                logger.warning(f"Session not found: {session_id}")
            return session
        except Exception as e:
            logger.error(f"Error getting session {session_id}: {str(e)}")
            raise
    
    async def get_session_with_messages(self, session_id: str, limit: Optional[int] = None) -> Optional[SessionWithMessagesResponse]:
        """Get a session with its messages"""
        try:
            session_with_messages = await self.chat_service.get_session_with_messages(session_id, limit)
            if not session_with_messages:
                logger.warning(f"Session not found: {session_id}")
            return session_with_messages
        except Exception as e:
            logger.error(f"Error getting session with messages {session_id}: {str(e)}")
            raise
    
    async def list_sessions(self, user_id: Optional[str] = None, page: int = 1, page_size: int = 20) -> tuple[List[SessionResponse], int]:
        """List chat sessions with pagination"""
        try:
            sessions, total = await self.chat_service.list_sessions(
                user_id=user_id, 
                page=page, 
                page_size=page_size
            )
            return sessions, total
        except Exception as e:
            logger.error(f"Error listing sessions: {str(e)}")
            raise
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a chat session and all its messages"""
        try:
            success = await self.chat_service.delete_session(session_id)
            if success:
                logger.info(f"Deleted session: {session_id}")
            else:
                logger.warning(f"Failed to delete session: {session_id}")
            return success
        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {str(e)}")
            raise
    
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """
        Main chat method - handles user message and generates AI response
        This is the core method for the chat flow
        """
        try:
            session_id = request.session_id
            
            # Create new session if not provided or if explicitly requested
            if not session_id or request.create_new_session:
                create_session_req = CreateSessionRequest(
                    user_id=request.user_id,
                    title=f"Chat Session - {request.message[:30]}..."
                )
                session = await self.create_session(create_session_req)
                session_id = session.id
                logger.info(f"Created new session for chat: {session_id}")
            
            # Verify session exists
            session = await self.get_session(session_id)
            if not session:
                raise Exception(f"Session not found: {session_id}")
            
            # Store user message
            user_message_req = CreateMessageRequest(
                session_id=session_id,
                role=MessageRole.USER,
                content=request.message,
                message_type=MessageType.TEXT
            )
            user_message = await self.chat_service.create_message(user_message_req)
            
            # Generate AI response
            ai_response_content = await self._generate_ai_response(session_id)
            
            # Store AI response
            ai_message_req = CreateMessageRequest(
                session_id=session_id,
                role=MessageRole.ASSISTANT,
                content=ai_response_content,
                message_type=MessageType.TEXT
            )
            ai_message = await self.chat_service.create_message(ai_message_req)
            
            logger.info(f"Chat completed for session: {session_id}")
            
            return ChatResponse(
                session_id=session_id,
                message=user_message,
                assistant_response=ai_message,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            return ChatResponse(
                session_id=session_id or "unknown",
                success=False,
                error=str(e)
            )
    
    async def _generate_ai_response(self, session_id: str) -> str:
        """
        Generate AI response based on chat history
        """
        try:
            # Get recent messages for context (last 10 messages)
            recent_messages = await self.chat_service.get_recent_messages(session_id, count=10)
            
            # Convert to LLM format
            llm_messages = []
            
            # Add system message
            llm_messages.append({
                "role": "system",
                "content": "You are a helpful AI assistant. Provide accurate, helpful, and engaging responses to user questions."
            })
            
            # Add conversation history
            for msg in recent_messages:
                if msg.role == MessageRole.USER:
                    llm_messages.append({
                        "role": "user",
                        "content": msg.content
                    })
                elif msg.role == MessageRole.ASSISTANT:
                    llm_messages.append({
                        "role": "assistant", 
                        "content": msg.content
                    })
            
            # Generate response using LLM
            response = await self.generation_client.chat(
                messages=llm_messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating AI response: {str(e)}")
            return "I apologize, but I'm having trouble generating a response right now. Please try again."
    
    async def get_session_messages(self, session_id: str, page: int = 1, page_size: int = 50) -> List[MessageResponse]:
        """Get messages for a session with pagination"""
        try:
            messages = await self.chat_service.get_session_messages(
                session_id=session_id,
                page=page,
                page_size=page_size
            )
            return messages
        except Exception as e:
            logger.error(f"Error getting session messages {session_id}: {str(e)}")
            raise
