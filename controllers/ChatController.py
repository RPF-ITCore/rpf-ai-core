import logging
from datetime import datetime
import uuid
from typing import Optional, List
from motor.motor_asyncio import AsyncIOMotorDatabase

from services.chat import ChatService
from llm.llm_config import get_generation_client, get_embedding_client
from schemas.chat import MessageRole, MessageType
from dtos.chat import (
    CreateSessionRequest, CreateMessageRequest, ChatRequest,
    SessionResponse, MessageResponse, ChatResponse, SessionWithMessagesResponse,
    UpdateSessionRequest
)
from controllers.AIController import AIController
from core.config import get_settings

logger = logging.getLogger(__name__)

class ChatController:
    def __init__(
        self,
        db: AsyncIOMotorDatabase,
        vectordb_client=None,
        embedding_client=None
    ):
        self.db = db
        self.chat_service = ChatService(db)
        self.generation_client = get_generation_client()
        self.vectordb_client = vectordb_client
        self.embedding_client = embedding_client or get_embedding_client()
        self.settings = get_settings()
        
        # Initialize AI Controller for RAG if vector DB is available
        self.ai_controller = None
        if self.vectordb_client:
            try:
                self.ai_controller = AIController(
                    vectordb_client=self.vectordb_client,
                    embedding_client=self.embedding_client,
                    generation_client=self.generation_client,
                    settings=self.settings
                )
                logger.info("AIController initialized for RAG support")
            except Exception as e:
                logger.warning(f"Failed to initialize AIController: {str(e)}")
                self.ai_controller = None
        
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

    async def update_session(self, session_id: str, request: UpdateSessionRequest) -> Optional[SessionResponse]:
        """Update a chat session (e.g., rename title, change status, metadata)"""
        try:
            updated = await self.chat_service.update_session(session_id, request)
            if not updated:
                logger.warning(f"Session not found or not updated: {session_id}")
            return updated
        except Exception as e:
            logger.error(f"Error updating session {session_id}: {str(e)}")
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
                    title=f"New Chat - {datetime.utcnow().strftime('%Y-%m-%d')}"
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
        Generate AI response based on chat history with RAG enhancement.
        Falls back to non-RAG response if RAG is unavailable.
        """
        try:
            # Get recent messages for context (last 5 messages)
            recent_messages = await self.chat_service.get_recent_messages(session_id, count=5)
            
            if not recent_messages:
                logger.warning("No recent messages found")
                return "I apologize, but I'm having trouble generating a response right now. Please try again."
            
            # Extract the most recent user message (current query)
            # Messages are in chronological order, so the last one is most recent
            current_user_message = None
            conversation_history = []
            
            # Process messages in reverse to find the current query (most recent user message)
            found_current_query = False
            for msg in reversed(recent_messages):
                if not found_current_query and msg.role == MessageRole.USER:
                    # This is the current query
                    current_user_message = msg.content
                    found_current_query = True
                else:
                    # These are part of conversation history
                    if msg.role == MessageRole.USER:
                        conversation_history.insert(0, {
                            "role": "user",
                            "content": msg.content
                        })
                    elif msg.role == MessageRole.ASSISTANT:
                        conversation_history.insert(0, {
                            "role": "assistant",
                            "content": msg.content
                        })
            
            if not current_user_message:
                # Fallback: use the last message as query if it's not a user message
                logger.warning("No user message found, using last message as query")
                current_user_message = recent_messages[-1].content if recent_messages else ""
            
            # Try to use RAG if available
            if self.ai_controller:
                try:
                    # Get collection name from settings
                    collection_name = self.settings.RPF_KB_COLLECTION_NAME
                    
                    # Check if RAG should be used
                    if self.ai_controller.should_use_rag(collection_name):
                        logger.info(f"Using RAG with collection '{collection_name}'")
                        
                        # Generate RAG-enhanced response
                        response, retrieved_docs = await self.ai_controller.generate_rag_response(
                            query=current_user_message,
                            collection_name=collection_name,
                            conversation_history=conversation_history,
                            max_results=5,
                            max_tokens=1000,
                            temperature=0.7
                        )
                        
                        if retrieved_docs:
                            logger.info(f"RAG response generated with {len(retrieved_docs)} retrieved documents")
                        else:
                            logger.info("RAG response generated but no documents retrieved")
                        
                        return response
                    else:
                        logger.info(f"RAG collection '{collection_name}' not available, falling back to non-RAG")
                except Exception as e:
                    logger.warning(f"RAG generation failed, falling back to non-RAG: {str(e)}")
            
            # Fallback to non-RAG response
            logger.info("Using non-RAG response generation")
            
            # Convert to LLM format
            llm_messages = []
            
            # Add system message
            llm_messages.append({
                "role": "system",
                "content": (
                    "You are a professional AI assistant. Be clear, accurate, and helpful. "
                    "Prefer concise answers; include specifics when useful. If unsure, say so rather than guessing."
                )
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
