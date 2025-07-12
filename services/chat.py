from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import DESCENDING
import logging

from schemas.chat import ChatSession, ChatMessage, SessionStatus, MessageRole
from dtos.chat import (
    CreateSessionRequest, UpdateSessionRequest, CreateMessageRequest,
    SessionResponse, MessageResponse, SessionWithMessagesResponse
)


logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.sessions_collection = db.chat_sessions
        self.messages_collection = db.chat_messages
        
    async def create_indexes(self):
        """Create necessary database indexes for performance"""
        try:
            # Session indexes
            await self.sessions_collection.create_index("user_id")
            await self.sessions_collection.create_index("status")
            await self.sessions_collection.create_index("created_at")
            await self.sessions_collection.create_index("last_activity")
            
            # Message indexes
            await self.messages_collection.create_index("session_id")
            await self.messages_collection.create_index([("session_id", 1), ("created_at", 1)])
            await self.messages_collection.create_index("created_at")
            
            logger.info("Chat service indexes created successfully")
        except Exception as e:
            logger.error(f"Error creating chat service indexes: {str(e)}")

    # Session CRUD Operations
    async def create_session(self, request: CreateSessionRequest) -> SessionResponse:
        """Create a new chat session"""
        try:
            session = ChatSession(
                user_id=request.user_id,
                title=request.title or f"Chat Session {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
                metadata=request.metadata or {}
            )
            
            result = await self.sessions_collection.insert_one(session.model_dump(by_alias=True, exclude={"id"}))
            
            # Fetch the created session
            created_session = await self.sessions_collection.find_one({"_id": result.inserted_id})
            
            # Convert MongoDB document to SessionResponse format
            session_data = {
                "id": str(created_session["_id"]),
                **{k: v for k, v in created_session.items() if k != "_id"}
            }
            return SessionResponse(**session_data)
            
        except Exception as e:
            logger.error(f"Error creating session: {str(e)}")
            raise

    async def get_session(self, session_id: str) -> Optional[SessionResponse]:
        """Get a session by ID"""
        try:
            session = await self.sessions_collection.find_one({"_id": ObjectId(session_id)})
            if session:
                session_data = {
                    "id": str(session["_id"]),
                    **{k: v for k, v in session.items() if k != "_id"}
                }
                return SessionResponse(**session_data)
            return None
        except Exception as e:
            logger.error(f"Error getting session {session_id}: {str(e)}")
            return None

    async def get_session_with_messages(self, session_id: str, limit: Optional[int] = None) -> Optional[SessionWithMessagesResponse]:
        """Get a session with its messages"""
        try:
            session = await self.get_session(session_id)
            if not session:
                return None
                
            messages = await self.get_session_messages(session_id, limit=limit)
            
            return SessionWithMessagesResponse(
                **session.model_dump(),
                messages=messages
            )
        except Exception as e:
            logger.error(f"Error getting session with messages {session_id}: {str(e)}")
            return None

    async def update_session(self, session_id: str, request: UpdateSessionRequest) -> Optional[SessionResponse]:
        """Update a session"""
        try:
            update_data = {k: v for k, v in request.model_dump().items() if v is not None}
            update_data["updated_at"] = datetime.utcnow()
            
            result = await self.sessions_collection.update_one(
                {"_id": ObjectId(session_id)},
                {"$set": update_data}
            )
            
            if result.modified_count > 0:
                return await self.get_session(session_id)
            return None
            
        except Exception as e:
            logger.error(f"Error updating session {session_id}: {str(e)}")
            return None

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its messages"""
        try:
            # Delete messages first
            await self.messages_collection.delete_many({"session_id": session_id})
            
            # Delete session
            result = await self.sessions_collection.delete_one({"_id": ObjectId(session_id)})
            return result.deleted_count > 0
            
        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {str(e)}")
            return False

    async def list_sessions(self, user_id: Optional[str] = None, status: Optional[SessionStatus] = None, 
                           page: int = 1, page_size: int = 20) -> tuple[List[SessionResponse], int]:
        """List sessions with pagination"""
        try:
            filter_query = {}
            if user_id:
                filter_query["user_id"] = user_id
            if status:
                filter_query["status"] = status.value
                
            # Count total documents
            total = await self.sessions_collection.count_documents(filter_query)
            
            # Get paginated results
            skip = (page - 1) * page_size
            cursor = self.sessions_collection.find(filter_query).sort("last_activity", DESCENDING).skip(skip).limit(page_size)
            
            sessions = []
            async for session in cursor:
                session_data = {
                    "id": str(session["_id"]),
                    **{k: v for k, v in session.items() if k != "_id"}
                }
                sessions.append(SessionResponse(**session_data))
                
            return sessions, total
            
        except Exception as e:
            logger.error(f"Error listing sessions: {str(e)}")
            return [], 0

    # Message CRUD Operations
    async def create_message(self, request: CreateMessageRequest) -> MessageResponse:
        """Create a new message"""
        try:
            message = ChatMessage(
                session_id=request.session_id,
                role=request.role,
                content=request.content,
                message_type=request.message_type,
                metadata=request.metadata or {},
                tool_calls=request.tool_calls
            )
            
            result = await self.messages_collection.insert_one(message.model_dump(by_alias=True, exclude={"id"}))
            
            # Update session message count and last activity
            await self.sessions_collection.update_one(
                {"_id": ObjectId(request.session_id)},
                {
                    "$inc": {"message_count": 1},
                    "$set": {"last_activity": datetime.utcnow(), "updated_at": datetime.utcnow()}
                }
            )
            
            # Fetch the created message
            created_message = await self.messages_collection.find_one({"_id": result.inserted_id})
            
            # Convert MongoDB document to MessageResponse format
            message_data = {
                "id": str(created_message["_id"]),
                **{k: v for k, v in created_message.items() if k != "_id"}
            }
            return MessageResponse(**message_data)
            
        except Exception as e:
            logger.error(f"Error creating message: {str(e)}")
            raise

    async def get_session_messages(self, session_id: str, limit: Optional[int] = None, 
                                  page: int = 1, page_size: int = 50) -> List[MessageResponse]:
        """Get messages for a session"""
        try:
            query = {"session_id": session_id}
            cursor = self.messages_collection.find(query).sort("created_at", 1)
            
            if limit:
                cursor = cursor.limit(limit)
            else:
                skip = (page - 1) * page_size
                cursor = cursor.skip(skip).limit(page_size)
            
            messages = []
            async for message in cursor:
                message_data = {
                    "id": str(message["_id"]),
                    **{k: v for k, v in message.items() if k != "_id"}
                }
                messages.append(MessageResponse(**message_data))
                
            return messages
            
        except Exception as e:
            logger.error(f"Error getting session messages {session_id}: {str(e)}")
            return []

    async def get_recent_messages(self, session_id: str, count: int = 10) -> List[MessageResponse]:
        """Get recent messages for context (useful for LangChain/LangGraph)"""
        try:
            cursor = self.messages_collection.find({"session_id": session_id}).sort("created_at", DESCENDING).limit(count)
            
            messages = []
            async for message in cursor:
                message_data = {
                    "id": str(message["_id"]),
                    **{k: v for k, v in message.items() if k != "_id"}
                }
                messages.append(MessageResponse(**message_data))
                
            # Reverse to get chronological order
            return list(reversed(messages))
            
        except Exception as e:
            logger.error(f"Error getting recent messages for session {session_id}: {str(e)}")
            return []

    async def update_session_context(self, session_id: str, context: Dict[str, Any]) -> bool:
        """Update session context (useful for LangChain/LangGraph state)"""
        try:
            result = await self.sessions_collection.update_one(
                {"_id": ObjectId(session_id)},
                {
                    "$set": {
                        "context": context,
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Error updating session context {session_id}: {str(e)}")
            return False

    async def archive_old_sessions(self, days_old: int = 30) -> int:
        """Archive sessions older than specified days"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            
            result = await self.sessions_collection.update_many(
                {
                    "last_activity": {"$lt": cutoff_date},
                    "status": SessionStatus.ACTIVE.value
                },
                {
                    "$set": {
                        "status": SessionStatus.ARCHIVED.value,
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            
            return result.modified_count
            
        except Exception as e:
            logger.error(f"Error archiving old sessions: {str(e)}")
            return 0 