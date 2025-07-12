from pydantic import BaseModel
from typing import List, Optional

class ChatRequestSchema(BaseModel):
    message: str
    collection_name: str = "syria_knowledge_base"
    max_results: int = 5

class ChatResponseSchema(BaseModel):
    response: str
    sources: Optional[List[str]] = None
    confidence_scores: Optional[List[float]] = None

class KnowledgeBaseIngestSchema(BaseModel):
    collection_name: str = "syria_knowledge_base"
    chunk_size: int = 1200
    overlap_size: int = 200
    reset_collection: bool = False
    
########################################################################################################

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from bson import ObjectId
from enum import Enum

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class MessageType(str, Enum):
    TEXT = "text"
    TOOL_CALL = "tool_call"
    TOOL_RESPONSE = "tool_response"

class ChatMessage(BaseModel):
    id: Optional[str] = Field(default=None, alias="_id")
    session_id: str
    role: MessageRole
    content: str
    message_type: MessageType = MessageType.TEXT
    metadata: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        populate_by_name = True
        json_encoders = {
            ObjectId: str,
            datetime: lambda dt: dt.isoformat()
        }

class SessionStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    ARCHIVED = "archived"

class ChatSession(BaseModel):
    id: Optional[str] = Field(default=None, alias="_id")
    user_id: Optional[str] = None  # For future user authentication
    title: Optional[str] = None
    status: SessionStatus = SessionStatus.ACTIVE
    context: Optional[Dict[str, Any]] = None  # For LangChain/LangGraph context
    metadata: Optional[Dict[str, Any]] = None
    message_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        populate_by_name = True
        json_encoders = {
            ObjectId: str,
            datetime: lambda dt: dt.isoformat()
        }

# For aggregation and complex queries
class ChatSessionWithMessages(ChatSession):
    messages: List[ChatMessage] = []
    
class SessionSummary(BaseModel):
    total_sessions: int
    active_sessions: int
    total_messages: int
    average_messages_per_session: float 