from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from schemas.chat import MessageRole, MessageType, SessionStatus

# Request DTOs
class CreateSessionRequest(BaseModel):
    user_id: Optional[str] = None
    title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class UpdateSessionRequest(BaseModel):
    title: Optional[str] = None
    status: Optional[SessionStatus] = None
    metadata: Optional[Dict[str, Any]] = None

class CreateMessageRequest(BaseModel):
    session_id: str
    role: MessageRole
    content: str
    message_type: MessageType = MessageType.TEXT
    metadata: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

class ChatRequest(BaseModel):
    session_id: Optional[str] = None  # If None, creates new session
    message: str
    user_id: Optional[str] = None
    create_new_session: bool = False

# Response DTOs
class MessageResponse(BaseModel):
    id: str
    session_id: str
    role: MessageRole
    content: str
    message_type: MessageType
    metadata: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    created_at: datetime

class SessionResponse(BaseModel):
    id: str
    user_id: Optional[str] = None
    title: Optional[str] = None
    status: SessionStatus
    context: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    message_count: int
    created_at: datetime
    updated_at: datetime
    last_activity: datetime

class SessionWithMessagesResponse(SessionResponse):
    messages: List[MessageResponse]

class ChatResponse(BaseModel):
    session_id: str
    message: Optional[MessageResponse] = None
    assistant_response: Optional[MessageResponse] = None
    success: bool = True
    error: Optional[str] = None

class SessionListResponse(BaseModel):
    sessions: List[SessionResponse]
    total: int
    page: int
    page_size: int
    has_next: bool

class MessageListResponse(BaseModel):
    messages: List[MessageResponse]
    total: int
    page: int
    page_size: int
    has_next: bool 