from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class UploadDocumentRequest(BaseModel):
    knowledge_base_id: str
    name: str
    type: str  # pdf, txt, etc.
    description: Optional[str] = None

class UploadDocumentResponse(BaseModel):
    document_id: str
    knowledge_base_id: str
    chunk_count: int
    vector_db_collection: str
    message: str

class DataChunkDTO(BaseModel):
    document_id: str
    text_chunk: str
    order: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
