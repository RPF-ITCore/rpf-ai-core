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