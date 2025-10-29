from pydantic import BaseModel, Field, validator
from typing import Optional, Generic, TypeVar, List

class RetrievedDocumentSchema(BaseModel):
    text : str
    score : float


# ================= Standard API Response Shapes =================

T = TypeVar("T")

class ErrorItem(BaseModel):
    message: str
    code: Optional[str] = None
    field: Optional[str] = None


class ApiResponse(BaseModel, Generic[T]):
    message: str
    data: Optional[T] = None
    errors: List[ErrorItem] = []
