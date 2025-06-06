from pydantic import BaseModel, Field, validator
from typing import Optional

class RetrievedDocumentSchema(BaseModel):
    text : str
    score : float