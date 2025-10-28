from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import JSONResponse
from dependencies.database import get_db_client
from services.knowledge_base_service import KnowledgeBaseMongoService
from llm.llm_config import get_embedding_client
from vectordb import VectorDBProviderFactory
from dtos.knowledge_base import UploadDocumentResponse
from controllers.FileController import FileController
from core.config import get_settings
import os

router = APIRouter()

@router.post("/knowledge-base/upload", response_model=UploadDocumentResponse)
async def upload_document(
    knowledge_base_id: str = Form(...),
    name: str = Form(...),
    description: str = Form(None),
    file: UploadFile = File(...),
    db=Depends(get_db_client)
):
    # Validate file
    file_ctrl = FileController()
    valid, msg = file_ctrl.validate_uploaded_file(file)
    if not valid:
        raise HTTPException(status_code=400, detail=msg)
    # Save file temporarily
    settings = get_settings()
    upload_dir = os.path.join(settings.BASE_DIR, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    # Get type
    type_ = file.filename.split(".")[-1].lower()
    # Init services
    vectordb_client = db.app.vectordb_client if hasattr(db, 'app') and hasattr(db.app, 'vectordb_client') else None
    embedding_client = get_embedding_client()
    kb_service = KnowledgeBaseMongoService(db, vectordb_client, embedding_client)
    # Ingest
    try:
        resp = await kb_service.ingest_document(
            kb_id=knowledge_base_id,
            file_path=file_path,
            name=name,
            type_=type_,
            description=description
        )
        return resp
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
