from core.config import get_settings
import os
import json
from bson.objectid import ObjectId
from datetime import datetime
from fastapi.responses import JSONResponse
from typing import Any, List
from schemas.utils import ErrorItem


class BaseController:

    def __init__(self):
        self.app_settings = get_settings()

        self.base_dir = os.path.dirname( os.path.dirname(__file__) )
        self.files_dir = os.path.join(
            self.base_dir,
            "assets/files"
        ) 
        self.database_dir = os.path.join(
            self.base_dir,
            "assets/database"
        )


    def get_vector_database_path(self, db_name):
        database_path = os.path.join(
            self.database_dir,
            db_name
        )
        if not os.path.exists(database_path):
            os.makedirs(database_path)
        
        return database_path
    
    
    
    def get_json_serializable_object(self, info):
        def custom_serializer(obj):
            if isinstance(obj, ObjectId):
                return str(obj)  # Convert ObjectId to string
            if isinstance(obj, datetime):
                return obj.isoformat()  # Convert datetime to ISO format string
            try:
                return obj.__dict__
            except AttributeError:
                return str(obj)
        
        return json.loads(json.dumps(info, default=custom_serializer))

    # ---------- Standardized envelope helpers ----------
    def ok(self, data: Any = None, message: str = "OK", status_code: int = 200):
        payload = {
            "message": message,
            "data": data,
            "errors": []
        }
        return JSONResponse(
            status_code=status_code,
            content=self.get_json_serializable_object(payload)
        )

    def fail(self, message: str, errors: List[ErrorItem | str] = [], status_code: int = 400):
        normalized_errors = []
        for e in errors:
            if isinstance(e, ErrorItem):
                normalized_errors.append(e.model_dump())
            elif isinstance(e, dict):
                normalized_errors.append(e)
            else:
                normalized_errors.append({"message": str(e)})
        payload = {
            "message": message,
            "data": None,
            "errors": normalized_errors
        }
        return JSONResponse(
            status_code=status_code,
            content=self.get_json_serializable_object(payload)
        )

