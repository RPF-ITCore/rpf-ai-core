from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.responses import Response
from starlette import status as http_status
from fastapi import HTTPException
from core.config import get_settings
import logging
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from vectordb import VectorDBProviderFactory
from llm import LLMProviderFactory
from llm.prompt_templates import TemplateParser
from routes import base_router, data_router, chat_router, chat_session_router, auth_router, stats_router
app = FastAPI()

# =================Logger Configurations=================
logging.basicConfig(
    level=logging.INFO,  
    format='%(name)s - %(levelname)s - %(message)s',  # Message format
    datefmt='%Y-%m-%d %H:%M:%S',  
    handlers=[
        logging.StreamHandler(),  # Logs to the console
    ]
)



logger = logging.getLogger(__name__)

# =================CORS Configurations=================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # This allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # This allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # This allows all headers
)

@app.on_event("startup")
async def startup():
    settings = get_settings()

    # ======================MongoDB Intialization ======================
    try:
        app.mongo_conn = AsyncIOMotorClient(settings.MONGODB_URL)
        app.db_client = app.mongo_conn[settings.MONGODB_DATABASE]
        logger.info(f"Connected to MongoDB Atlas")
    except Exception as e:
        logger.error(f"Error connecting to MongoDB: {str(e)}")
    
    
        # =================VectorDB Initialization=================
    try:
        vector_db_provider_factory = VectorDBProviderFactory(settings)

        app.vectordb_client = vector_db_provider_factory.create(provider = settings.VECTORDB_BACKEND)
        app.vectordb_client.connect(url=settings.VECTORDB_URL, api_key=settings.QDRANT_API_KEY)

        logger.info("VectorDB provider has been initialized successfully")
    
    except Exception as e:
        logger.error(f"Error initializing VectorDB: {str(e)}")
        
        
        # =================LLM Initialization=================
    try:
        llm_provider_factory = LLMProviderFactory(settings)

        # Generation Client
        app.generation_client = llm_provider_factory.create(provider = settings.GENERATION_BACKEND)
        app.generation_client.set_generation_model(model_id = settings.GENERATION_MODEL_ID)
        
        # Embedding Client
        app.embedding_client = llm_provider_factory.create(provider=settings.EMBEDDING_BACKEND)
        # Get embedding size from settings - this must match the actual model output dimensions
        # Parse as int, with fallback to 1536 if not provided
        try:
            embedding_size = int(settings.EMBEDDING_MODEL_SIZE) if settings.EMBEDDING_MODEL_SIZE else 1536
        except (ValueError, TypeError):
            embedding_size = 1536
            logger.warning(f"Could not parse EMBEDDING_MODEL_SIZE from settings, using default: {embedding_size}")
        
        app.embedding_client.set_embedding_model(
            model_id=settings.EMBEDDING_MODEL_ID,
            embedding_size=embedding_size
        )
        logger.info(f"Embedding model '{settings.EMBEDDING_MODEL_ID}' configured with size: {embedding_size} dimensions")

        logger.info(f"LLM Generation Model has beed initialized : {settings.GENERATION_MODEL_ID}")
        logger.info(f"LLM Embedding Model has beed initialized : {settings.EMBEDDING_MODEL_ID}")

    except Exception as e:
        logger.error(f"Error initializing LLM: {str(e)}")
    
    
    # =================Template Parser Initialization=================
    try:
        app.template_parser = TemplateParser(
        language = settings.PRIMARY_LANGUAGE,
        default_language = settings.DEFAULT_LANGUAGE
        )
        logger.info(f"Template Parser has been initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing Template Parser: {str(e)}")


@app.on_event("shutdown")
async def shutdown():
    if hasattr(app, 'mongo_conn'):
        app.mongo_conn.close()
    if hasattr(app, 'vectordb_client'):
        app.vectordb_client.disconnect()     
    
    
    
# =================Routers Configurations=================  
app.include_router(base_router)
app.include_router(auth_router)
app.include_router(data_router)
app.include_router(chat_router)
app.include_router(chat_session_router)
app.include_router(stats_router)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "message": exc.detail if isinstance(exc.detail, str) else "HTTP error",
            "data": None,
            "errors": [exc.detail] if isinstance(exc.detail, str) else [str(exc.detail)],
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for err in exc.errors():
        errors.append({
            "message": err.get("msg", "Validation error"),
            "code": err.get("type"),
            "field": ".".join([str(x) for x in err.get("loc", [])])
        })
    return JSONResponse(
        status_code=http_status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "message": "Validation error",
            "data": None,
            "errors": errors,
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "message": "Internal server error",
            "data": None,
            "errors": [str(exc)],
        },
    )
