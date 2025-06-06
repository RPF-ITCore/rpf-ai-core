from fastapi import FastAPI
from core.config import get_settings
import logging
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from vectordb import VectorDBProviderFactory
from llm import LLMProviderFactory
from llm.prompt_templates import TemplateParser
from routes import base_router, data_router, chat_router
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

    # # ======================MongoDB Intialization ======================
    # try:
    #     app.mongo_conn = AsyncIOMotorClient(settings.MONGODB_URL)
    #     app.db_client = app.mongo_conn[settings.MONGODB_DATABASE]
    #     logger.info(f"Connected to MongoDB Atlas")
    # except Exception as e:
    #     logger.error(f"Error connecting to MongoDB: {str(e)}")
    
    
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
        app.embedding_client.set_embedding_model(
            model_id=settings.EMBEDDING_MODEL_ID,
            embedding_size=1536  # For OpenAI's text-embedding-ada-002 model
        )

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
    # app.mongo_conn.close()  # Commented out since MongoDB is not initialized
    if hasattr(app, 'vectordb_client'):
        app.vectordb_client.disconnect()     
    
    
    
# =================Routers Configurations=================  
app.include_router(base_router)
app.include_router(data_router)
app.include_router(chat_router)


