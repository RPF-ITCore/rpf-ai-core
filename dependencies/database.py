from motor.motor_asyncio import AsyncIOMotorClient
from core.config import  get_settings
import logging

logger = logging.getLogger(__name__)
settings = get_settings()

def get_db_client():
    try:
        mongo_conn = AsyncIOMotorClient(settings.MONGODB_URL)
        db_client = mongo_conn[settings.MONGODB_DATABASE]
        logger.info(f"Connected to MongoDB Atlas")
        return db_client
    except Exception as e:
        logger.error(f"Error connecting to MongoDB: {str(e)}")