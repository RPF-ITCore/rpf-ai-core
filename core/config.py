from pydantic_settings import BaseSettings , SettingsConfigDict


class Settings(BaseSettings):

    APP_NAME : str
    APP_VERSION : str

    MONGODB_URL : str
    MONGODB_DATABASE : str

    
    # FILE_ALLOWED_TYPES : list[str]
    # FILE_MAX_SIZE : int
    # FILE_DEFAULT_CHUNK_SIZE : int
    
    GENERATION_BACKEND : str
    EMBEDDING_BACKEND : str

    OPENAI_API_KEY : str 
    OPENAI_API_URL : str

    COHERE_API_KEY : str

    GENERATION_MODEL_ID : str
    EMBEDDING_MODEL_ID : str
    EMBEDDING_MODEL_SIZE : str

    DEFAULT_INPUT_MAX_CHARACTERS : int = None
    DEFAULT_GENERATION_MAX_OUTPUT_TOKENS : int = None
    DEFAULT_GENERATION_TEMPREATUER : float = None

    VECTORDB_BACKEND : str
    VECTORDB_PATH : str
    VECTORDB_DISTANCE_METHOD : str
    VECTORDB_URL : str
    QDRANT_API_KEY : str

    DEFAULT_LANGUAGE : str = 'en'
    PRIMARY_LANGUAGE : str

    class Config(SettingsConfigDict):
        env_file = '.env'


def get_settings():
    return Settings()
