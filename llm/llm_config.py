from core.config import get_settings
from llm import LLMProviderFactory
import logging

logger = logging.getLogger(__name__)

class LLMConfig:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMConfig, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self.settings = get_settings()
        self.generation_client = None
        self.embedding_client = None
        self._initialize_clients()
        self._initialized = True

    def _initialize_clients(self):
        try:
            llm_provider_factory = LLMProviderFactory(self.settings)

            # Generation Client
            self.generation_client = llm_provider_factory.create(provider=self.settings.GENERATION_BACKEND)
            self.generation_client.set_generation_model(model_id=self.settings.GENERATION_MODEL_ID)
            
            # Embedding Client
            self.embedding_client = llm_provider_factory.create(provider=self.settings.EMBEDDING_BACKEND)
            self.embedding_client.set_embedding_model(
                model_id=self.settings.EMBEDDING_MODEL_ID,
                embedding_size=self.settings.EMBEDDING_MODEL_SIZE  # Using EMBEDDING_DIM from settings
            )

            logger.info(f"LLMConfig: Generation Model initialized: {self.settings.GENERATION_MODEL_ID}")
            logger.info(f"LLMConfig: Embedding Model initialized: {self.settings.EMBEDDING_MODEL_ID}")

        except Exception as e:
            logger.error(f"LLMConfig: Error initializing LLM clients: {str(e)}")
            # Optionally re-raise or handle as appropriate for your application
            raise

# Singleton instance
llm_config = LLMConfig()

def get_llm_config():
    return llm_config

def get_generation_client():
    return llm_config.generation_client

def get_embedding_client():
    return llm_config.embedding_client 