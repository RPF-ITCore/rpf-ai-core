from .LLMEnums import LLMEnums
from .providers import OpenAIProvider , CoHereProvider
class LLMProviderFactory:

    def __init__(self, config : dict):
        self.config = config

    
    def create(self, provider : str):
        normalized = (provider or "").strip().upper()
        if normalized == LLMEnums.OPENAI.value:
            return OpenAIProvider(
                api_key =  self.config.OPENAI_API_KEY,
                api_url =  self.config.OPENAI_API_URL,
                default_generation_max_output_tokens = self.config.DEFAULT_GENERATION_MAX_OUTPUT_TOKENS,
                default_input_max_characters = self.config.DEFAULT_INPUT_MAX_CHARACTERS,
                default_generation_temperature = self.config.DEFAULT_GENERATION_TEMPREATUER 
            )
        
        if normalized == LLMEnums.COHERE.value:
            return CoHereProvider(
                api_key =  self.config.COHERE_API_KEY,
                default_generation_max_output_tokens = self.config.DEFAULT_GENERATION_MAX_OUTPUT_TOKENS,
                default_input_max_characters = self.config.DEFAULT_INPUT_MAX_CHARACTERS,
                default_generation_tempreature = self.config.DEFAULT_GENERATION_TEMPREATUER 
            )

        valid = f"{LLMEnums.OPENAI.value}, {LLMEnums.COHERE.value}"
        raise ValueError(
            f"Unknown LLM provider {provider!r} (normalized: {normalized!r}). "
            f"Set GENERATION_BACKEND or EMBEDDING_BACKEND to one of: {valid}."
        )


