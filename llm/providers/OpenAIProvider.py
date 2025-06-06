from ..LLMInterface import LLMInterface
from ..LLMEnums import OpenAIEnums
from openai import OpenAI
import logging
from typing import List, Dict

class OpenAIProvider(LLMInterface):

    def __init__(self, api_key: str, api_url: str=None,
                       default_input_max_characters: int=1000,
                       default_generation_max_output_tokens: int=1000,
                       default_generation_temperature: float=0.1):
        
        self.api_key = api_key
        self.api_url = api_url

        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = None

        self.embedding_model_id = None
        self.embedding_size = None
        
        client_kwargs = {
            "api_key": self.api_key,
        }
        
        if self.api_url and len(self.api_url.strip()) > 0:
            client_kwargs["base_url"] = self.api_url

        self.enums = OpenAIEnums 

        self.client = OpenAI(
            **client_kwargs
        )

        self.logger = logging.getLogger(__name__)

    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int):
        """
        Set the embedding model and its dimension size
        
        Args:
            model_id: The model ID (e.g., 'text-embedding-ada-002')
            embedding_size: The expected embedding dimension (e.g., 1536 for ada-002)
        """
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size
        self.logger.info(f"Set embedding model {model_id} with dimension {embedding_size}")

    def process_text(self, text: str):
        return text[:self.default_input_max_characters].strip()

    def generate_text(self, prompt: str, chat_history: list=[], max_output_tokens: int=None,
                            temperature: float = None):
        
        if not self.client:
            self.logger.error("OpenAI client was not set")
            return None

        if not self.generation_model_id:
            self.logger.error("Generation model for OpenAI was not set")
            return None
        
        max_output_tokens = max_output_tokens if max_output_tokens else self.default_generation_max_output_tokens
        temperature = temperature if temperature else self.default_generation_temperature

        chat_history.append(
            self.construct_prompt(prompt=prompt, role=OpenAIEnums.USER.value)
        )

        try:
            response = self.client.chat.completions.create(
                model=self.generation_model_id,
                messages=chat_history,
                max_tokens=max_output_tokens,
                temperature=temperature
            )
            
            self.logger.info(f"Response generated successfully")
            
            if not response or not response.choices or len(response.choices) == 0:
                self.logger.error("Empty response from OpenAI")
                raise Exception("Empty response from OpenAI")
            
            # Access the content attribute directly
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"Error generating text with OpenAI: {str(e)}")
            raise e

    async def chat(self, messages: List[Dict[str, str]], max_tokens: int = None, temperature: float = None) -> str:
        """
        Chat method expected by flows. Takes list of message dicts with 'role' and 'content' keys.
        This is the primary method used by flows.
        """
        if not self.client:
            self.logger.error("OpenAI client was not set")
            raise Exception("OpenAI client was not set")

        if not self.generation_model_id:
            self.logger.error("Generation model for OpenAI was not set")
            raise Exception("Generation model for OpenAI was not set")
        
        max_tokens = max_tokens if max_tokens else self.default_generation_max_output_tokens
        temperature = temperature if temperature else self.default_generation_temperature

        try:
            # Process the messages to ensure content is within character limits
            processed_messages = []
            for msg in messages:
                processed_messages.append({
                    "role": msg["role"],
                    "content": self.process_text(msg["content"])
                })

            response = self.client.chat.completions.create(
                model=self.generation_model_id,
                messages=processed_messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            self.logger.info(f"Chat response generated successfully")
            
            if not response or not response.choices or len(response.choices) == 0:
                self.logger.error("Empty response from OpenAI")
                raise Exception("Empty response from OpenAI")
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"Error in chat with OpenAI: {str(e)}")
            raise e

    def embed_text(self, text: str, document_type: str = None):
        
        if not self.client:
            self.logger.error("OpenAI client was not set")
            return None

        if not self.embedding_model_id:
            self.logger.error("Embedding model for OpenAI was not set")
            return None
        
        response = self.client.embeddings.create(
            model = self.embedding_model_id,
            input = text,
        )

        if not response or not response.data or len(response.data) == 0 or not response.data[0].embedding:
            self.logger.error("Error while embedding text with OpenAI")
            return None

        return response.data[0].embedding
    
    def batch_embed(self, texts: List[str], document_type: str | None = None) -> List[List[float]]:
        """
        Embed a batch of texts and return a list of embedding vectors.

        Args:
            texts: List of input strings to embed.
            document_type: Optional tag for downstream bookkeeping.

        Returns:
            List[List[float]] – one embedding vector per input text.
        """
        if not self.client:
            self.logger.error("OpenAI client was not set")
            return None

        if not self.embedding_model_id:
            self.logger.error("Embedding model for OpenAI was not set")
            return None

        response = self.client.embeddings.create(
            model=self.embedding_model_id,
            input=texts,
        )

        if (not response) or (not response.data):
            self.logger.error("Error while embedding texts with OpenAI")
            return None

        # response.data is an array of objects with an `.embedding` attribute
        return [item.embedding for item in response.data]


    #TODO: call the process_text
    def construct_prompt(self, prompt: str, role: str):
        return {
            "role": role,
            "content": prompt
        }
    


    