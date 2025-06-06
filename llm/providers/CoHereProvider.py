from ..LLMInterface import LLMInterface
from ..LLMEnums import CoHereEnums, DocumentTypeEnum
import cohere
import logging
from typing import List, Dict

class CoHereProvider(LLMInterface):

    def __init__(self, api_key: str,
                 default_input_max_characters: int = 1000,
                 default_generation_max_output_tokens: int = 1000,
                 default_generation_temperature: float = 0.3):
        
        self.api_key = api_key
        
        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = None
        self.embedding_model_id = None
        self.embedding_size = None

        self.client = cohere.Client(api_key=self.api_key)

        self.enums = CoHereEnums

        self.logger = logging.getLogger(__name__)
    

    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size
    
    def generate_text(self, prompt: str, chat_history: list = [], max_output_tokens: int = None,
                      temperature: float = None) -> str:

        if not self.client:
            self.logger.error("CoHere client not initialized.")
            return None

        if not self.generation_model_id:
            self.logger.error("CoHere generation model not set")
            return None
        

        max_output_tokens = max_output_tokens if max_output_tokens else self.default_generation_max_output_tokens

        temperature = temperature if temperature else self.default_generation_temperature
        
        try:
            response = self.client.chat(
                model=self.generation_model_id,
                chat_history=chat_history,
                message=prompt,  # CoHere expects just the prompt string
                temperature=temperature,
                max_tokens=max_output_tokens
            )

            if not response or not response.text:
                self.logger.error("CoHere Client: Failed to generate text.")
                return None

            return response.text
            
        except Exception as e:
            self.logger.error(f"Error generating text with CoHere: {str(e)}")
            raise e

    async def chat(self, messages: List[Dict[str, str]], max_tokens: int = None, temperature: float = None) -> str:
        """
        Chat method expected by flows. Takes list of message dicts with 'role' and 'content' keys.
        This converts the messages format to what CoHere expects.
        """
        if not self.client:
            self.logger.error("CoHere client not initialized.")
            raise Exception("CoHere client not initialized.")
        
        if not self.generation_model_id:
            self.logger.error("CoHere generation model not set")
            raise Exception("CoHere generation model not set")
        
        max_tokens = max_tokens if max_tokens else self.default_generation_max_output_tokens
        temperature = temperature if temperature else self.default_generation_temperature
        
        try:
            # Convert messages format to CoHere's expected format
            chat_history = []
            current_message = None
            
            for msg in messages:
                processed_content = self.process_text(msg["content"])
                
                if msg["role"] == "system":
                    # CoHere doesn't have a direct system role, so we treat it as the current message
                    current_message = processed_content
                elif msg["role"] == "user":
                    current_message = processed_content
                elif msg["role"] == "assistant":
                    # Add previous exchanges to chat history
                    if current_message:
                        chat_history.append({
                            "role": "USER",
                            "message": current_message
                        })
                        chat_history.append({
                            "role": "CHATBOT", 
                            "message": processed_content
                        })
                        current_message = None
            
            # If we only have a system/user message without assistant response, use it as current message
            if current_message is None and messages:
                current_message = self.process_text(messages[-1]["content"])
            
            response = self.client.chat(
                model=self.generation_model_id,
                chat_history=chat_history,
                message=current_message or "Hello",
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            self.logger.info(f"Chat response generated successfully")
            
            if not response or not response.text:
                self.logger.error("CoHere Client: Failed to generate chat response.")
                raise Exception("Empty response from CoHere")
            
            return response.text
            
        except Exception as e:
            self.logger.error(f"Error in chat with CoHere: {str(e)}")
            raise e

    def embed_text(self, text: str, document_type: str = None):

        if not self.client:
            self.logger.error("CoHere client not initialized.")
            return None
        
        if not self.embedding_model_id:
            self.logger.error("CoHere Embedding model not set.")
            return None
        
        input_type = CoHereEnums.DOCUMENT.value 
        if document_type == DocumentTypeEnum.QUERY:
            input_type = CoHereEnums.QUERY.value

        try:
            response = self.client.embed(
                model=self.embedding_model_id,
                texts=[self.process_text(text)],
                input_type=input_type,
                embedding_types=['float'],
            )

            if not response or not response.embeddings or not response.embeddings.float:
                self.logger.error("CoHere Client: Failed to embed text.")
                return None

            return response.embeddings.float[0]
            
        except Exception as e:
            self.logger.error(f"Error embedding text with CoHere: {str(e)}")
            return None

    def construct_prompt(self, prompt: str, role: str):
        return {
            "role": role,
            "content": self.process_text(text=prompt),
        }

    def process_text(self, text: str):
        return text[:self.default_input_max_characters].strip()


