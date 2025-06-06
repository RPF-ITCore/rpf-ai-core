from abc import ABC, abstractmethod
from typing import List, Dict, Any

class LLMInterface(ABC):

    @abstractmethod
    def set_generation_model(self, model_id : str):
        pass

    @abstractmethod
    def set_embedding_model(self, model_id : str, embedding_size : int):
        pass

    @abstractmethod
    def generate_text(self, prompt : str, chat_history : list = [], max_output_tokens : int = None,
                      temperature : float = None) -> str:  #tempatraure should be low if we want the model to be precise and not creative
        pass
    
    @abstractmethod
    async def chat(self, messages: List[Dict[str, str]], max_tokens: int = None, temperature: float = None) -> str:
        """
        Chat method expected by flows. Takes list of message dicts with 'role' and 'content' keys.
        This should be the primary method used by flows.
        """
        pass
    
    @abstractmethod
    def embed_text(self, text : str, document_type : str = None):
        pass

    @abstractmethod
    def construct_prompt(self, prompt : str, role : str):
        pass

    


