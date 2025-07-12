from typing import List


class FileConstants:
    """Constants related to file processing and uploads."""
    
    # Allowed file types for upload
    ALLOWED_TYPES: List[str] = ["text/plain", "application/pdf"]
    
    # Maximum file size in MB
    MAX_SIZE: int = 10
    
    # Default chunk size for text processing (in characters)
    DEFAULT_CHUNK_SIZE: int = 512000


class AppConstants:
    """General application constants."""
    
    # Application name
    APP_NAME: str = "RPF AI Core"
    
    # Version
    VERSION: str = "1.0.0"


# For backward compatibility, you can keep the old constants as aliases
FILE_ALLOWED_TYPES = FileConstants.ALLOWED_TYPES
FILE_MAX_SIZE = FileConstants.MAX_SIZE
FILE_DEFAULT_CHUNK_SIZE = FileConstants.DEFAULT_CHUNK_SIZE