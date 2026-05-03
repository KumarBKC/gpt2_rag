import os
from dataclasses import dataclass

@dataclass
class AppConfig:
    # Model settings
    MODEL_PATH: str = "models/gpt2"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Path settings
    RAW_DATA_PATH: str = "data/raw"
    INDEX_STORAGE_PATH: str = "data/index"
    
    # Text splitter settings
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    
    # Generation settings
    MAX_NEW_TOKENS: int = 50
    TEMPERATURE: float = 0.1
    TOP_K: int = 40
    REPETITION_PENALTY: float = 1.2

    @classmethod
    def get_default(cls):
        return cls()
