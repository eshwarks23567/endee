"""
Configuration settings for the Endee Research Assistant.
Uses pydantic-settings for environment variable management.
"""

import os
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    
    # Endee Configuration
    endee_host: str = Field(default="localhost", alias="ENDEE_HOST")
    endee_port: int = Field(default=8080, alias="ENDEE_PORT")
    
    # Google Gemini Configuration (for RAG features)
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    
    # Embedding Configuration
    embedding_model: str = Field(default="all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")
    embedding_dimension: int = Field(default=384, alias="EMBEDDING_DIMENSION")
    
    # Collection Configuration
    collection_name: str = Field(default="research_papers", alias="COLLECTION_NAME")
    
    # Application Settings
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    debug: bool = Field(default=False, alias="DEBUG")
    
    # Chunk Configuration
    chunk_size: int = Field(default=512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, alias="CHUNK_OVERLAP")
    
    @property
    def endee_url(self) -> str:
        """Get the full Endee server API URL."""
        return f"http://{self.endee_host}:{self.endee_port}/api/v1"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()
