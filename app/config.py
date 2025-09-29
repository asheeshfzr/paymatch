# app/config.py
"""
Centralized configuration using pydantic-settings with validation and feature flags.
"""

import os
from pathlib import Path
from typing import Optional, List
from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Database and data file configuration."""
    model_config = SettingsConfigDict(env_prefix="DB_")
    
    users_csv_path: str = Field(default="data/users.csv", description="Path to users CSV file")
    transactions_csv_path: str = Field(default="data/transactions.csv", description="Path to transactions CSV file")
    
    @validator('users_csv_path', 'transactions_csv_path')
    def validate_csv_paths(cls, v):
        """Validate that CSV files exist or provide helpful error message."""
        path = Path(v)
        if not path.exists():
            # Try alternative paths
            alt_paths = [
                Path("/mnt/data") / path.name,
                Path("data") / path.name,
                path
            ]
            for alt_path in alt_paths:
                if alt_path.exists():
                    return str(alt_path)
            raise ValueError(f"CSV file not found: {v}. Tried: {[str(p) for p in alt_paths]}")
        return str(path)


class LLMSettings(BaseSettings):
    """LLM configuration with validation."""
    model_config = SettingsConfigDict(env_prefix="LLM_")
    
    api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    model: str = Field(default="gpt-4o-mini", description="OpenAI model to use")
    timeout: float = Field(default=30.0, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_delay: float = Field(default=1.0, description="Initial retry delay in seconds")
    max_tokens_extract: int = Field(default=32, description="Max tokens for name extraction")
    max_tokens_expand: int = Field(default=120, description="Max tokens for query expansion")
    max_tokens_rerank: int = Field(default=256, description="Max tokens for reranking")
    temperature_extract: float = Field(default=0.0, description="Temperature for name extraction")
    temperature_expand: float = Field(default=0.2, description="Temperature for query expansion")
    temperature_rerank: float = Field(default=0.0, description="Temperature for reranking")
    cost_budget_per_request: float = Field(default=0.01, description="Max cost per request in USD")
    
    @validator('api_key')
    def validate_api_key(cls, v):
        """Validate API key format."""
        if v and not v.startswith('sk-'):
            raise ValueError("OpenAI API key must start with 'sk-'")
        return v
    
    @validator('timeout', 'retry_delay')
    def validate_positive_floats(cls, v):
        """Validate positive float values."""
        if v <= 0:
            raise ValueError("Must be positive")
        return v
    
    @validator('max_retries')
    def validate_retries(cls, v):
        """Validate retry count."""
        if v < 0 or v > 10:
            raise ValueError("Must be between 0 and 10")
        return v


class QdrantSettings(BaseSettings):
    """Qdrant vector database configuration."""
    model_config = SettingsConfigDict(env_prefix="QDRANT_")
    
    host: str = Field(default="localhost", description="Qdrant host")
    port: int = Field(default=6333, description="Qdrant port")
    collection_name: str = Field(default="transactions", description="Collection name")
    timeout: float = Field(default=10.0, description="Connection timeout")
    
    @validator('port')
    def validate_port(cls, v):
        """Validate port number."""
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v


class EmbeddingSettings(BaseSettings):
    """Embedding model configuration."""
    model_config = SettingsConfigDict(env_prefix="EMBEDDING_")
    
    model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="Embedding model")
    batch_size: int = Field(default=32, description="Batch size for embedding computation")
    max_sequence_length: int = Field(default=512, description="Max sequence length")
    cache_size: int = Field(default=1000, description="Cache size for embeddings")
    
    @validator('batch_size', 'cache_size')
    def validate_positive_ints(cls, v):
        """Validate positive integer values."""
        if v <= 0:
            raise ValueError("Must be positive")
        return v


class APISettings(BaseSettings):
    """API configuration."""
    model_config = SettingsConfigDict(env_prefix="API_")
    
    host: str = Field(default="127.0.0.1", description="API host")
    port: int = Field(default=8000, description="API port")
    max_top_k: int = Field(default=100, description="Maximum top_k parameter")
    default_top_k: int = Field(default=10, description="Default top_k parameter")
    max_query_length: int = Field(default=1000, description="Maximum query length")
    request_timeout: float = Field(default=60.0, description="Request timeout")
    
    @validator('port')
    def validate_port(cls, v):
        """Validate port number."""
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v


class FeatureFlags(BaseSettings):
    """Feature flags for optional functionality."""
    model_config = SettingsConfigDict(env_prefix="FEATURE_")
    
    enable_llm: bool = Field(default=True, description="Enable LLM features")
    enable_qdrant: bool = Field(default=True, description="Enable Qdrant vector search")
    enable_caching: bool = Field(default=True, description="Enable result caching")
    enable_metrics: bool = Field(default=True, description="Enable Prometheus metrics")
    enable_tracing: bool = Field(default=True, description="Enable OpenTelemetry tracing")
    enable_async: bool = Field(default=True, description="Enable async endpoints")


class LoggingSettings(BaseSettings):
    """Logging configuration."""
    model_config = SettingsConfigDict(env_prefix="LOG_")
    
    level: str = Field(default="INFO", description="Log level")
    format: str = Field(default="json", description="Log format (json or text)")
    include_request_id: bool = Field(default=True, description="Include request IDs in logs")
    include_correlation_id: bool = Field(default=True, description="Include correlation IDs in logs")
    
    @validator('level')
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Must be one of: {valid_levels}")
        return v.upper()


class Settings(BaseSettings):
    """Main application settings combining all configuration sections."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Sub-configurations
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    api: APISettings = Field(default_factory=APISettings)
    features: FeatureFlags = Field(default_factory=FeatureFlags)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    
    # Application metadata
    app_name: str = Field(default="Deel Staff AI", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    environment: str = Field(default="development", description="Environment")
    
    @validator('environment')
    def validate_environment(cls, v):
        """Validate environment."""
        valid_envs = ['development', 'staging', 'production']
        if v not in valid_envs:
            raise ValueError(f"Must be one of: {valid_envs}")
        return v
    
    def validate_critical_config(self) -> None:
        """Validate critical configuration and raise helpful errors."""
        errors = []
        
        # Check if LLM is enabled but no API key provided
        if self.features.enable_llm and not self.llm.api_key:
            errors.append("LLM features enabled but OPENAI_API_KEY not provided")
        
        # Check if Qdrant is enabled but might not be reachable
        if self.features.enable_qdrant:
            try:
                # Check if qdrant-client is available
                import qdrant_client
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex((self.qdrant.host, self.qdrant.port))
                sock.close()
                if result != 0:
                    errors.append(f"Qdrant not reachable at {self.qdrant.host}:{self.qdrant.port}")
            except ImportError:
                errors.append("Qdrant enabled but qdrant-client not installed. Install with: pip install qdrant-client")
            except Exception as e:
                errors.append(f"Qdrant connection check failed: {e}")
        
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
            raise ValueError(error_msg)


# Global settings instance
settings = Settings()

# Validate critical configuration on import
try:
    settings.validate_critical_config()
except ValueError as e:
    print(f"Configuration Error: {e}")
    print("Please check your environment variables and configuration.")
    raise
