"""
Configuration management using Pydantic Settings.
Loads from environment variables with .env file support.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """Application settings."""
    
    # Application
    app_name: str = "VoiceFlow ML Service"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # API
    api_v1_prefix: str = "/api"
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Database
    postgres_user: str = "voiceflow"
    postgres_password: str = "voiceflow_password"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "voiceflow"
    
    @property
    def database_url(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
    
    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    @property
    def redis_url(self) -> str:
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
    # Rust Inference Service
    rust_service_url: str = "http://localhost:3000"
    rust_service_timeout: int = 30
    
    # JWT Authentication
    jwt_secret_key: str = "your-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 60
    
    # Rate Limiting
    rate_limit_requests_per_minute: int = 100
    
    # Model Storage
    models_dir: str = "../models"
    data_dir: str = "../data"
    
    # Training
    default_batch_size: int = 32
    default_learning_rate: float = 0.001
    default_epochs: int = 50
    
    # Audio Processing
    sample_rate: int = 16000
    audio_max_duration_seconds: int = 1800  # 30 minutes
    audio_max_size_mb: int = 100
    
    # CORS
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:8000"]
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )


# Global settings instance
settings = Settings()
