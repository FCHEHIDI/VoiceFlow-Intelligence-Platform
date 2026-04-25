"""
Configuration management using Pydantic Settings.
Secrets must come from environment (or from AWS Secrets Manager in production when ARNs are set).
"""

from typing import List, Optional
from urllib.parse import quote_plus

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from core.secrets_manager import get_secrets_loader


class Settings(BaseSettings):
    """Application settings. No default secrets: set JWT_SECRET_KEY and POSTGRES_PASSWORD in .env."""

    # Application
    app_name: str = "VoiceFlow ML Service"
    app_version: str = "1.0.0"
    debug: bool = False
    # Deployment environment (read from `ENV` in the process environment)
    app_env: str = Field(
        default="development",
        validation_alias="ENV",
        description="One of: development, staging, production",
    )

    # API
    api_v1_prefix: str = "/api"
    host: str = "0.0.0.0"
    port: int = 8000

    # Database — password must be supplied; never use a default
    postgres_user: str = "voiceflow"
    postgres_password: str
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "voiceflow"

    @property
    def database_url(self) -> str:
        user = quote_plus(self.postgres_user)
        password = quote_plus(self.postgres_password)
        return (
            f"postgresql://{user}:{password}"
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
            p = quote_plus(self.redis_password)
            return f"redis://:{p}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    # Rust Inference Service
    rust_service_url: str = "http://localhost:3000"
    rust_service_timeout: int = 30

    # JWT — no placeholder default; optional ARN for production
    jwt_secret_key: str = Field(default="", validation_alias="JWT_SECRET_KEY")
    jwt_secret_arn: Optional[str] = Field(default=None, validation_alias="JWT_SECRET_ARN")
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 60

    @field_validator("jwt_secret_key", mode="before")
    @classmethod
    def _jwt_strip(cls, v: object) -> object:
        if v is None:
            return ""
        if isinstance(v, str):
            return v.strip()
        return v

    @model_validator(mode="after")
    def _resolve_jwt_from_aws_in_production(self) -> "Settings":
        if self.jwt_secret_arn and (self.app_env or "").lower() == "production":
            key = get_secrets_loader().get_aws_secret_string(self.jwt_secret_arn)
            object.__setattr__(self, "jwt_secret_key", key.strip())
        if not self.jwt_secret_key or len(self.jwt_secret_key) < 32:
            raise ValueError(
                "JWT secret must be at least 32 characters. "
                "Set JWT_SECRET_KEY in the environment, or in production set JWT_SECRET_ARN. "
                "Generate a local key: openssl rand -hex 32"
            )
        return self

    @field_validator("app_env")
    @classmethod
    def _normalize_app_env(cls, v: str) -> str:
        return (v or "development").strip().lower()

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
    audio_max_duration_seconds: int = 1800
    audio_max_size_mb: int = 100

    # CORS: comma-separated origins (strict; no wildcards in production)
    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:8000",
        validation_alias="CORS_ORIGINS",
    )

    @property
    def cors_allowed_origins(self) -> List[str]:
        return [o.strip() for o in (self.cors_origins or "").split(",") if o.strip()]

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()
