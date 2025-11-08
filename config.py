"""
Configuration Management for Moshi TTS API
==========================================
Centralized configuration using pydantic-settings for type-safe,
validated environment variable management.
"""

from functools import lru_cache
from typing import Optional, List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or .env file.

    All settings have sensible defaults and are validated on startup.
    Override any setting by setting the corresponding environment variable.
    """

    # ==========================================
    # Server Configuration
    # ==========================================
    host: str = Field(
        default="0.0.0.0",
        description="Server host address"
    )
    port: int = Field(
        default=8000,
        description="Server port number"
    )
    log_level: str = Field(
        default="info",
        description="Logging level (debug, info, warning, error)"
    )
    workers: int = Field(
        default=1,
        description="Number of uvicorn worker processes"
    )

    # ==========================================
    # API Configuration
    # ==========================================
    api_version: str = Field(
        default="1.1.0",
        description="API version number"
    )
    api_title: str = Field(
        default="Moshi TTS API",
        description="API title shown in documentation"
    )
    max_text_length: int = Field(
        default=5000,
        description="Maximum text length for synthesis (characters)"
    )

    # ==========================================
    # Model Configuration
    # ==========================================
    default_tts_repo: str = Field(
        default="kyutai/tts-1.6b-en_fr",
        description="HuggingFace repository for TTS model"
    )
    default_voice_repo: str = Field(
        default="kyutai/tts-voices",
        description="HuggingFace repository for voice presets"
    )
    sample_rate: int = Field(
        default=24000,
        description="Audio sample rate in Hz"
    )
    model_device: Optional[str] = Field(
        default=None,
        description="Force device (cuda/cpu). Auto-detect if None"
    )
    model_dtype: str = Field(
        default="auto",
        description="Model dtype (auto/bfloat16/float32)"
    )
    model_n_q: int = Field(
        default=32,
        description="Number of codebooks for model"
    )
    model_temp: float = Field(
        default=0.6,
        description="Temperature for generation"
    )
    model_cfg_coef: float = Field(
        default=2.0,
        description="CFG coefficient for generation"
    )

    # ==========================================
    # Performance Configuration
    # ==========================================
    max_workers: int = Field(
        default=2,
        description="Maximum thread pool workers for synthesis"
    )
    hf_home: str = Field(
        default="/app/models",
        description="HuggingFace cache directory"
    )
    transformers_cache: str = Field(
        default="/app/models",
        description="Transformers cache directory"
    )

    # ==========================================
    # CORS Configuration
    # ==========================================
    cors_origins: str = Field(
        default="*",
        description="CORS allowed origins (comma-separated or *)"
    )
    cors_credentials: bool = Field(
        default=True,
        description="CORS allow credentials"
    )
    cors_methods: str = Field(
        default="*",
        description="CORS allowed methods"
    )
    cors_headers: str = Field(
        default="*",
        description="CORS allowed headers"
    )

    # ==========================================
    # Environment Settings
    # ==========================================
    environment: str = Field(
        default="production",
        description="Environment name (development/staging/production)"
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )

    @property
    def cors_origins_list(self) -> List[str]:
        """
        Parse CORS_ORIGINS into a list.

        Returns:
            List of allowed origins. ["*"] if wildcard is used.
        """
        if self.cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",")]

    @property
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment.lower() == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment.lower() == "development"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,  # Allow HOST or host
        extra="ignore",  # Ignore extra environment variables
        protected_namespaces=()  # Allow model_* field names
    )


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses lru_cache to ensure settings are loaded only once.
    This is the recommended way to access settings throughout the application.

    Returns:
        Settings instance with all configuration loaded
    """
    return Settings()
