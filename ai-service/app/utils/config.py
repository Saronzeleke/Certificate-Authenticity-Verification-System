import os
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


class Settings(BaseSettings):
    # API Configuration
    api_key: str = Field(default_factory=lambda: os.getenv("API_KEY", "fallback-key-12345"))
    environment: str = Field(default="development")

    # Database Configuration
    database_url: str = Field(
        default="postgresql://serveease:StrongP@ssw0rd!@localhost:5432/certificate_db"
    )

    # External Services
    cloudinary_url: str = Field(default="")

    # AI Model Configuration
    ai_service_url: str = Field(default="http://localhost:8000")

    # Certificate Analysis Thresholds
    reject_threshold: float = Field(default=0.75)
    low_quality_threshold: float = Field(default=0.7)

    # OCR Configuration
    ocr_languages: List[str] = Field(default_factory=lambda: ["eng", "amh"])
    ocr_quality_weight: float = Field(default=0.4)
    tampering_weight: float = Field(default=0.4)
    field_completeness_weight: float = Field(default=0.2)

    # Security
    analysis_id_salt: str = Field(default="serveease-cert-salt-2024")

    # Redis Cache
    redis_url: str = Field(default="redis://localhost:6379/0")
    cache_ttl: int = Field(default=3600)

    # Performance
    max_file_size: int = Field(default=10 * 1024 * 1024)
    max_batch_size: int = Field(default=5)
    processing_timeout: int = Field(default=120)
    upload_timeout: int = Field(default=30)

    # Allowed File Types
    allowed_extensions: List[str] = Field(
        default_factory=lambda: ["pdf", "jpg", "jpeg", "png", "tiff", "bmp", "heic", "webp"]
    )

    # Models
    tamper_model_name: str = Field(default="facebook/dinov2-small")
    ocr_model_name: str = Field(default="microsoft/trocr-base-handwritten")

    # Storage
    temp_dir: str = Field(default="/tmp/serveease")
    keep_temp_files: bool = Field(default=False)

    # Monitoring
    enable_metrics: bool = Field(default=True)
    log_level: str = Field(default="INFO")

    # Admin
    admin_email: Optional[str] = Field(default=None)
    auto_reject_enabled: bool = Field(default=True)
    notify_on_fraud: bool = Field(default=True)

    # Pydantic Settings Config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # =========================
    # Pydantic V2 Validators
    # =========================
    @field_validator("allowed_extensions", mode="before")
    def validate_allowed_ext(cls, v):
        if isinstance(v, str):
            return [ext.strip().lower() for ext in v.split(",")]
        return v

    @field_validator("ocr_languages", mode="before")
    def validate_languages(cls, v):
        if isinstance(v, str):
            return [lang.strip().lower() for lang in v.split(",")]
        return v

    @field_validator("reject_threshold", "low_quality_threshold")
    def validate_threshold_range(cls, v, info):
        if not 0 <= v <= 1:
            raise ValueError(f"{info.field_name} must be between 0 and 1")
        return v

    @field_validator("database_url")
    def validate_db_url(cls, v):
        if not v.startswith(("postgresql://", "postgres://", "mysql://", "sqlite://")):
            raise ValueError("Invalid database URL format")
        return v

    # Utility Methods
    def is_production(self) -> bool:
        return self.environment.lower() == "production"

    def is_development(self) -> bool:
        return self.environment.lower() == "development"

    def is_testing(self) -> bool:
        return self.environment.lower() == "testing"

    def get_allowed_mime_types(self) -> List[str]:
        mime_map = {
            "pdf": "application/pdf",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "tiff": "image/tiff",
            "bmp": "image/bmp",
            "heic": "image/heic",
            "webp": "image/webp",
        }
        return [mime_map.get(ext, f"image/{ext}") for ext in self.allowed_extensions]


# ================================
# CREATE SETTINGS INSTANCE HERE
# ================================
settings = Settings()
