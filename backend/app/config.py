"""
Vigilia — Backend | Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Configuración centralizada usando Pydantic Settings.
Lee variables del .env y del entorno del contenedor.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuración del servicio Backend."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ---- API Config ----------------------------------------------------------
    PROJECT_NAME: str = "Vigilia API"
    API_V1_PREFIX: str = "/api/v1"
    DEBUG: bool = True

    # ---- PostgreSQL ----------------------------------------------------------
    POSTGRES_HOST: str = "postgres"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "vigilia"
    POSTGRES_PASSWORD: str = "vigilia_secret"
    POSTGRES_DB: str = "vigilia_db"
    DATABASE_URL: str = "postgresql+asyncpg://vigilia:vigilia_secret@postgres:5432/vigilia_db"

    # ---- MongoDB -------------------------------------------------------------
    MONGO_HOST: str = "mongodb"
    MONGO_PORT: int = 27017
    MONGO_USER: str = "vigilia"
    MONGO_PASSWORD: str = "vigilia_secret"
    MONGO_DB: str = "vigilia_db"
    MONGO_URL: str = "mongodb://vigilia:vigilia_secret@mongodb:27017/vigilia_db?authSource=admin"

    # ---- Redis ---------------------------------------------------------------
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = "vigilia_secret"
    REDIS_URL: str = "redis://:vigilia_secret@redis:6379/0"

    # ---- RabbitMQ ------------------------------------------------------------
    RABBITMQ_HOST: str = "rabbitmq"
    RABBITMQ_PORT: int = 5672
    RABBITMQ_USER: str = "vigilia"
    RABBITMQ_PASSWORD: str = "vigilia_secret"
    RABBITMQ_URL: str = "amqp://vigilia:vigilia_secret@rabbitmq:5672/"

    # ---- MinIO ---------------------------------------------------------------
    MINIO_ENDPOINT: str = "minio:9000"
    MINIO_ACCESS_KEY: str = "vigilia"
    MINIO_SECRET_KEY: str = "vigilia123"
    MINIO_BUCKET_VIDEOS: str = "vigilia-videos"
    MINIO_BUCKET_FRAMES: str = "vigilia-frames"
    MINIO_BUCKET_RESULTS: str = "vigilia-results"

    # ---- JWT -----------------------------------------------------------------
    JWT_SECRET_KEY: str = "super-secret-dev-key-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # ---- IA Service ----------------------------------------------------------
    IA_SERVICE_URL: str = "http://modelo-ia:8001"


settings = Settings()
