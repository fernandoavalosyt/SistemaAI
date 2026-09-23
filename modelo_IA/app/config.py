"""
Vigilia — modelo_IA | Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Configuración centralizada usando Pydantic Settings.
Lee variables del .env y del entorno del contenedor.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuración del servicio de IA."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ---- Modelo YOLO ---------------------------------------------------------
    YOLO_MODEL: str = "yolov8n.pt"
    YOLO_CONFIDENCE: float = 0.5
    YOLO_IOU_THRESHOLD: float = 0.45

    # ---- ByteTrack -----------------------------------------------------------
    TRACK_BUFFER: int = 30
    TRACK_THRESH: float = 0.5
    MATCH_THRESH: float = 0.8

    # ---- Processing ----------------------------------------------------------
    FRAME_SKIP: int = 2
    MAX_CONCURRENT_STREAMS: int = 4
    BATCH_SIZE: int = 8

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

    # ---- Gemini AI -----------------------------------------------------------
    GEMINI_API_KEY: str = ""


settings = Settings()
