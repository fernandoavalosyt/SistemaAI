"""
Vigilia — Backend | Main Application
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Entry point del servicio Backend.
FastAPI con CORS, WebSockets, y estructura de routers.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.config import settings


# =============================================================================
# Lifespan — Startup / Shutdown
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestiona el ciclo de vida de la aplicación."""
    logger.info("⚙️  Vigilia Backend — Iniciando servicio...")
    logger.info(f"   PostgreSQL: {settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}")
    logger.info(f"   MongoDB: {settings.MONGO_HOST}:{settings.MONGO_PORT}")
    logger.info(f"   Redis: {settings.REDIS_HOST}:{settings.REDIS_PORT}")
    logger.info(f"   RabbitMQ: {settings.RABBITMQ_HOST}:{settings.RABBITMQ_PORT}")
    logger.info(f"   MinIO: {settings.MINIO_ENDPOINT}")

    # TODO: Inicializar conexiones
    # - Pool de PostgreSQL (SQLAlchemy async)
    # - Cliente MongoDB (Motor)
    # - Cliente Redis
    # - Conexión RabbitMQ (consumers)
    # - Cliente MinIO

    yield

    # Shutdown
    logger.info("⚙️  Vigilia Backend — Cerrando servicio...")
    # TODO: Cerrar conexiones gracefully


# =============================================================================
# FastAPI App
# =============================================================================
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Vigilia Backend — API REST + WebSockets para videovigilancia inteligente",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — permitir frontend en desarrollo
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://vigilia-frontend:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Endpoints
# =============================================================================
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check para Docker y load balancers."""
    return {
        "status": "healthy",
        "service": "backend",
        "version": "0.1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/status", tags=["Health"])
async def service_status():
    """Estado detallado del servicio."""
    return {
        "service": "backend",
        "version": "0.1.0",
        "debug": settings.DEBUG,
        "connections": {
            "postgres": f"{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}",
            "mongodb": f"{settings.MONGO_HOST}:{settings.MONGO_PORT}",
            "redis": f"{settings.REDIS_HOST}:{settings.REDIS_PORT}",
            "rabbitmq": f"{settings.RABBITMQ_HOST}:{settings.RABBITMQ_PORT}",
            "minio": settings.MINIO_ENDPOINT,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# =============================================================================
# API Routers (se irán agregando conforme se desarrolle)
# =============================================================================
# from app.api.v1 import auth, cameras, alerts, analytics
# app.include_router(auth.router, prefix="/api/v1/auth", tags=["Auth"])
# app.include_router(cameras.router, prefix="/api/v1/cameras", tags=["Cameras"])
# app.include_router(alerts.router, prefix="/api/v1/alerts", tags=["Alerts"])
