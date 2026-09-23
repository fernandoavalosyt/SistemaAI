"""
Vigilia — modelo_IA | Main Application
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Entry point del servicio de Inteligencia Artificial.
Expone endpoints de health/status y arranca los workers de procesamiento.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from loguru import logger
import os
import cv2
import tempfile

from app.config import settings


# =============================================================================
# Lifespan — Startup / Shutdown
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestiona el ciclo de vida de la aplicación."""
    logger.info("🧠 Vigilia IA — Iniciando servicio...")
    logger.info(f"   Modelo YOLO: {settings.YOLO_MODEL}")
    logger.info(f"   RabbitMQ: {settings.RABBITMQ_HOST}:{settings.RABBITMQ_PORT}")
    logger.info(f"   Redis: {settings.REDIS_HOST}:{settings.REDIS_PORT}")
    logger.info(f"   MinIO: {settings.MINIO_ENDPOINT}")

    # TODO: Inicializar conexiones
    # - Cargar modelo YOLO
    # - Conectar a RabbitMQ (consumers)
    # - Conectar a Redis (cache)
    # - Conectar a MinIO (storage)

    yield

    # Shutdown
    logger.info("🧠 Vigilia IA — Cerrando servicio...")
    # TODO: Cerrar conexiones gracefully


# =============================================================================
# FastAPI App
# =============================================================================
app = FastAPI(
    title="Vigilia IA Service",
    description="Servicio de Inteligencia Artificial — YOLO, ByteTrack, Behavior Engine, Risk Score",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
        "service": "modelo-ia",
        "version": "0.1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/status", tags=["Health"])
async def service_status():
    """Estado detallado del servicio."""
    return {
        "service": "modelo-ia",
        "version": "0.1.0",
        "model": settings.YOLO_MODEL,
        "confidence": settings.YOLO_CONFIDENCE,
        "max_streams": settings.MAX_CONCURRENT_STREAMS,
        "connections": {
            "rabbitmq": settings.RABBITMQ_HOST,
            "redis": settings.REDIS_HOST,
            "minio": settings.MINIO_ENDPOINT,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def process_video_task(input_path: str, output_path: str):
    """Tarea en background para procesar el video frame por frame."""
    from app.core.pipeline import VideoPipeline
    logger.info(f"Iniciando procesamiento de video: {input_path}")
    
    pipeline = VideoPipeline()
    cap = cv2.VideoCapture(input_path)
    
    # Propiedades del video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        annotated_frame, _, _ = pipeline.process_frame(frame)
        out.write(annotated_frame)
        
        frame_count += 1
        if frame_count % 30 == 0:
            logger.info(f"Procesado {frame_count} frames...")
            
    cap.release()
    out.release()
    logger.info(f"Video procesado guardado en: {output_path}")

@app.post("/test-video", tags=["Testing"])
async def test_video_upload(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Sube un video mp4 para probar la IA localmente."""
    
    temp_dir = tempfile.gettempdir()
    input_path = os.path.join(temp_dir, f"in_{file.filename}")
    output_path = os.path.join(temp_dir, f"out_{file.filename}")
    
    with open(input_path, "wb") as buffer:
        buffer.write(await file.read())
        
    background_tasks.add_task(process_video_task, input_path, output_path)
    
    return {
        "message": "Procesamiento iniciado en background",
        "input_file": input_path,
        "output_file": output_path
    }
