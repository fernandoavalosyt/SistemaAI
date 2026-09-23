###############################################################################
# Vigilia — modelo_IA Dockerfile
# Servicio de Inteligencia Artificial (YOLO, OpenCV, ByteTrack, Gemini)
###############################################################################

# ---------------------------------------------------------------------------
# Para GPU NVIDIA, descomentar la siguiente línea y comentar la de CPU:
# FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS base
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS base

# Metadatos
LABEL maintainer="Vigilia Team"
LABEL description="Vigilia AI Service — YOLO, OpenCV, ByteTrack, Gemini"

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Instalar dependencias del sistema para OpenCV y video processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgstreamer1.0-0 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Directorio de trabajo
WORKDIR /app

# Copiar e instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY . .

# Puerto del servicio
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Comando de inicio (desarrollo con hot-reload)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]
