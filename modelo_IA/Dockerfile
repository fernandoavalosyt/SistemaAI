###############################################################################
# Vigilia — modelo_IA Dockerfile
# Pipeline de vision (S1..S9) + Visualizador Byrack (S10)
###############################################################################

# ---------------------------------------------------------------------------
# GPU NVIDIA: usar una imagen base CUDA, instalar torch con el indice cu12x
# (https://download.pytorch.org/whl/cu121) y descomentar el bloque `deploy`
# de docker-compose.yml. Por defecto: CPU.
# ---------------------------------------------------------------------------
# bookworm fijo: libgl1/libglib2.0-0 cambian de nombre en versiones posteriores.
FROM python:3.11-slim-bookworm

LABEL maintainer="Vigilia Team"
LABEL description="Vigilia AI pipeline — Video, YOLO, Tracking, Pose, Behavior, Decision, Evidence, Alerts, Visualizer"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgl1: ultralytics arrastra opencv-python (no headless), que la necesita.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install -r requirements.txt \
    # ultralytics instala opencv-python; dejar solo la variante headless evita
    # que ambos paquetes se pisen el modulo cv2.
    && pip uninstall -y opencv-python \
    && pip install --force-reinstall --no-deps opencv-python-headless

COPY . .

# 8001: visualizador (/stream) + /health   |   8080: servicio de alertas
EXPOSE 8001 8080

# start-period amplio: el primer arranque descarga los pesos de YOLO.
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -fsS http://localhost:8001/health || exit 1

CMD ["python", "main_orchestrator.py"]
