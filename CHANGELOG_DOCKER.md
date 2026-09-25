# Vigilia — Registro de Cambios Docker

> Cada sección documenta qué se hizo, qué archivo se creó/modificó y por qué.

---

## 2026-09-19 — Configuración inicial de infraestructura Docker

### Archivos Root creados

| Archivo | Qué hace |
|---------|----------|
| `.env` | Variables de entorno (credenciales dev, puertos, DSNs) |
| `.env.example` | Template de referencia sin valores sensibles |
| `.dockerignore` | Excluye archivos innecesarios del build Docker |
| `.gitignore` | Ignora archivos generados y sensibles en Git |
| `docker-compose.yml` | Orquesta los 8 servicios (IA, backend, frontend, postgres, mongodb, redis, rabbitmq, minio) |
| `docker-compose.override.yml` | Overrides para desarrollo local (hot-reload, volumes) |

### modelo_IA/ — Servicio de Inteligencia Artificial

| Archivo | Qué hace |
|---------|----------|
| `Dockerfile` | Imagen Python 3.11 (opción CUDA comentada para futuro con GPU NVIDIA) |
| `requirements.txt` | Dependencias: YOLO, OpenCV, FastAPI, RabbitMQ, Redis, MinIO, Gemini |
| `app/main.py` | Entry point FastAPI con health check |
| `app/config.py` | Configuración con Pydantic (conexiones a servicios) |
| `app/__init__.py` | Marca el directorio como paquete Python |
| `app/core/__init__.py` | Marca core/ como paquete Python |

### backend/ — Servicio Backend API

| Archivo | Qué hace |
|---------|----------|
| `Dockerfile` | Imagen Python 3.11 con uvicorn hot-reload |
| `requirements.txt` | Dependencias: FastAPI, SQLAlchemy, Alembic, MongoDB, Redis, JWT |
| `alembic.ini` | Configuración de migraciones de base de datos |
| `app/main.py` | Entry point FastAPI con CORS y health check |
| `app/config.py` | Configuración de conexiones a toda la infraestructura |
| `app/__init__.py` | Marca el directorio como paquete Python |

### frontend/ — Servicio Frontend

| Archivo | Qué hace |
|---------|----------|
| `Dockerfile` | Node 20 con multi-stage build (dev + producción con Nginx) |
| `nginx.conf` | Configuración Nginx para SPA en producción |
| Proyecto Vite | React + TypeScript inicializado con `npm create vite@latest` |

---

### Notas importantes

- **GPU NVIDIA**: No disponible actualmente. La línea CUDA está **comentada** en `modelo_IA/Dockerfile`. Cuando tengas GPU, descomenta la línea `FROM nvidia/cuda:...` y comenta `FROM python:3.11-slim`.
- **Versiones usadas**: Python 3.11, Node 20, PostgreSQL 16, MongoDB 7, Redis 7, RabbitMQ 3.
- **Para levantar todo**: `docker compose up -d` desde la raíz del proyecto.

---

## 2026-09-24 — Pipeline de IA (S1..S9) + Visualizador Byrack (S10)

### modelo_IA/

| Archivo | Qué hace |
|---------|----------|
| `main_orchestrator.py` | Nuevo entry point: instancia S1..S9, los conecta con hilos "pump" y levanta el visualizador |
| `IA/visualizer/visualizerService.py` | S10: dibuja tracks, esqueleto, comportamientos, alertas y evidencia sobre el frame original; MJPEG en `:8001/stream` |
| `IA/*` | Carpetas renombradas sin guiones (`video_ingestion`, `yolo_detection`, …) + `__init__.py` para poder importarlas |
| `IA/video_ingestion/videoService.py` | Cámara IP por RTSP activa (`RTSP_URL`, `RTSP_USER`, `RTSP_PASSWORD`); `frame_id` ya no se reinicia al reconectar |
| `Dockerfile` / `requirements.txt` | Recreados: Python 3.11 (bookworm), torch CPU, OpenCV headless, Flask; `CMD python main_orchestrator.py` |
| `.dockerignore` | Excluye `venv/`, `evidence_store/` y pesos del contexto de build |
| `config/zones.json`, `config/camera_context.json` | Zonas de ejemplo (1920x1080) con los mismos `zone_id`; salida con merodeo a 3 s |
| `.env.example` | Variables del pipeline |

### Raíz

| Archivo | Cambio |
|---------|--------|
| `docker-compose.yml` | `modelo-ia`: puerto 8080 (alertas), `VISUALIZER_HOST/ALERT_HOST=0.0.0.0`, `ALERT_OPEN_BROWSER=false`, `stop_grace_period: 30s` |
| `docker-compose.override.yml` | `modelo-ia` arranca con `python main_orchestrator.py` (antes `uvicorn app.main:app`, que ya no existe) |
| `.env.example` | `ALERT_PORT` |
| `.gitignore` | `modelo_IA/evidence_store/` |

### Notas

- El puerto 8001 se mantiene: backend (`IA_SERVICE_URL`) y frontend (`/health`) ya apuntan ahí. Frontend: `<img src="http://localhost:8001/stream" />`.
- Los pesos `yolov8n.pt` y `yolov8n-pose.pt` se descargan solos al primer arranque dentro del volumen `ia-models` (`AUTO_DOWNLOAD_MODELS=true`).
