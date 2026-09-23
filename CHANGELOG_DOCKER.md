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
