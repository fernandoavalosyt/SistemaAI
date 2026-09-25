"""
evidenceService.py

Servicio de Event / Evidence (Servicio 8 del pipeline).

Responsabilidad UNICA: recibir un DecisionResult (EVENT_GENERATED) desde el
"Decision Engine" (Servicio 7), recolectar el video alrededor del evento
(pre-roll desde un buffer circular + post-roll esperado de forma asincrona),
ensamblar un clip, almacenarlo junto con su metadata de forma integra
(SHA-256) y emitir un EvidenceResult hacia el "Alert Service" (Servicio 9).

Este servicio NO ejecuta modelos de IA, NO decide si algo es un robo (eso ya
lo decidio el Decision Engine) y NO envia alertas ni notificaciones.

Concurrencia: todo el trabajo corre en un event loop de asyncio dentro de un
hilo dedicado. submit_decision() retorna inmediatamente (nunca bloquea al
hilo llamador esperando el post-roll); la espera, el render (en thread pool)
y el guardado ocurren en tareas asincronas independientes por evento.
"""

from __future__ import annotations

import abc
import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import stat
import tempfile
import threading
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, Union

import numpy as np


# =============================================================================
# Logging
# =============================================================================
# Nunca se loguea contenido binario (frames, clips) ni payloads completos.

logger = logging.getLogger("evidence_service")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


# =============================================================================
# Excepciones y enums
# =============================================================================

class InputValidationError(Exception):
    """DecisionResult malformado o huerfano."""


class StorageError(Exception):
    """Fallo del proveedor de almacenamiento."""


class RenderError(Exception):
    """Fallo al ensamblar el clip."""


class EvidenceState(str, Enum):
    CREATED = "CREATED"
    COLLECTING = "COLLECTING"   # esperando el post-roll
    PROCESSING = "PROCESSING"   # render + almacenamiento
    READY = "READY"
    FAILED = "FAILED"


_ALLOWED_TRANSITIONS: Dict[EvidenceState, Tuple[EvidenceState, ...]] = {
    EvidenceState.CREATED: (EvidenceState.COLLECTING, EvidenceState.READY, EvidenceState.FAILED),
    EvidenceState.COLLECTING: (EvidenceState.PROCESSING, EvidenceState.FAILED),
    EvidenceState.PROCESSING: (EvidenceState.READY, EvidenceState.FAILED),
    EvidenceState.READY: (),
    EvidenceState.FAILED: (),
}

_VALID_PRIORITIES = ("LOW", "MEDIUM", "HIGH", "CRITICAL")
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")
_EVIDENCE_NAMESPACE = uuid.UUID("b3e7a3f2-2c1d-4e6a-9f0b-5d8c7e6a1f24")


def evidence_id_for(event_id: str) -> str:
    """evidence_id determinista: el mismo evento siempre produce la misma
    evidencia, incluso despues de un reinicio del servicio."""
    return str(uuid.uuid5(_EVIDENCE_NAMESPACE, event_id))


# =============================================================================
# Configuracion
# =============================================================================

@dataclass(frozen=True)
class EvidenceConfig:
    storage_root: str = "evidence_store"
    pre_roll_seconds: float = 15.0
    post_roll_seconds: float = 10.0
    max_clip_seconds: float = 120.0          # tope de seguridad (RAM/disco)
    collect_grace_seconds: float = 5.0       # espera extra si el video llega con retraso
    poll_interval_seconds: float = 0.25
    min_coverage_ratio: float = 0.3          # por debajo -> FAILED (insufficient_video)
    gap_threshold_seconds: float = 1.0
    max_concurrent_renders: int = 2          # el render es CPU-bound
    max_pending_events: int = 200
    completed_cache_size: int = 5000
    output_queue_size: int = 1000
    default_fps: float = 10.0
    buffer_retention_seconds: float = 45.0   # >= pre_roll + post_roll + margen
    buffer_max_bytes_per_camera: int = 150 * 1024 * 1024
    jpeg_quality: int = 80
    max_explanation_bytes: int = 16384

    @staticmethod
    def from_env() -> "EvidenceConfig":
        return EvidenceConfig(
            storage_root=os.getenv("EVIDENCE_STORAGE_ROOT", "evidence_store"),
            pre_roll_seconds=float(os.getenv("EVIDENCE_PRE_ROLL_SECONDS", "15")),
            post_roll_seconds=float(os.getenv("EVIDENCE_POST_ROLL_SECONDS", "10")),
            max_clip_seconds=float(os.getenv("EVIDENCE_MAX_CLIP_SECONDS", "120")),
            collect_grace_seconds=float(os.getenv("EVIDENCE_COLLECT_GRACE_SECONDS", "5")),
            max_concurrent_renders=int(os.getenv("EVIDENCE_MAX_CONCURRENT_RENDERS", "2")),
            max_pending_events=int(os.getenv("EVIDENCE_MAX_PENDING_EVENTS", "200")),
            buffer_retention_seconds=float(os.getenv("EVIDENCE_BUFFER_RETENTION_SECONDS", "45")),
            buffer_max_bytes_per_camera=int(os.getenv("EVIDENCE_BUFFER_MAX_BYTES", str(150 * 1024 * 1024))),
        )


# =============================================================================
# Buffer de video (pasado reciente por camara)
# =============================================================================

@dataclass(frozen=True)
class BufferedFrame:
    timestamp: float
    data: bytes            # JPEG
    width: int
    height: int


class VideoBufferManager(abc.ABC):
    """Mantiene los ultimos N segundos de video por camara. Es la fuente
    del pre-roll: sin el, el video anterior al evento ya no existiria."""

    @abc.abstractmethod
    def append_frame(self, camera_id: str, timestamp: float, frame: Union[np.ndarray, bytes],
                     width: int = 0, height: int = 0) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def get_segment(self, camera_id: str, start: float, end: float) -> List[BufferedFrame]:
        raise NotImplementedError

    @abc.abstractmethod
    def latest_timestamp(self, camera_id: str) -> Optional[float]:
        raise NotImplementedError

    @abc.abstractmethod
    def oldest_timestamp(self, camera_id: str) -> Optional[float]:
        raise NotImplementedError

    @abc.abstractmethod
    def stats(self) -> Dict[str, Any]:
        raise NotImplementedError


class _CameraRing:
    def __init__(self) -> None:
        self.frames: Deque[BufferedFrame] = deque()
        self.bytes = 0
        self.lock = threading.Lock()


class InMemoryRingBuffer(VideoBufferManager):
    """Ring buffer en memoria por camara, acotado por tiempo Y por bytes.
    Los frames se guardan comprimidos (JPEG) para reducir RAM ~10-20x.
    append_frame() es thread-safe y se invoca desde el hilo de ingesta."""

    def __init__(self, retention_seconds: float, max_bytes_per_camera: int, jpeg_quality: int = 80) -> None:
        self._retention = retention_seconds
        self._max_bytes = max_bytes_per_camera
        self._jpeg_quality = jpeg_quality
        self._rings: Dict[str, _CameraRing] = {}
        self._rings_lock = threading.Lock()

    def _ring(self, camera_id: str) -> _CameraRing:
        with self._rings_lock:
            ring = self._rings.get(camera_id)
            if ring is None:
                ring = _CameraRing()
                self._rings[camera_id] = ring
            return ring

    def _encode(self, frame: np.ndarray) -> Tuple[bytes, int, int]:
        import cv2  # import diferido: solo para comprimir frames entrantes
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality])
        if not ok:
            raise ValueError("No se pudo codificar el frame a JPEG")
        return encoded.tobytes(), int(frame.shape[1]), int(frame.shape[0])

    def append_frame(self, camera_id: str, timestamp: float, frame: Union[np.ndarray, bytes],
                     width: int = 0, height: int = 0) -> None:
        if isinstance(frame, np.ndarray):
            if frame.ndim != 3 or frame.size == 0:
                return
            data, width, height = self._encode(frame)
        elif isinstance(frame, (bytes, bytearray)) and frame:
            data = bytes(frame)
        else:
            return

        ring = self._ring(camera_id)
        with ring.lock:
            if ring.frames and timestamp <= ring.frames[-1].timestamp:
                return  # fuera de orden / duplicado: se ignora
            ring.frames.append(BufferedFrame(timestamp, data, width, height))
            ring.bytes += len(data)
            cutoff = timestamp - self._retention
            while ring.frames and (ring.frames[0].timestamp < cutoff or ring.bytes > self._max_bytes):
                ring.bytes -= len(ring.frames.popleft().data)

    def get_segment(self, camera_id: str, start: float, end: float) -> List[BufferedFrame]:
        ring = self._ring(camera_id)
        with ring.lock:
            # Copia de referencias (bytes inmutables): barato y seguro fuera del lock.
            return [f for f in ring.frames if start <= f.timestamp <= end]

    def latest_timestamp(self, camera_id: str) -> Optional[float]:
        ring = self._ring(camera_id)
        with ring.lock:
            return ring.frames[-1].timestamp if ring.frames else None

    def oldest_timestamp(self, camera_id: str) -> Optional[float]:
        ring = self._ring(camera_id)
        with ring.lock:
            return ring.frames[0].timestamp if ring.frames else None

    def stats(self) -> Dict[str, Any]:
        with self._rings_lock:
            rings = dict(self._rings)
        result: Dict[str, Any] = {}
        for camera_id, ring in rings.items():
            with ring.lock:
                span = ring.frames[-1].timestamp - ring.frames[0].timestamp if len(ring.frames) > 1 else 0.0
                result[camera_id] = {
                    "frames": len(ring.frames),
                    "seconds_buffered": round(span, 2),
                    "megabytes": round(ring.bytes / (1024 * 1024), 2),
                }
        return result


# -----------------------------------------------------------------------------
# NOTA ARQUITECTONICA: backends de buffer preparados, NO activos.
# Deben respetar el contrato de VideoBufferManager; el servicio no cambia.
# -----------------------------------------------------------------------------

class NVRBufferManager(VideoBufferManager):
    """[STUB] Recupera el segmento directamente del NVR/DVR via su API de
    playback por rango de tiempo (el NVR ya graba de forma continua, por lo
    que no se necesita ring buffer propio). Credenciales SOLO desde
    variables de entorno o gestor de secretos (ej. NVR_URL, NVR_USER,
    NVR_PASSWORD); nunca hardcodeadas ni logueadas."""

    def append_frame(self, camera_id: str, timestamp: float, frame: Union[np.ndarray, bytes],
                     width: int = 0, height: int = 0) -> None:
        pass  # el NVR graba por su cuenta

    def get_segment(self, camera_id: str, start: float, end: float) -> List[BufferedFrame]:
        raise NotImplementedError("NVRBufferManager aun no esta implementado.")

    def latest_timestamp(self, camera_id: str) -> Optional[float]:
        raise NotImplementedError("NVRBufferManager aun no esta implementado.")

    def oldest_timestamp(self, camera_id: str) -> Optional[float]:
        raise NotImplementedError("NVRBufferManager aun no esta implementado.")

    def stats(self) -> Dict[str, Any]:
        return {"backend": "NVR", "implemented": False}


class RedisBufferManager(VideoBufferManager):
    """[STUB] Ring buffer compartido en Redis (ej. un Stream por camara con
    MAXLEN aproximado y frames JPEG como valor). Permite que ingesta y
    evidencia corran en procesos/maquinas distintas. Conexion via REDIS_URL."""

    def append_frame(self, camera_id: str, timestamp: float, frame: Union[np.ndarray, bytes],
                     width: int = 0, height: int = 0) -> None:
        raise NotImplementedError("RedisBufferManager aun no esta implementado.")

    def get_segment(self, camera_id: str, start: float, end: float) -> List[BufferedFrame]:
        raise NotImplementedError("RedisBufferManager aun no esta implementado.")

    def latest_timestamp(self, camera_id: str) -> Optional[float]:
        raise NotImplementedError("RedisBufferManager aun no esta implementado.")

    def oldest_timestamp(self, camera_id: str) -> Optional[float]:
        raise NotImplementedError("RedisBufferManager aun no esta implementado.")

    def stats(self) -> Dict[str, Any]:
        return {"backend": "REDIS", "implemented": False}


# =============================================================================
# Ensamblado del clip
# =============================================================================

@dataclass(frozen=True)
class RenderedClip:
    path: str
    format: str
    mime_type: str
    frame_count: int
    fps: float
    width: int
    height: int
    corrupt_frames_skipped: int


class ClipRenderer(abc.ABC):
    """Convierte una lista de frames en un archivo. Es CPU-bound: el
    servicio siempre lo ejecuta en un thread pool (asyncio.to_thread)."""

    @abc.abstractmethod
    def render(self, frames: List[BufferedFrame], output_base_path: str, fps: float) -> RenderedClip:
        raise NotImplementedError


class OpenCVClipRenderer(ClipRenderer):
    """Clip MP4 real (codec mp4v) via OpenCV VideoWriter."""

    def render(self, frames: List[BufferedFrame], output_base_path: str, fps: float) -> RenderedClip:
        import cv2

        path = output_base_path + ".mp4"
        writer = None
        width = height = written = corrupt = 0
        try:
            for frame in frames:
                image = cv2.imdecode(np.frombuffer(frame.data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    corrupt += 1
                    continue
                if writer is None:
                    height, width = image.shape[:2]
                    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                    if not writer.isOpened():
                        raise RenderError("cv2.VideoWriter no pudo abrir el archivo de salida")
                if image.shape[1] != width or image.shape[0] != height:
                    image = cv2.resize(image, (width, height))
                writer.write(image)
                written += 1
        finally:
            if writer is not None:
                writer.release()
        if written == 0:
            raise RenderError("Ningun frame valido para renderizar")
        return RenderedClip(path, "mp4", "video/mp4", written, fps, width, height, corrupt)


class SimulatedClipRenderer(ClipRenderer):
    """Contenedor MJPEG simple (cabecera JSON + frames JPEG con prefijo de
    longitud). Sin dependencias; util en entornos sin codecs o para pruebas."""

    def render(self, frames: List[BufferedFrame], output_base_path: str, fps: float) -> RenderedClip:
        path = output_base_path + ".mjpeg"
        if not frames:
            raise RenderError("Ningun frame para renderizar")
        header = json.dumps({
            "container": "vigilia-mjpeg-v1", "fps": fps, "frames": len(frames),
            "start": frames[0].timestamp, "end": frames[-1].timestamp,
        }).encode("utf-8")
        with open(path, "wb") as handle:
            handle.write(len(header).to_bytes(4, "big"))
            handle.write(header)
            for frame in frames:
                handle.write(frame.timestamp.hex().encode("ascii").ljust(32, b" "))
                handle.write(len(frame.data).to_bytes(8, "big"))
                handle.write(frame.data)
        return RenderedClip(path, "mjpeg", "application/octet-stream", len(frames), fps,
                            frames[0].width, frames[0].height, 0)


class AutoClipRenderer(ClipRenderer):
    """MP4 si OpenCV esta disponible; si falla, degrada al contenedor MJPEG
    en lugar de perder la evidencia."""

    def __init__(self) -> None:
        self._primary = OpenCVClipRenderer()
        self._fallback = SimulatedClipRenderer()

    def render(self, frames: List[BufferedFrame], output_base_path: str, fps: float) -> RenderedClip:
        try:
            return self._primary.render(frames, output_base_path, fps)
        except (ImportError, RenderError) as exc:
            logger.warning("Render MP4 no disponible (%s); usando contenedor MJPEG", exc)
            return self._fallback.render(frames, output_base_path, fps)


# =============================================================================
# Almacenamiento
# =============================================================================

@dataclass(frozen=True)
class StoredObject:
    reference: str        # referencia estable (local://..., s3://...)
    size_bytes: int
    sha256: str


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class EvidenceStorageProvider(abc.ABC):
    backend_name: str = "abstract"

    @abc.abstractmethod
    def save_video(self, evidence_id: str, camera_id: str, event_time: float, source_path: str,
                   extension: str) -> StoredObject:
        raise NotImplementedError

    @abc.abstractmethod
    def save_metadata(self, evidence_id: str, metadata: Dict[str, Any]) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def load_metadata(self, evidence_id: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_access_url(self, reference: str, ttl_seconds: int = 900) -> str:
        """URL de acceso para el consumidor (Alert Service / frontend). En S3
        seria una URL prefirmada con expiracion."""
        raise NotImplementedError

    @abc.abstractmethod
    def health(self) -> Dict[str, Any]:
        raise NotImplementedError


class LocalFileStorage(EvidenceStorageProvider):
    """Almacenamiento local:
        <root>/<camera_id>/<YYYY-MM-DD>/<evidence_id>.<ext>   (video, solo lectura)
        <root>/_metadata/<evidence_id>.json                    (indice O(1) por evidencia)

    Escrituras atomicas (archivo temporal + os.replace), checksum SHA-256
    calculado en streaming, componentes de ruta validados contra path
    traversal y archivos finales marcados como solo lectura."""

    backend_name = "local"

    def __init__(self, root: str) -> None:
        self._root = os.path.abspath(root)
        self._meta_dir = os.path.join(self._root, "_metadata")
        os.makedirs(self._meta_dir, exist_ok=True)

    def _inside_root(self, path: str) -> str:
        resolved = os.path.abspath(path)
        if os.path.commonpath([resolved, self._root]) != self._root:
            raise StorageError("Ruta fuera del directorio de almacenamiento")
        return resolved

    @staticmethod
    def _safe(component: str) -> str:
        if not _SAFE_COMPONENT.match(component):
            raise StorageError(f"Componente de ruta invalido: {component!r}")
        return component

    def save_video(self, evidence_id: str, camera_id: str, event_time: float, source_path: str,
                   extension: str) -> StoredObject:
        day = datetime.fromtimestamp(event_time, tz=timezone.utc).strftime("%Y-%m-%d")
        directory = self._inside_root(os.path.join(self._root, self._safe(camera_id), day))
        os.makedirs(directory, exist_ok=True)
        final_path = self._inside_root(os.path.join(directory, f"{self._safe(evidence_id)}.{self._safe(extension)}"))
        temp_path = final_path + ".part"

        digest = hashlib.sha256()
        size = 0
        try:
            with open(source_path, "rb") as src, open(temp_path, "wb") as dst:
                for chunk in iter(lambda: src.read(1024 * 1024), b""):
                    digest.update(chunk)
                    dst.write(chunk)
                    size += len(chunk)
                dst.flush()
                os.fsync(dst.fileno())
            if os.path.exists(final_path):
                os.chmod(final_path, stat.S_IWRITE | stat.S_IREAD)
            os.replace(temp_path, final_path)
            os.chmod(final_path, stat.S_IREAD)
        except OSError as exc:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise StorageError(f"No se pudo guardar el video: {exc}") from exc

        relative = os.path.relpath(final_path, self._root).replace(os.sep, "/")
        return StoredObject(reference=f"local://{relative}", size_bytes=size, sha256=digest.hexdigest())

    def save_metadata(self, evidence_id: str, metadata: Dict[str, Any]) -> str:
        path = self._inside_root(os.path.join(self._meta_dir, f"{self._safe(evidence_id)}.json"))
        temp_path = path + ".part"
        try:
            with open(temp_path, "w", encoding="utf-8") as handle:
                json.dump(metadata, handle, ensure_ascii=False, indent=2, default=str)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
        except OSError as exc:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise StorageError(f"No se pudo guardar la metadata: {exc}") from exc
        return f"local://_metadata/{evidence_id}.json"

    def load_metadata(self, evidence_id: str) -> Optional[Dict[str, Any]]:
        path = os.path.join(self._meta_dir, f"{self._safe(evidence_id)}.json")
        if not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def resolve_path(self, reference: str) -> str:
        if not reference.startswith("local://"):
            raise StorageError("Referencia no local")
        return self._inside_root(os.path.join(self._root, reference[len("local://"):]))

    def get_access_url(self, reference: str, ttl_seconds: int = 900) -> str:
        return "file:///" + self.resolve_path(reference).replace(os.sep, "/").lstrip("/")

    def health(self) -> Dict[str, Any]:
        probe = os.path.join(self._root, ".write_probe")
        try:
            with open(probe, "w", encoding="utf-8") as handle:
                handle.write("ok")
            os.remove(probe)
            writable = True
        except OSError:
            writable = False
        usage = shutil.disk_usage(self._root)
        return {"backend": self.backend_name, "writable": writable, "free_gb": round(usage.free / 1024 ** 3, 2)}


class S3EvidenceStorage(EvidenceStorageProvider):
    """[STUB] Almacenamiento en S3/MinIO: put_object con ChecksumSHA256,
    Object Lock (WORM) para cadena de custodia y get_access_url() como URL
    prefirmada con expiracion. Credenciales desde el entorno/rol de IAM,
    jamas en codigo ni en logs."""

    backend_name = "s3"

    def save_video(self, evidence_id: str, camera_id: str, event_time: float, source_path: str,
                   extension: str) -> StoredObject:
        raise NotImplementedError("S3EvidenceStorage aun no esta implementado.")

    def save_metadata(self, evidence_id: str, metadata: Dict[str, Any]) -> str:
        raise NotImplementedError("S3EvidenceStorage aun no esta implementado.")

    def load_metadata(self, evidence_id: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError("S3EvidenceStorage aun no esta implementado.")

    def get_access_url(self, reference: str, ttl_seconds: int = 900) -> str:
        raise NotImplementedError("S3EvidenceStorage aun no esta implementado.")

    def health(self) -> Dict[str, Any]:
        return {"backend": self.backend_name, "implemented": False}


# =============================================================================
# Validacion de entrada
# =============================================================================

def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_decision_result(payload: Any) -> bool:
    """True si es un evento procesable; False si es un NO_EVENT (se ignora
    sin error). Lanza InputValidationError si esta malformado o huerfano."""
    if not isinstance(payload, dict) or not payload:
        raise InputValidationError("DecisionResult nulo o vacio")
    if payload.get("decision") == "NO_EVENT":
        return False
    if payload.get("decision") != "EVENT_GENERATED":
        raise InputValidationError(f"decision invalida: {payload.get('decision')!r}")

    for key in ("event_id", "camera_id", "session_id", "track_id", "event_type", "priority", "timestamp"):
        if payload.get(key) in (None, ""):
            raise InputValidationError(f"Evento huerfano: falta '{key}'")
    try:
        uuid.UUID(str(payload["event_id"]))
    except ValueError as exc:
        raise InputValidationError("event_id no es un UUID valido") from exc
    if not _SAFE_COMPONENT.match(str(payload["camera_id"])):
        raise InputValidationError("camera_id con caracteres no permitidos")
    if payload["priority"] not in _VALID_PRIORITIES:
        raise InputValidationError(f"priority invalida: {payload['priority']!r}")
    if not _is_number(payload["timestamp"]) or payload["timestamp"] <= 0:
        raise InputValidationError("timestamp invalido")
    start, end = payload.get("time_window_start"), payload.get("time_window_end")
    if start is not None and not _is_number(start):
        raise InputValidationError("time_window_start invalido")
    if end is not None and not _is_number(end):
        raise InputValidationError("time_window_end invalido")
    if start is not None and end is not None and end <= start:
        raise InputValidationError("Ventana temporal invalida (end <= start)")
    if not isinstance(payload.get("explanation"), dict):
        raise InputValidationError("Evento sin explicacion (explanation) de la decision")
    return True


# =============================================================================
# Trabajo de evidencia y contrato de salida
# =============================================================================

@dataclass
class EvidenceResult:
    evidence_id: str
    event_id: str
    camera_id: str
    session_id: str
    track_id: str
    event_type: str
    priority: str
    video_reference: Optional[str]
    start_time: float
    end_time: float
    duration: float
    size: int
    format: Optional[str]
    status: EvidenceState
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "event_id": self.event_id,
            "camera_id": self.camera_id,
            "session_id": self.session_id,
            "track_id": self.track_id,
            "event_type": self.event_type,
            "priority": self.priority,
            "video_reference": self.video_reference,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": round(self.duration, 3),
            "size": self.size,
            "format": self.format,
            "status": self.status.value,
            "metadata": self.metadata,
        }


@dataclass
class EvidenceJob:
    event_id: str
    evidence_id: str
    decision: Dict[str, Any]
    start_time: float
    end_time: float
    state: EvidenceState = EvidenceState.CREATED
    created_wall: float = field(default_factory=time.time)
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    task: Optional["asyncio.Task[None]"] = None

    def transition(self, new_state: EvidenceState, detail: str = "") -> None:
        if new_state not in _ALLOWED_TRANSITIONS[self.state]:
            raise RuntimeError(f"Transicion invalida {self.state.value} -> {new_state.value}")
        self.state = new_state
        self.timeline.append({"state": new_state.value, "at": time.time(), "detail": detail})
        logger.info("Evento %s inicio %s%s", self.event_id, new_state.value, f" ({detail})" if detail else "")


def analyze_coverage(frames: List[BufferedFrame], start: float, end: float, gap_threshold: float) -> Dict[str, Any]:
    requested = max(end - start, 1e-6)
    if not frames:
        return {"requested_seconds": round(requested, 3), "covered_seconds": 0.0, "coverage_ratio": 0.0,
                "gaps": 0, "gap_seconds": 0.0, "partial_pre_roll": True, "partial_post_roll": True}
    gaps = gap_seconds = 0.0
    for previous, current in zip(frames, frames[1:]):
        delta = current.timestamp - previous.timestamp
        if delta > gap_threshold:
            gaps += 1
            gap_seconds += delta
    covered = max(0.0, frames[-1].timestamp - frames[0].timestamp - gap_seconds)
    return {
        "requested_seconds": round(requested, 3),
        "covered_seconds": round(covered, 3),
        "coverage_ratio": round(min(1.0, covered / requested), 4),
        "gaps": int(gaps),
        "gap_seconds": round(gap_seconds, 3),
        "first_frame": frames[0].timestamp,
        "last_frame": frames[-1].timestamp,
        "partial_pre_roll": frames[0].timestamp > start + gap_threshold,
        "partial_post_roll": frames[-1].timestamp < end - gap_threshold,
    }


def estimate_fps(frames: List[BufferedFrame], default: float) -> float:
    if len(frames) < 2:
        return default
    deltas = np.diff([f.timestamp for f in frames])
    deltas = deltas[deltas > 0]
    if deltas.size == 0:
        return default
    return float(np.clip(1.0 / float(np.median(deltas)), 1.0, 60.0))


# =============================================================================
# Metricas y cola de salida
# =============================================================================

class EvidenceMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.clips_generados = 0
        self.clips_fallidos = 0
        self.eventos_recibidos = 0
        self.eventos_duplicados = 0
        self.eventos_invalidos = 0
        self.eventos_ignorados_no_event = 0
        self.eventos_rechazados_capacidad = 0
        self.bytes_almacenados = 0
        self._processing_total = 0.0
        self._processing_last = 0.0

    def add(self, name: str, amount: Union[int, float] = 1) -> None:
        with self._lock:
            setattr(self, name, getattr(self, name) + amount)

    def processing_time(self, seconds: float) -> None:
        with self._lock:
            self._processing_total += seconds
            self._processing_last = seconds

    def to_dict(self, pending: int) -> Dict[str, Any]:
        with self._lock:
            finished = self.clips_generados + self.clips_fallidos
            return {
                "clips_generados": self.clips_generados,
                "clips_fallidos": self.clips_fallidos,
                "eventos_pendientes": pending,
                "tiempo_procesamiento": {
                    "promedio_s": round(self._processing_total / finished, 3) if finished else 0.0,
                    "ultimo_s": round(self._processing_last, 3),
                },
                "eventos_recibidos": self.eventos_recibidos,
                "eventos_duplicados": self.eventos_duplicados,
                "eventos_invalidos": self.eventos_invalidos,
                "eventos_ignorados_no_event": self.eventos_ignorados_no_event,
                "eventos_rechazados_capacidad": self.eventos_rechazados_capacidad,
                "megabytes_almacenados": round(self.bytes_almacenados / (1024 * 1024), 3),
            }


class BoundedQueue:
    def __init__(self, max_size: int) -> None:
        self._max_size = max_size
        self._buffer: Deque[Any] = deque()
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)

    def put(self, item: Any) -> None:
        with self._not_empty:
            if len(self._buffer) >= self._max_size:
                self._buffer.popleft()
                logger.warning("Cola de salida de evidencias llena: se descarta el resultado mas antiguo")
            self._buffer.append(item)
            self._not_empty.notify()

    def get(self, timeout: Optional[float] = None) -> Optional[Any]:
        with self._not_empty:
            if not self._buffer:
                self._not_empty.wait(timeout=timeout)
            return self._buffer.popleft() if self._buffer else None

    def qsize(self) -> int:
        with self._lock:
            return len(self._buffer)


# =============================================================================
# Servicio principal
# =============================================================================

class EvidenceService:
    """Punto de entrada del Servicio 8.

    - submit_decision(): thread-safe, NO bloqueante. Registra el evento
      (idempotencia por event_id) y agenda su tarea en el event loop.
    - Cada evento: CREATED -> COLLECTING (espera async del post-roll) ->
      PROCESSING (render en thread pool, limitado por semaforo) ->
      READY | FAILED. Eventos de distintas camaras/tracks corren en paralelo.
    - Los EvidenceResult (READY y FAILED) se publican en una cola de salida
      y, opcionalmente, en un callback.
    """

    def __init__(
        self,
        config: Optional[EvidenceConfig] = None,
        buffer: Optional[VideoBufferManager] = None,
        storage: Optional[EvidenceStorageProvider] = None,
        renderer: Optional[ClipRenderer] = None,
        on_result: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.config = config or EvidenceConfig.from_env()
        self.buffer = buffer or InMemoryRingBuffer(
            self.config.buffer_retention_seconds, self.config.buffer_max_bytes_per_camera, self.config.jpeg_quality
        )
        self.storage = storage or LocalFileStorage(self.config.storage_root)
        self.renderer = renderer or AutoClipRenderer()
        self.metrics = EvidenceMetrics()
        self._on_result = on_result

        self._processing_events: Dict[str, EvidenceJob] = {}
        self._completed: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._registry_lock = threading.Lock()
        self._output = BoundedQueue(self.config.output_queue_size)

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._loop_ready = threading.Event()
        self._render_semaphore: Optional[asyncio.Semaphore] = None
        self._accepting = False
        self._start_wall = time.time()

    # -------------------------------------------------------------------
    # Ciclo de vida
    # -------------------------------------------------------------------

    def start(self) -> None:
        if self._loop_thread is not None and self._loop_thread.is_alive():
            return
        self._loop_ready.clear()
        self._loop_thread = threading.Thread(target=self._run_loop, name="EvidenceLoop", daemon=True)
        self._loop_thread.start()
        self._loop_ready.wait(timeout=5.0)
        self._accepting = True
        logger.info("EvidenceService iniciado | storage=%s pre_roll=%.1fs post_roll=%.1fs",
                    self.storage.backend_name, self.config.pre_roll_seconds, self.config.post_roll_seconds)

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._render_semaphore = asyncio.Semaphore(self.config.max_concurrent_renders)
        self._loop_ready.set()
        try:
            loop.run_forever()
        finally:
            loop.close()

    def stop(self, drain_timeout: float = 30.0) -> None:
        """Cierre controlado: deja de aceptar eventos, espera a que terminen
        los pendientes (hasta drain_timeout) y cancela el resto marcandolos
        como FAILED (service_shutdown)."""
        self._accepting = False
        if self._loop is None:
            return
        logger.info("Deteniendo EvidenceService (pendientes=%d)...", len(self._processing_events))
        future = asyncio.run_coroutine_threadsafe(self._drain(drain_timeout), self._loop)
        try:
            future.result(timeout=drain_timeout + 10)
        except Exception:
            logger.exception("Error durante el drenado de tareas")
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=10)
        logger.info("EvidenceService detenido")

    async def _drain(self, timeout: float) -> None:
        with self._registry_lock:
            tasks = [job.task for job in self._processing_events.values() if job.task is not None]
        if not tasks:
            return
        _, pending = await asyncio.wait(tasks, timeout=timeout)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    # -------------------------------------------------------------------
    # Entrada
    # -------------------------------------------------------------------

    def submit_decision(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Registra un DecisionResult y retorna un recibo inmediato. Nunca
        espera el post-roll."""
        self.metrics.add("eventos_recibidos")
        try:
            if not validate_decision_result(payload):
                self.metrics.add("eventos_ignorados_no_event")
                return {"accepted": False, "reason": "no_event_ignored"}
        except InputValidationError as exc:
            self.metrics.add("eventos_invalidos")
            logger.warning("DecisionResult descartado: %s", exc)
            return {"accepted": False, "reason": f"invalid: {exc}"}

        if not self._accepting or self._loop is None:
            return {"accepted": False, "reason": "service_not_running"}

        event_id = str(payload["event_id"])
        evidence_id = evidence_id_for(event_id)

        with self._registry_lock:
            if event_id in self._completed:
                self.metrics.add("eventos_duplicados")
                return {"accepted": True, "duplicate": True, "event_id": event_id, "evidence_id": evidence_id,
                        "status": self._completed[event_id]["status"]}
            job = self._processing_events.get(event_id)
            if job is not None:
                self.metrics.add("eventos_duplicados")
                logger.info("Evento %s duplicado: ya existe una tarea en %s", event_id, job.state.value)
                return {"accepted": True, "duplicate": True, "event_id": event_id, "evidence_id": evidence_id,
                        "status": job.state.value}
            if len(self._processing_events) >= self.config.max_pending_events:
                self.metrics.add("eventos_rechazados_capacidad")
                logger.error("Capacidad maxima de eventos pendientes alcanzada; evento %s rechazado", event_id)
                return {"accepted": False, "reason": "capacity_exceeded", "event_id": event_id}

            start, end = self._clip_window(payload)
            job = EvidenceJob(event_id=event_id, evidence_id=evidence_id, decision=payload,
                              start_time=start, end_time=end)
            job.timeline.append({"state": EvidenceState.CREATED.value, "at": time.time(), "detail": ""})
            self._processing_events[event_id] = job

        logger.info("Evento %s CREATED | tipo=%s prioridad=%s camara=%s ventana=%.1fs",
                    event_id, payload["event_type"], payload["priority"], payload["camera_id"], end - start)
        self._loop.call_soon_threadsafe(self._spawn, job)
        return {"accepted": True, "duplicate": False, "event_id": event_id, "evidence_id": evidence_id,
                "status": EvidenceState.CREATED.value}

    def _spawn(self, job: EvidenceJob) -> None:
        job.task = asyncio.get_running_loop().create_task(self._run_job(job), name=f"evidence-{job.event_id}")

    def _clip_window(self, payload: Dict[str, Any]) -> Tuple[float, float]:
        """Garantiza al menos [t - pre_roll, t + post_roll] y respeta una
        ventana mas amplia del Decision Engine (ej. secuencias), acotada a
        max_clip_seconds conservando el momento del evento y su post-roll."""
        timestamp = float(payload["timestamp"])
        start = timestamp - self.config.pre_roll_seconds
        end = timestamp + self.config.post_roll_seconds
        if payload.get("time_window_start") is not None:
            start = min(start, float(payload["time_window_start"]))
        if payload.get("time_window_end") is not None:
            end = max(end, min(float(payload["time_window_end"]), timestamp + self.config.max_clip_seconds))
        if end - start > self.config.max_clip_seconds:
            start = end - self.config.max_clip_seconds
        return start, end

    # -------------------------------------------------------------------
    # Salida / consulta
    # -------------------------------------------------------------------

    def get_next_result(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        return self._output.get(timeout=timeout)

    def get_status(self, event_id: str) -> Optional[Dict[str, Any]]:
        with self._registry_lock:
            if event_id in self._completed:
                return self._completed[event_id]
            job = self._processing_events.get(event_id)
            if job is not None:
                return {"event_id": event_id, "evidence_id": job.evidence_id, "status": job.state.value}
        return None

    def health_check(self) -> Dict[str, Any]:
        with self._registry_lock:
            pending = len(self._processing_events)
            by_state: Dict[str, int] = {}
            for job in self._processing_events.values():
                by_state[job.state.value] = by_state.get(job.state.value, 0) + 1
        return {
            "service_active": self._loop_thread is not None and self._loop_thread.is_alive() and self._accepting,
            "pending_by_state": by_state,
            "storage": self.storage.health(),
            "buffer": self.buffer.stats(),
            "renderer": type(self.renderer).__name__,
            "output_queue": self._output.qsize(),
            "uptime_seconds": round(time.time() - self._start_wall, 3),
            "metrics": self.metrics.to_dict(pending),
        }

    # -------------------------------------------------------------------
    # Pipeline asincrono por evento
    # -------------------------------------------------------------------

    async def _run_job(self, job: EvidenceJob) -> None:
        started = time.time()
        result: Optional[EvidenceResult] = None
        workdir: Optional[str] = None
        try:
            rehydrated = await asyncio.to_thread(self._rehydrate, job)
            if rehydrated is not None:
                job.transition(EvidenceState.READY, "evidencia existente en almacenamiento (reintento)")
                result = rehydrated
                return

            job.transition(EvidenceState.COLLECTING,
                           f"esperando post-roll hasta +{max(0.0, job.end_time - time.time()):.1f}s")
            timed_out = await self._wait_for_post_roll(job)
            frames = await asyncio.to_thread(
                self.buffer.get_segment, str(job.decision["camera_id"]), job.start_time, job.end_time
            )
            coverage = analyze_coverage(frames, job.start_time, job.end_time, self.config.gap_threshold_seconds)
            coverage["post_roll_wait_timed_out"] = timed_out
            if not frames or coverage["coverage_ratio"] < self.config.min_coverage_ratio:
                result = self._failed(job, "insufficient_video", coverage)
                return

            job.transition(EvidenceState.PROCESSING, f"{len(frames)} frames, cobertura {coverage['coverage_ratio']:.0%}")
            workdir = tempfile.mkdtemp(prefix="evidence_")
            assert self._render_semaphore is not None
            async with self._render_semaphore:
                clip = await asyncio.to_thread(
                    self.renderer.render, frames, os.path.join(workdir, job.evidence_id),
                    estimate_fps(frames, self.config.default_fps),
                )
            stored = await asyncio.to_thread(
                self.storage.save_video, job.evidence_id, str(job.decision["camera_id"]),
                float(job.decision["timestamp"]), clip.path, clip.format,
            )
            metadata = self._build_metadata(job, clip, stored, coverage, frames)
            job.transition(EvidenceState.READY, "clip y metadata almacenados")
            metadata["timeline"] = list(job.timeline)
            metadata["storage"]["metadata_reference"] = await asyncio.to_thread(
                self.storage.save_metadata, job.evidence_id, metadata
            )
            result = self._result(job, EvidenceState.READY, stored.reference, frames[0].timestamp,
                                  frames[-1].timestamp, stored.size_bytes, clip.format, metadata)
            self.metrics.add("bytes_almacenados", stored.size_bytes)
            logger.info("Evidencia %s guardada exitosamente | evento=%s formato=%s tamaño=%.2fMB sha256=%s...",
                        job.evidence_id, job.event_id, clip.format, stored.size_bytes / (1024 * 1024), stored.sha256[:12])
        except asyncio.CancelledError:
            result = self._failed(job, "service_shutdown", {})
            raise
        except Exception as exc:
            logger.exception("Fallo generando evidencia para el evento %s", job.event_id)
            result = self._failed(job, f"{type(exc).__name__}: {exc}", {})
        finally:
            if workdir is not None:
                shutil.rmtree(workdir, ignore_errors=True)
            self._finalize(job, result, time.time() - started)

    async def _wait_for_post_roll(self, job: EvidenceJob) -> bool:
        """Espera NO bloqueante hasta que el buffer contenga video posterior
        a end_time. Retorna True si se agoto la espera (video atrasado o
        camara caida), en cuyo caso se continua con lo disponible."""
        camera_id = str(job.decision["camera_id"])
        deadline = time.time() + max(0.0, job.end_time - time.time()) + self.config.collect_grace_seconds
        while True:
            latest = self.buffer.latest_timestamp(camera_id)
            if latest is not None and latest >= job.end_time:
                return False
            if time.time() >= deadline:
                logger.warning("Evento %s: post-roll incompleto (ultimo frame=%s)", job.event_id, latest)
                return True
            await asyncio.sleep(self.config.poll_interval_seconds)

    def _rehydrate(self, job: EvidenceJob) -> Optional[EvidenceResult]:
        """Idempotencia entre reinicios: si la evidencia ya fue almacenada
        para este event_id, se reconstruye el resultado sin re-renderizar."""
        metadata = self.storage.load_metadata(job.evidence_id)
        if not metadata or metadata.get("status") != EvidenceState.READY.value:
            return None
        metadata["rehydrated"] = True
        return self._result(job, EvidenceState.READY, metadata.get("video_reference"),
                            float(metadata["clip"]["start_time"]), float(metadata["clip"]["end_time"]),
                            int(metadata["clip"]["size"]), metadata["clip"]["format"], metadata)

    # -------------------------------------------------------------------
    # Construccion de resultados
    # -------------------------------------------------------------------

    def _compact(self, value: Any) -> Any:
        encoded = json.dumps(value, default=str)
        if len(encoded) <= self.config.max_explanation_bytes:
            return value
        return {"truncated": True, "size_bytes": len(encoded)}

    def _build_metadata(self, job: EvidenceJob, clip: RenderedClip, stored: StoredObject,
                        coverage: Dict[str, Any], frames: List[BufferedFrame]) -> Dict[str, Any]:
        decision = job.decision
        explanation = decision.get("explanation") or {}
        rule = explanation.get("rule") or {}
        return {
            "status": EvidenceState.READY.value,
            "evidence_id": job.evidence_id,
            "event_id": job.event_id,
            "video_reference": stored.reference,
            "rules_activated": [{
                "rule_id": decision.get("rule_id") or rule.get("rule_id"),
                "rule_name": rule.get("name"),
                "rules_version": rule.get("rules_version"),
                "summary": explanation.get("summary"),
            }],
            "checksum": {"algorithm": "sha256", "value": stored.sha256},
            "decision": {
                "event_type": decision["event_type"],
                "priority": decision["priority"],
                "timestamp": decision["timestamp"],
                "requested_window": [decision.get("time_window_start"), decision.get("time_window_end")],
                "source_result_keys": decision.get("source_result_keys", []),
                "matched_conditions": self._compact(explanation.get("matched_conditions")),
            },
            "context": self._compact(explanation.get("context")),
            "clip": {
                "format": clip.format,
                "mime_type": clip.mime_type,
                "size": stored.size_bytes,
                "frame_count": clip.frame_count,
                "fps": round(clip.fps, 2),
                "width": clip.width,
                "height": clip.height,
                "corrupt_frames_skipped": clip.corrupt_frames_skipped,
                "start_time": frames[0].timestamp,
                "end_time": frames[-1].timestamp,
                "requested_start": job.start_time,
                "requested_end": job.end_time,
            },
            "coverage": coverage,
            "storage": {"backend": self.storage.backend_name},
            "created_at": time.time(),
        }

    def _result(self, job: EvidenceJob, status: EvidenceState, reference: Optional[str], start: float,
                end: float, size: int, fmt: Optional[str], metadata: Dict[str, Any]) -> EvidenceResult:
        decision = job.decision
        return EvidenceResult(
            evidence_id=job.evidence_id,
            event_id=job.event_id,
            camera_id=str(decision["camera_id"]),
            session_id=str(decision["session_id"]),
            track_id=str(decision["track_id"]),
            event_type=str(decision["event_type"]),
            priority=str(decision["priority"]),
            video_reference=reference,
            start_time=start,
            end_time=end,
            duration=max(0.0, end - start),
            size=size,
            format=fmt,
            status=status,
            metadata=metadata,
        )

    def _failed(self, job: EvidenceJob, reason: str, coverage: Dict[str, Any]) -> EvidenceResult:
        if job.state not in (EvidenceState.READY, EvidenceState.FAILED):
            job.transition(EvidenceState.FAILED, reason)
        explanation = job.decision.get("explanation") or {}
        metadata = {
            "failure_reason": reason,
            "rules_activated": [{"rule_id": job.decision.get("rule_id"), "summary": explanation.get("summary")}],
            "coverage": coverage,
            "timeline": list(job.timeline),
        }
        # Aunque falle el video, el Alert Service recibe el evento: la
        # decision de negocio sigue siendo valida sin clip.
        return self._result(job, EvidenceState.FAILED, None, job.start_time, job.end_time, 0, None, metadata)

    def _finalize(self, job: EvidenceJob, result: Optional[EvidenceResult], elapsed: float) -> None:
        if result is None:
            result = self._failed(job, "unknown_error", {})
        payload = result.to_dict()
        self.metrics.processing_time(elapsed)
        self.metrics.add("clips_generados" if result.status == EvidenceState.READY else "clips_fallidos")
        with self._registry_lock:
            self._processing_events.pop(job.event_id, None)
            self._completed[job.event_id] = {"event_id": job.event_id, "evidence_id": job.evidence_id,
                                              "status": result.status.value,
                                              "video_reference": result.video_reference}
            while len(self._completed) > self.config.completed_cache_size:
                self._completed.popitem(last=False)
        self._output.put(payload)
        if self._on_result is not None:
            try:
                self._on_result(payload)
            except Exception:
                logger.exception("El callback on_result fallo (se ignora)")


# =============================================================================
# Smoke test manual (ingesta simulada en tiempo real, ventanas cortas)
# =============================================================================

if __name__ == "__main__":
    import cv2

    root = tempfile.mkdtemp(prefix="vigilia_evidence_demo_")
    config = EvidenceConfig(storage_root=root, pre_roll_seconds=3.0, post_roll_seconds=2.0,
                            collect_grace_seconds=1.5, buffer_retention_seconds=10.0)
    service = EvidenceService(config=config)
    service.start()

    stop_feed = threading.Event()

    def feed(camera_id: str, fps: float) -> None:
        index = 0
        while not stop_feed.is_set():
            frame = np.full((240, 320, 3), 40, dtype=np.uint8)
            cv2.putText(frame, f"{camera_id} #{index}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            service.buffer.append_frame(camera_id, time.time(), frame)
            index += 1
            time.sleep(1.0 / fps)

    feeders = [threading.Thread(target=feed, args=(cam, 10.0), daemon=True) for cam in ("CAM-001", "CAM-002")]
    for thread in feeders:
        thread.start()
    time.sleep(4.0)  # llenar el pre-roll

    def decision(camera: str, track: str, event_type: str, priority: str) -> Dict[str, Any]:
        now = time.time()
        return {
            "event_id": str(uuid.uuid4()), "camera_id": camera, "session_id": "s-demo", "track_id": track,
            "subject_id": f"{camera}:{track}", "decision": "EVENT_GENERATED", "event_type": event_type,
            "priority": priority, "timestamp": now, "time_window_start": now - 3.0, "time_window_end": now + 2.0,
            "rule_id": "POSSIBLE_THEFT_SEQUENCE", "source_result_keys": ["rid:x"],
            "explanation": {"summary": "Regla de prueba cumplida", "rule": {"rule_id": "POSSIBLE_THEFT_SEQUENCE",
                            "name": "Robo (demo)", "rules_version": "demo"}, "context": {"zone_type": "exit"}},
        }

    theft = decision("CAM-001", "CAM-001-3", "POSSIBLE_THEFT", "HIGH")
    t0 = time.time()
    print("submit ->", service.submit_decision(theft))
    print(f"submit retorno en {(time.time() - t0) * 1000:.1f} ms (no espera el post-roll)")
    print("duplicado ->", service.submit_decision(theft))
    print("paralelo ->", service.submit_decision(decision("CAM-002", "CAM-002-1", "SLEEPING_GUARD", "CRITICAL")))
    print("sin video ->", service.submit_decision(decision("CAM-999", "CAM-999-1", "SUSPICIOUS_BEHAVIOR", "MEDIUM")))
    print("no_event ->", service.submit_decision({"decision": "NO_EVENT", "event_id": str(uuid.uuid4())}))
    print("malformado ->", service.submit_decision({**theft, "event_id": "no-es-uuid"}))

    results = []
    deadline = time.time() + 15
    while len(results) < 3 and time.time() < deadline:
        item = service.get_next_result(timeout=0.5)
        if item is not None:
            results.append(item)
    stop_feed.set()

    storage = service.storage
    assert isinstance(storage, LocalFileStorage)
    for r in results:
        print(f"\n{r['event_type']} [{r['status']}] evidence_id={r['evidence_id']}")
        if r["status"] == "READY":
            path = storage.resolve_path(r["video_reference"])
            print(f"  video={r['video_reference']} formato={r['format']} tamaño={r['size']}B duracion={r['duration']:.2f}s")
            print(f"  frames={r['metadata']['clip']['frame_count']} fps={r['metadata']['clip']['fps']} "
                  f"cobertura={r['metadata']['coverage']['coverage_ratio']}")
            print(f"  checksum verificado: {_sha256_file(path) == r['metadata']['checksum']['value']}")
            print(f"  timeline: {[s['state'] for s in r['metadata']['timeline']]}")
        else:
            print(f"  motivo: {r['metadata']['failure_reason']}")

    print("\nreintento tras completar ->", service.submit_decision(theft))
    health = service.health_check()
    print("HEALTH:", json.dumps({k: health[k] for k in ("pending_by_state", "storage", "metrics")}, indent=1))
    service.stop()

    def _force_remove(func, path, _exc):
        os.chmod(path, stat.S_IWRITE)
        func(path)

    shutil.rmtree(root, onerror=_force_remove)
