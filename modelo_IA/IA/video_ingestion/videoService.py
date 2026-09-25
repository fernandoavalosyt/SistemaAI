"""
videoService.py

Servicio de Ingesta de Video (Video Ingestion Service).

Responsabilidad UNICA: capturar frames desde una fuente de video, normalizarlos,
validarlos y exponerlos a traves de una cola/buffer en tiempo real para que un
componente consumidor (pipeline de IA) los consuma.

Este modulo NO realiza inferencia, deteccion, tracking ni ningun tipo de logica
de negocio de IA. Es puramente una capa de ingestion/transporte de video.
"""

from __future__ import annotations

import abc
import logging
import os
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, Generator, List, Optional
from urllib.parse import quote, urlsplit, urlunsplit

import cv2
import numpy as np


# =============================================================================
# Logging
# =============================================================================
# Logging estructurado. Nunca se registran credenciales ni datos sensibles.
# Los eventos de alta frecuencia (ej. frame descartado) se loguean con
# rate-limiting para no inundar el log.

logger = logging.getLogger("video_service")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


# =============================================================================
# Enums / Estados
# =============================================================================

class ConnectionState(str, Enum):
    """Estados posibles de conexion de una fuente de video."""

    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    RECONNECTING = "RECONNECTING"
    ERROR = "ERROR"
    STOPPED = "STOPPED"


class SourceType(str, Enum):
    """Tipos de fuente de video soportados (activos o preparados a futuro)."""

    LOCAL_CAMERA = "LOCAL_CAMERA"
    RTSP_CAMERA = "RTSP_CAMERA"   # Activo: camaras IP (RTSP / MJPEG sobre HTTP).
    NVR = "NVR"                   # Reservado. No activo aun.
    DVR = "DVR"                   # Reservado. No activo aun.


# =============================================================================
# Estructuras de datos
# =============================================================================

@dataclass
class SourceMetadata:
    """Metadata de identificacion de una fuente/camara.

    Diseñada para escalar a multiples camaras simultaneas sin colisiones:
    camera_id es explicito y unico por fuente, session_id se genera por
    cada ciclo de conexion y stream_id identifica el flujo logico de datos.
    """

    camera_id: str
    source_type: SourceType
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    stream_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    connection_state: ConnectionState = ConnectionState.DISCONNECTED


@dataclass
class Frame:
    """Objeto normalizado de salida del servicio (Frame + Metadata).

    Este es el UNICO contrato de salida expuesto al consumidor (pipeline de IA).
    """

    camera_id: str
    frame_id: int
    session_id: str
    stream_id: str
    timestamp: float
    image: np.ndarray
    width: int
    height: int
    channels: int


# =============================================================================
# Excepciones propias
# =============================================================================

class VideoSourceError(Exception):
    """Error generico de una fuente de video."""


class FrameValidationError(Exception):
    """Error de validacion de un frame capturado."""


# =============================================================================
# Metricas / Health check
# =============================================================================

@dataclass
class ServiceMetrics:
    """Contadores y variables para observabilidad del servicio."""

    frames_received: int = 0
    frames_delivered: int = 0
    frames_dropped_total: int = 0
    frames_dropped_invalid: int = 0
    frames_dropped_fps_control: int = 0
    frames_dropped_backpressure: int = 0
    reconnect_attempts: int = 0
    reconnect_successes: int = 0
    last_frame_timestamp: Optional[float] = None
    service_start_timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, object]:
        return {
            "frames_received": self.frames_received,
            "frames_delivered": self.frames_delivered,
            "frames_dropped_total": self.frames_dropped_total,
            "frames_dropped_invalid": self.frames_dropped_invalid,
            "frames_dropped_fps_control": self.frames_dropped_fps_control,
            "frames_dropped_backpressure": self.frames_dropped_backpressure,
            "reconnect_attempts": self.reconnect_attempts,
            "reconnect_successes": self.reconnect_successes,
            "last_frame_timestamp": self.last_frame_timestamp,
            "seconds_since_last_frame": (
                None
                if self.last_frame_timestamp is None
                else round(time.time() - self.last_frame_timestamp, 3)
            ),
            "service_uptime_seconds": round(time.time() - self.service_start_timestamp, 3),
        }


# =============================================================================
# Log rate limiter (para eventos de alta frecuencia, ej. frames descartados)
# =============================================================================

class _RateLimitedLogger:
    """Evita saturar el log con eventos repetitivos (ej. cada frame descartado).

    Acumula el conteo de eventos y solo emite una linea de log cada
    `interval_seconds`, con el total acumulado en ese periodo.
    """

    def __init__(self, logger_: logging.Logger, interval_seconds: float = 5.0) -> None:
        self._logger = logger_
        self._interval = interval_seconds
        self._counts: Dict[str, int] = {}
        self._last_emit: Dict[str, float] = {}
        self._lock = threading.Lock()

    def log(self, key: str, level: int, message_template: str) -> None:
        now = time.time()
        with self._lock:
            self._counts[key] = self._counts.get(key, 0) + 1
            last = self._last_emit.get(key, 0.0)
            if now - last >= self._interval:
                count = self._counts[key]
                self._logger.log(level, message_template, count)
                self._counts[key] = 0
                self._last_emit[key] = now


_rate_limited_logger = _RateLimitedLogger(logger, interval_seconds=5.0)


# =============================================================================
# Frame ring buffer con politica "descartar los mas antiguos"
# =============================================================================

class FrameBuffer:
    """Cola/Buffer de frames con tamaño maximo estricto.

    Politica de backpressure: si el buffer esta lleno, se descarta el frame
    mas antiguo para dar espacio al mas reciente. Se privilegia siempre el
    tiempo real; el buffer NUNCA crece de forma indefinida.
    """

    def __init__(self, max_size: int, metrics: ServiceMetrics) -> None:
        if max_size <= 0:
            raise ValueError("max_size debe ser mayor a 0")
        self._max_size = max_size
        self._buffer: Deque[Frame] = deque(maxlen=None)  # control manual del limite
        self._metrics = metrics
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)

    def put(self, frame: Frame) -> None:
        with self._not_empty:
            if len(self._buffer) >= self._max_size:
                dropped = self._buffer.popleft()
                self._metrics.frames_dropped_total += 1
                self._metrics.frames_dropped_backpressure += 1
                _rate_limited_logger.log(
                    "buffer_overflow",
                    logging.WARNING,
                    "Buffer lleno: %s frame(s) descartados por backpressure (mas antiguos primero)",
                )
                del dropped
            self._buffer.append(frame)
            self._not_empty.notify()

    def get_next(self, timeout: Optional[float] = None) -> Optional[Frame]:
        """Bloquea hasta que haya un frame disponible o expire el timeout."""
        with self._not_empty:
            if not self._buffer:
                self._not_empty.wait(timeout=timeout)
            if not self._buffer:
                return None
            return self._buffer.popleft()

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()

    def qsize(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def max_size(self) -> int:
        return self._max_size


# =============================================================================
# Abstraccion de fuente de video
# =============================================================================

class VideoSource(abc.ABC):
    """Interfaz abstracta de la que deben heredar todas las fuentes de video.

    El pipeline principal (CameraWorker) trabaja UNICAMENTE contra esta
    abstraccion, nunca contra una implementacion concreta. Esto permite
    agregar nuevas fuentes (RTSP, NVR, DVR) sin modificar el pipeline.
    """

    def __init__(self, camera_id: str, source_type: SourceType) -> None:
        self.metadata = SourceMetadata(camera_id=camera_id, source_type=source_type)

    @abc.abstractmethod
    def connect(self) -> None:
        """Establece la conexion/apertura del dispositivo de video."""
        raise NotImplementedError

    @abc.abstractmethod
    def read(self) -> Optional[np.ndarray]:
        """Lee un frame crudo (numpy array) de la fuente. None si no hay frame valido."""
        raise NotImplementedError

    @abc.abstractmethod
    def release(self) -> None:
        """Libera recursos de hardware/software asociados a la fuente."""
        raise NotImplementedError

    @abc.abstractmethod
    def is_opened(self) -> bool:
        """Indica si la fuente esta actualmente abierta/disponible para lectura."""
        raise NotImplementedError


class LocalCameraSource(VideoSource):
    """Fuente de video activa: camara integrada/local de la laptop via OpenCV."""

    def __init__(
        self,
        camera_id: str,
        device_index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ) -> None:
        super().__init__(camera_id=camera_id, source_type=SourceType.LOCAL_CAMERA)
        self._device_index = device_index
        self._width = width
        self._height = height
        self._fps = fps
        self._capture: Optional[cv2.VideoCapture] = None

    def connect(self) -> None:
        capture = cv2.VideoCapture(self._device_index)
        if not capture.isOpened():
            capture.release()
            raise VideoSourceError(
                f"No fue posible abrir la camara local (device_index={self._device_index})"
            )

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        capture.set(cv2.CAP_PROP_FPS, self._fps)

        self._capture = capture

    def read(self) -> Optional[np.ndarray]:
        if self._capture is None:
            return None
        ok, frame = self._capture.read()
        if not ok or frame is None:
            return None
        return frame

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def is_opened(self) -> bool:
        return self._capture is not None and self._capture.isOpened()


def mask_stream_url(url: str) -> str:
    """Oculta usuario/contraseña de una URL para poder loguearla."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return "<url invalida>"
    if not parts.netloc or "@" not in parts.netloc:
        return url
    host = parts.netloc.rsplit("@", 1)[1]
    return urlunsplit((parts.scheme, f"***:***@{host}", parts.path, parts.query, parts.fragment))


class RTSPCameraSource(VideoSource):
    """Fuente de video ACTIVA para camaras IP (RTSP) via OpenCV/FFmpeg.

    La URL se lee de variables de entorno en cada conexion (permite rotar
    credenciales sin reiniciar):
        RTSP_URL       ej. rtsp://192.168.1.64:554/Streaming/Channels/101
        RTSP_USER      opcional; se inyecta en la URL codificado
        RTSP_PASSWORD  opcional; se inyecta en la URL codificado
        RTSP_TRANSPORT tcp (default, evita perdida de paquetes) | udp

    Tambien acepta http(s):// (camaras IP que exponen MJPEG) y, solo para
    pruebas locales, la ruta de un archivo de video.

    Las credenciales nunca se loguean ni aparecen en excepciones. Se fuerza
    el backend CAP_FFMPEG: el fallback a CAP_IMAGES de OpenCV imprime la URL
    completa (con contraseña) cuando la apertura falla.
    """

    _ALLOWED_SCHEMES = ("rtsp", "rtsps", "http", "https")

    def __init__(
        self,
        camera_id: str,
        url: Optional[str] = None,
        env_url_var: str = "RTSP_URL",
        open_timeout_ms: int = 10000,
        read_timeout_ms: int = 10000,
    ) -> None:
        super().__init__(camera_id=camera_id, source_type=SourceType.RTSP_CAMERA)
        self._explicit_url = url
        self._env_url_var = env_url_var
        self._open_timeout_ms = open_timeout_ms
        self._read_timeout_ms = read_timeout_ms
        self._capture: Optional[cv2.VideoCapture] = None

    def _resolve_url(self) -> str:
        url = (self._explicit_url or os.getenv(self._env_url_var, "")).strip()
        if not url:
            raise VideoSourceError(f"Variable de entorno {self._env_url_var} no configurada")

        scheme = urlsplit(url).scheme.lower()
        if scheme not in self._ALLOWED_SCHEMES:
            if os.path.isfile(url):
                return url  # archivo local: solo para pruebas
            raise VideoSourceError(f"Esquema de URL no soportado: {scheme or '<vacio>'}")

        user = os.getenv("RTSP_USER", "")
        password = os.getenv("RTSP_PASSWORD", "")
        parts = urlsplit(url)
        if user and "@" not in parts.netloc:
            credentials = quote(user, safe="") + (":" + quote(password, safe="") if password else "")
            url = urlunsplit((parts.scheme, f"{credentials}@{parts.netloc}", parts.path, parts.query, parts.fragment))
        return url

    def connect(self) -> None:
        url = self._resolve_url()
        masked = mask_stream_url(url)

        if urlsplit(url).scheme.lower() in ("rtsp", "rtsps"):
            transport = os.getenv("RTSP_TRANSPORT", "tcp").lower()
            # Debe definirse ANTES de abrir el stream; FFmpeg lo lee al abrir.
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{transport}"

        params: List[int] = []
        if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            params += [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self._open_timeout_ms]
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            params += [cv2.CAP_PROP_READ_TIMEOUT_MSEC, self._read_timeout_ms]

        try:
            capture = cv2.VideoCapture(url, cv2.CAP_FFMPEG, params)
        except cv2.error:
            raise VideoSourceError(f"OpenCV no pudo inicializar el stream ({masked})") from None
        if not capture.isOpened():
            capture.release()
            raise VideoSourceError(f"No fue posible abrir el stream de la camara IP ({masked})")

        # Buffer minimo: se prioriza el frame mas reciente (tiempo real).
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._capture = capture
        logger.info(
            "Stream de camara IP abierto | camera_id=%s url=%s resolucion=%dx%d fps_reportados=%.1f",
            self.metadata.camera_id,
            masked,
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            capture.get(cv2.CAP_PROP_FPS) or 0.0,
        )

    def read(self) -> Optional[np.ndarray]:
        if self._capture is None:
            return None
        ok, frame = self._capture.read()
        if not ok or frame is None:
            return None
        return frame

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def is_opened(self) -> bool:
        return self._capture is not None and self._capture.isOpened()


# -----------------------------------------------------------------------------
# NOTA ARQUITECTONICA:
# Las siguientes clases son STUBS preparados para futuras fuentes de video.
# NO estan activas ni se instancian en el flujo actual del servicio.
# Cuando se habiliten, deberan:
#   - Leer credenciales/URLs desde variables de entorno o un gestor de
#     secretos (nunca hardcodear usuario/contraseña).
#   - Implementar connect()/read()/release()/is_opened() de forma analoga
#     a RTSPCameraSource, respetando el contrato de VideoSource.
#   - No requerir cambios en CameraWorker ni en el pipeline principal.
# -----------------------------------------------------------------------------


class NVRSource(VideoSource):
    """[STUB] Fuente de video para integracion con NVR (Network Video Recorder).

    Pendiente de implementacion. Credenciales/endpoints via variables de
    entorno o gestor de secretos.
    """

    def __init__(self, camera_id: str) -> None:
        super().__init__(camera_id=camera_id, source_type=SourceType.NVR)

    def connect(self) -> None:
        raise NotImplementedError("NVRSource aun no esta implementada.")

    def read(self) -> Optional[np.ndarray]:
        raise NotImplementedError("NVRSource aun no esta implementada.")

    def release(self) -> None:
        raise NotImplementedError("NVRSource aun no esta implementada.")

    def is_opened(self) -> bool:
        return False


class DVRSource(VideoSource):
    """[STUB] Fuente de video para integracion con DVR (Digital Video Recorder).

    Pendiente de implementacion. Credenciales/endpoints via variables de
    entorno o gestor de secretos.
    """

    def __init__(self, camera_id: str) -> None:
        super().__init__(camera_id=camera_id, source_type=SourceType.DVR)

    def connect(self) -> None:
        raise NotImplementedError("DVRSource aun no esta implementada.")

    def read(self) -> Optional[np.ndarray]:
        raise NotImplementedError("DVRSource aun no esta implementada.")

    def release(self) -> None:
        raise NotImplementedError("DVRSource aun no esta implementada.")

    def is_opened(self) -> bool:
        return False


# =============================================================================
# Validacion de frames
# =============================================================================

def validate_raw_frame(
    raw_frame: Optional[np.ndarray],
    expected_channels: int = 3,
) -> None:
    """Valida un frame crudo antes de normalizarlo y encolarlo.

    Lanza FrameValidationError si el frame no cumple los requisitos minimos.
    Nunca propaga un frame invalido al pipeline.
    """

    if raw_frame is None:
        raise FrameValidationError("Frame nulo/no recibido")

    if not isinstance(raw_frame, np.ndarray):
        raise FrameValidationError("Frame no es un array valido")

    if raw_frame.size == 0:
        raise FrameValidationError("Frame vacio (size=0)")

    if raw_frame.ndim != 3:
        raise FrameValidationError(f"Dimensiones invalidas: ndim={raw_frame.ndim}")

    height, width, channels = raw_frame.shape
    if height <= 0 or width <= 0:
        raise FrameValidationError(f"Dimensiones invalidas: {width}x{height}")

    if channels != expected_channels:
        raise FrameValidationError(
            f"Formato de canales inesperado: {channels} (se esperaba {expected_channels})"
        )


def validate_frame_identity(camera_id: str, session_id: str, timestamp: float) -> None:
    """Valida que los identificadores y el timestamp del frame sean validos."""

    if not camera_id:
        raise FrameValidationError("camera_id invalido/vacio")
    if not session_id:
        raise FrameValidationError("session_id invalido/vacio")
    if timestamp is None or timestamp <= 0:
        raise FrameValidationError("timestamp invalido")


# =============================================================================
# Control de FPS (muestreo/descarte matematico)
# =============================================================================

class FpsThrottler:
    """Descarta frames de forma controlada para pasar de FPS de entrada a
    un FPS objetivo menor, mediante muestreo matematico (nearest-interval).

    Tambien mide el FPS real de entrada (frames que llegan desde la fuente,
    antes del descarte).
    """

    def __init__(self, target_fps: float) -> None:
        if target_fps <= 0:
            raise ValueError("target_fps debe ser mayor a 0")
        self._target_interval = 1.0 / target_fps
        self._last_accepted_ts: Optional[float] = None

        self._input_frame_count = 0
        self._input_fps_window_start = time.time()
        self._measured_input_fps = 0.0

    def register_input_frame(self) -> None:
        self._input_frame_count += 1
        elapsed = time.time() - self._input_fps_window_start
        if elapsed >= 1.0:
            self._measured_input_fps = self._input_frame_count / elapsed
            self._input_frame_count = 0
            self._input_fps_window_start = time.time()

    def should_accept(self, timestamp: float) -> bool:
        """Determina si este frame debe conservarse segun el FPS objetivo."""
        if self._last_accepted_ts is None:
            self._last_accepted_ts = timestamp
            return True

        if timestamp - self._last_accepted_ts >= self._target_interval:
            self._last_accepted_ts = timestamp
            return True

        return False

    @property
    def measured_input_fps(self) -> float:
        return round(self._measured_input_fps, 2)


# =============================================================================
# Configuracion del worker de camara
# =============================================================================

@dataclass
class CameraWorkerConfig:
    """Configuracion de captura y control de flujo del worker."""

    capture_width: int = 1280
    capture_height: int = 720
    capture_fps: int = 30
    target_fps: float = 10.0
    buffer_max_size: int = 30
    max_reconnect_attempts: int = 10
    reconnect_base_delay_seconds: float = 1.0
    reconnect_max_delay_seconds: float = 30.0
    read_loop_error_tolerance: int = 5


# =============================================================================
# Worker principal: gestiona una fuente de video (hilo dedicado)
# =============================================================================

class CameraWorker:
    """Orquesta el ciclo de vida de una VideoSource: conexion, lectura,
    validacion, control de FPS, encolado en buffer, reconexion y cierre.

    Trabaja unicamente contra la abstraccion VideoSource (inyectada por
    constructor), nunca contra una implementacion concreta.
    """

    def __init__(
        self,
        source: VideoSource,
        config: Optional[CameraWorkerConfig] = None,
    ) -> None:
        self._source = source
        self._config = config or CameraWorkerConfig()

        self.metrics = ServiceMetrics()
        self._buffer = FrameBuffer(max_size=self._config.buffer_max_size, metrics=self.metrics)
        self._fps_throttler = FpsThrottler(target_fps=self._config.target_fps)

        self._frame_counter = 0
        self._reconnect_attempts_current_cycle = 0

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._state_lock = threading.Lock()

    # -------------------------------------------------------------------
    # Ciclo de vida del servicio
    # -------------------------------------------------------------------

    def start(self) -> None:
        """Inicia el worker en un hilo dedicado (no bloquea el hilo principal)."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("El servicio ya esta en ejecucion, se ignora start() duplicado")
            return

        logger.info(
            "Iniciando servicio de video | camera_id=%s source_type=%s",
            self._source.metadata.camera_id,
            self._source.metadata.source_type.value,
        )
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"CameraWorker-{self._source.metadata.camera_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Cierre controlado (graceful shutdown): detiene el hilo, libera la
        camara, vacia el buffer y libera todos los recursos asociados."""
        logger.info("Deteniendo servicio de video | camera_id=%s", self._source.metadata.camera_id)
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning(
                    "El hilo del worker no finalizo dentro del timeout de %.1fs", timeout
                )

        self._set_state(ConnectionState.STOPPED)
        self._safe_release_source()
        self._buffer.clear()
        logger.info("Servicio de video detenido y recursos liberados | camera_id=%s",
                    self._source.metadata.camera_id)

    # -------------------------------------------------------------------
    # Salida normalizada (output contract)
    # -------------------------------------------------------------------

    def get_next_frame(self, timeout: float = 1.0) -> Optional[Dict[str, object]]:
        """Entrega el siguiente frame normalizado disponible en el buffer,
        o None si no hay frames dentro del timeout indicado."""
        frame = self._buffer.get_next(timeout=timeout)
        if frame is None:
            return None
        self.metrics.frames_delivered += 1
        return self._frame_to_dict(frame)

    def frame_stream(self) -> Generator[Dict[str, object], None, None]:
        """Generador continuo de frames normalizados. Se detiene cuando el
        servicio recibe stop()."""
        while not self._stop_event.is_set():
            frame_dict = self.get_next_frame(timeout=1.0)
            if frame_dict is not None:
                yield frame_dict

    @staticmethod
    def _frame_to_dict(frame: Frame) -> Dict[str, object]:
        return {
            "camera_id": frame.camera_id,
            "frame_id": frame.frame_id,
            "session_id": frame.session_id,
            "stream_id": frame.stream_id,
            "timestamp": frame.timestamp,
            "image": frame.image,
            "width": frame.width,
            "height": frame.height,
            "channels": frame.channels,
        }

    # -------------------------------------------------------------------
    # Health check / metricas
    # -------------------------------------------------------------------

    def health_check(self) -> Dict[str, object]:
        """Retorna el estado de salud consultable del servicio."""
        return {
            "service_active": self._thread is not None and self._thread.is_alive(),
            "camera_id": self._source.metadata.camera_id,
            "source_type": self._source.metadata.source_type.value,
            "connection_state": self._source.metadata.connection_state.value,
            "session_id": self._source.metadata.session_id,
            "stream_id": self._source.metadata.stream_id,
            "measured_input_fps": self._fps_throttler.measured_input_fps,
            "target_fps": self._config.target_fps,
            "buffer_usage": f"{self._buffer.qsize()}/{self._buffer.max_size}",
            "metrics": self.metrics.to_dict(),
        }

    # -------------------------------------------------------------------
    # Estado interno
    # -------------------------------------------------------------------

    def _set_state(self, new_state: ConnectionState) -> None:
        with self._state_lock:
            old_state = self._source.metadata.connection_state
            self._source.metadata.connection_state = new_state
        if old_state != new_state:
            logger.info(
                "Cambio de estado | camera_id=%s %s -> %s",
                self._source.metadata.camera_id,
                old_state.value,
                new_state.value,
            )

    # -------------------------------------------------------------------
    # Loop principal (ejecutado en hilo dedicado)
    # -------------------------------------------------------------------

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                connected = self._connect_with_retry()
                if not connected:
                    logger.critical(
                        "No fue posible conectar la fuente tras agotar los intentos de "
                        "reconexion | camera_id=%s",
                        self._source.metadata.camera_id,
                    )
                    self._set_state(ConnectionState.ERROR)
                    return

                self._reconnect_attempts_current_cycle = 0
                self._read_loop()

                if self._stop_event.is_set():
                    break

                # Si _read_loop retorna sin haberse pedido stop, se perdio
                # la conexion: liberar recursos antes de reintentar.
                self._safe_release_source()
        except Exception:
            logger.exception(
                "Error inesperado en el worker de video | camera_id=%s. "
                "El fallo se contiene y no crashea el servicio.",
                self._source.metadata.camera_id,
            )
            self._set_state(ConnectionState.ERROR)

    def _connect_with_retry(self) -> bool:
        """Intenta conectar la fuente con backoff exponencial. Retorna True
        si logro conectar, False si agoto los intentos maximos."""
        attempt = 0
        while not self._stop_event.is_set():
            attempt += 1
            self._reconnect_attempts_current_cycle = attempt

            state = ConnectionState.CONNECTING if attempt == 1 else ConnectionState.RECONNECTING
            self._set_state(state)

            if attempt > 1:
                self.metrics.reconnect_attempts += 1
                logger.warning(
                    "Intento de reconexion %d/%d | camera_id=%s",
                    attempt,
                    self._config.max_reconnect_attempts,
                    self._source.metadata.camera_id,
                )

            try:
                self._source.connect()
                self._set_state(ConnectionState.CONNECTED)
                if attempt > 1:
                    self.metrics.reconnect_successes += 1
                    logger.info(
                        "Reconexion exitosa | camera_id=%s intento=%d",
                        self._source.metadata.camera_id,
                        attempt,
                    )
                else:
                    logger.info(
                        "Conexion establecida | camera_id=%s", self._source.metadata.camera_id
                    )
                # Nueva sesion logica tras cada (re)conexion exitosa. El
                # frame_id NO se reinicia: Tracking, Pose y Behavior descartan
                # por camara todo frame_id <= al ultimo procesado, asi que un
                # reinicio dejaria a la camara ciega aguas abajo.
                self._source.metadata.session_id = str(uuid.uuid4())
                return True

            except VideoSourceError as exc:
                logger.error(
                    "Fallo al conectar la fuente | camera_id=%s error=%s",
                    self._source.metadata.camera_id,
                    str(exc),
                )
                self._safe_release_source()

            if attempt >= self._config.max_reconnect_attempts:
                return False

            delay = min(
                self._config.reconnect_base_delay_seconds * (2 ** (attempt - 1)),
                self._config.reconnect_max_delay_seconds,
            )
            self._stop_event.wait(timeout=delay)

        return False

    def _read_loop(self) -> None:
        """Lee frames de la fuente mientras la conexion se mantenga sana."""
        consecutive_errors = 0

        while not self._stop_event.is_set():
            if not self._source.is_opened():
                logger.warning(
                    "Dispositivo desconectado detectado | camera_id=%s",
                    self._source.metadata.camera_id,
                )
                return

            try:
                raw_frame = self._source.read()
            except Exception as exc:  # Errores de hardware/driver no deben crashear el hilo
                consecutive_errors += 1
                logger.error(
                    "Error de lectura en la fuente | camera_id=%s error=%s (consecutivos=%d)",
                    self._source.metadata.camera_id,
                    str(exc),
                    consecutive_errors,
                )
                if consecutive_errors >= self._config.read_loop_error_tolerance:
                    logger.error(
                        "Tolerancia de errores de lectura agotada, se fuerza reconexion | "
                        "camera_id=%s",
                        self._source.metadata.camera_id,
                    )
                    return
                continue

            if raw_frame is None:
                consecutive_errors += 1
                _rate_limited_logger.log(
                    "invalid_read",
                    logging.WARNING,
                    "Lecturas invalidas/vacias detectadas (%d en la ultima ventana)",
                )
                if consecutive_errors >= self._config.read_loop_error_tolerance:
                    logger.warning(
                        "Flujo detenido / stream invalido de forma sostenida, se fuerza "
                        "reconexion | camera_id=%s",
                        self._source.metadata.camera_id,
                    )
                    return
                continue

            consecutive_errors = 0
            self._handle_raw_frame(raw_frame)

    def _handle_raw_frame(self, raw_frame: np.ndarray) -> None:
        timestamp = time.time()
        self.metrics.frames_received += 1
        self._fps_throttler.register_input_frame()

        try:
            validate_raw_frame(raw_frame)
        except FrameValidationError as exc:
            self.metrics.frames_dropped_total += 1
            self.metrics.frames_dropped_invalid += 1
            _rate_limited_logger.log(
                "invalid_frame",
                logging.WARNING,
                "Frames invalidos descartados (%d en la ultima ventana): " + str(exc),
            )
            return

        # Control de FPS: descarte matematico antes de encolar.
        if not self._fps_throttler.should_accept(timestamp):
            self.metrics.frames_dropped_total += 1
            self.metrics.frames_dropped_fps_control += 1
            _rate_limited_logger.log(
                "fps_throttle",
                logging.DEBUG,
                "Frames descartados por control de FPS (%d en la ultima ventana)",
            )
            return

        self._frame_counter += 1
        camera_id = self._source.metadata.camera_id
        session_id = self._source.metadata.session_id

        try:
            validate_frame_identity(camera_id, session_id, timestamp)
        except FrameValidationError as exc:
            self.metrics.frames_dropped_total += 1
            self.metrics.frames_dropped_invalid += 1
            logger.warning("Identidad de frame invalida, se descarta: %s", str(exc))
            return

        height, width, channels = raw_frame.shape
        frame = Frame(
            camera_id=camera_id,
            frame_id=self._frame_counter,
            session_id=session_id,
            stream_id=self._source.metadata.stream_id,
            timestamp=timestamp,
            image=raw_frame,
            width=width,
            height=height,
            channels=channels,
        )

        self.metrics.last_frame_timestamp = timestamp
        self._buffer.put(frame)

    # -------------------------------------------------------------------
    # Liberacion de recursos
    # -------------------------------------------------------------------

    def _safe_release_source(self) -> None:
        try:
            self._source.release()
        except Exception:
            logger.exception(
                "Error al liberar recursos de la fuente | camera_id=%s",
                self._source.metadata.camera_id,
            )
        finally:
            self._set_state(ConnectionState.DISCONNECTED)


# =============================================================================
# Punto de entrada de ejemplo (manual / smoke test local)
# =============================================================================

def build_camera_worker_from_env() -> CameraWorker:
    """Construye el CameraWorker segun variables de entorno.

    VIDEO_SOURCE_TYPE=RTSP_CAMERA usa la camara IP (RTSP_URL); si no se
    define, se usa RTSP cuando RTSP_URL existe y la camara local en otro
    caso. Resolucion y FPS son configurables para no hardcodear nada
    dependiente del despliegue.
    """
    camera_id = os.getenv("VIDEO_CAMERA_ID", "CAM-001")
    source_type = os.getenv("VIDEO_SOURCE_TYPE", "").strip().upper() or (
        SourceType.RTSP_CAMERA.value if os.getenv("RTSP_URL") else SourceType.LOCAL_CAMERA.value
    )

    source: VideoSource
    if source_type == SourceType.RTSP_CAMERA.value:
        source = RTSPCameraSource(
            camera_id=camera_id,
            open_timeout_ms=int(os.getenv("RTSP_OPEN_TIMEOUT_MS", "10000")),
            read_timeout_ms=int(os.getenv("RTSP_READ_TIMEOUT_MS", "10000")),
        )
    elif source_type == SourceType.LOCAL_CAMERA.value:
        source = LocalCameraSource(
            camera_id=camera_id,
            device_index=int(os.getenv("VIDEO_DEVICE_INDEX", "0")),
            width=int(os.getenv("VIDEO_CAPTURE_WIDTH", "1280")),
            height=int(os.getenv("VIDEO_CAPTURE_HEIGHT", "720")),
            fps=int(os.getenv("VIDEO_CAPTURE_FPS", "30")),
        )
    else:
        raise ValueError(f"VIDEO_SOURCE_TYPE no soportado: {source_type}")

    config = CameraWorkerConfig(
        capture_width=int(os.getenv("VIDEO_CAPTURE_WIDTH", "1280")),
        capture_height=int(os.getenv("VIDEO_CAPTURE_HEIGHT", "720")),
        capture_fps=int(os.getenv("VIDEO_CAPTURE_FPS", "30")),
        target_fps=float(os.getenv("VIDEO_TARGET_FPS", "10")),
        buffer_max_size=int(os.getenv("VIDEO_BUFFER_MAX_SIZE", "30")),
        max_reconnect_attempts=int(os.getenv("VIDEO_MAX_RECONNECT_ATTEMPTS", "10")),
    )

    return CameraWorker(source=source, config=config)


if __name__ == "__main__":
    worker = build_camera_worker_from_env()
    worker.start()

    try:
        while True:
            frame_dict = worker.get_next_frame(timeout=1.0)
            if frame_dict is not None:
                logger.debug(
                    "Frame entregado | camera_id=%s frame_id=%s",
                    frame_dict["camera_id"],
                    frame_dict["frame_id"],
                )
            else:
                logger.debug("Health: %s", worker.health_check())
    except KeyboardInterrupt:
        logger.info("Interrupcion manual recibida, iniciando cierre controlado...")
    finally:
        worker.stop()
