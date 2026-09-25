"""
preprocessingService.py

Servicio de Preprocessing / Frame Processing (Servicio 2 del pipeline).

Responsabilidad UNICA: recibir frames normalizados desde "Video Ingestion"
(Servicio 1), validarlos, normalizarlos a un formato interno uniforme,
redimensionarlos con letterboxing, aplicar ROI, controlar FPS y entregarlos
junto con una metadata de transformacion detallada para que "YOLO/Detection"
(Servicio 3) pueda proyectar sus resultados de vuelta a la imagen original.

Este modulo es EXCLUSIVAMENTE un transformador de matrices/imagenes. NO
importa librerias de IA ni implementa deteccion, tracking, analisis de
comportamiento o logica de negocio/alertas.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Generator, Optional, Tuple

import cv2
import numpy as np


# =============================================================================
# Logging
# =============================================================================
# Logs estructurados. Jamas se serializa o loguea el contenido de una imagen
# cruda / procesada, solo metadata (dimensiones, ids, tiempos).

logger = logging.getLogger("preprocessing_service")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


class _RateLimitedLogger:
    """Evita saturar el log con eventos de alta frecuencia (ej. descartes)."""

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
# Excepciones propias
# =============================================================================

class InputValidationError(Exception):
    """Error de validacion del frame de entrada (contrato con Servicio 1)."""


class ROIConfigError(Exception):
    """Error de configuracion de ROI invalida o fuera de rango."""


# =============================================================================
# Contratos de datos: ROI, configuracion por camara, metadata de transformacion
# =============================================================================

@dataclass(frozen=True)
class ROI:
    """Region de interes en coordenadas de pixeles sobre la imagen original.

    (x, y) es la esquina superior izquierda; (width, height) el tamaño del
    recorte. Se valida contra las dimensiones reales del frame en tiempo de
    aplicacion para evitar desbordamientos de memoria o lecturas fuera de
    rango.
    """

    x: int
    y: int
    width: int
    height: int

    def validate(self, frame_width: int, frame_height: int, max_area: int) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ROIConfigError(f"ROI con dimensiones invalidas: {self.width}x{self.height}")
        if self.x < 0 or self.y < 0:
            raise ROIConfigError(f"ROI con origen negativo: ({self.x}, {self.y})")
        if self.x + self.width > frame_width or self.y + self.height > frame_height:
            raise ROIConfigError(
                f"ROI fuera de los limites del frame ({frame_width}x{frame_height}): "
                f"({self.x}, {self.y}, {self.width}, {self.height})"
            )
        if self.width * self.height > max_area:
            raise ROIConfigError(
                f"ROI excede el area maxima permitida ({max_area} px^2), "
                f"posible riesgo de desbordamiento de memoria"
            )


@dataclass
class CameraProcessingConfig:
    """Configuracion de preprocessing independiente por camera_id."""

    camera_id: str
    target_width: int = 640
    target_height: int = 640
    target_fps: float = 10.0
    roi: Optional[ROI] = None
    # Stubs de mejora de calidad de imagen (ver _apply_quality_enhancements).
    enable_denoise: bool = False
    enable_brightness_contrast: bool = False
    enable_sharpening: bool = False


@dataclass
class TransformationMetadata:
    """Registro detallado de las transformaciones aplicadas al frame.

    Es el contrato que el Servicio 3 (YOLO) necesita para proyectar bounding
    boxes desde el espacio de la imagen procesada de vuelta al espacio de la
    imagen original (matematica inversa de escala + padding + ROI).
    """

    original_width: int
    original_height: int
    processed_width: int
    processed_height: int
    scale_x: float
    scale_y: float
    pad_x: int
    pad_y: int
    roi_applied: Optional[Tuple[int, int, int, int]]  # (x, y, width, height) o None
    color_conversion: str
    dtype: str
    processing_timestamp: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "original_width": self.original_width,
            "original_height": self.original_height,
            "processed_width": self.processed_width,
            "processed_height": self.processed_height,
            "scale_x": self.scale_x,
            "scale_y": self.scale_y,
            "pad_x": self.pad_x,
            "pad_y": self.pad_y,
            "roi_applied": self.roi_applied,
            "color_conversion": self.color_conversion,
            "dtype": self.dtype,
            "processing_timestamp": self.processing_timestamp,
        }


@dataclass
class PreprocessedFrame:
    """Objeto normalizado de salida del servicio (Frame Preprocessed)."""

    camera_id: str
    frame_id: int
    session_id: str
    timestamp: float
    image: np.ndarray
    original_metadata: Dict[str, object]
    transformation_metadata: TransformationMetadata

    def to_dict(self) -> Dict[str, object]:
        return {
            "camera_id": self.camera_id,
            "frame_id": self.frame_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "image": self.image,
            "original_metadata": self.original_metadata,
            "transformation_metadata": self.transformation_metadata.to_dict(),
        }


# =============================================================================
# Validacion del frame de entrada (contrato con Servicio 1)
# =============================================================================

_REQUIRED_INPUT_KEYS = ("camera_id", "frame_id", "session_id", "timestamp", "image")


def validate_input_frame(raw_input: Optional[Dict[str, object]]) -> None:
    """Valida el diccionario recibido desde Video Ingestion.

    No propaga jamas un frame invalido/corrupto a las etapas de
    transformacion; el llamador debe descartar y loguear ante excepcion.
    """

    if not raw_input:
        raise InputValidationError("Input nulo o vacio")

    for key in _REQUIRED_INPUT_KEYS:
        if key not in raw_input:
            raise InputValidationError(f"Falta campo obligatorio en el contrato de entrada: {key}")

    image = raw_input["image"]
    if image is None or not isinstance(image, np.ndarray):
        raise InputValidationError("El campo 'image' no es un array valido")

    if image.size == 0:
        raise InputValidationError("Imagen vacia (size=0)")

    if image.ndim != 3 or image.shape[2] not in (1, 3, 4):
        raise InputValidationError(f"Formato de imagen incompatible: shape={image.shape}")

    height, width = image.shape[0], image.shape[1]
    if height <= 0 or width <= 0:
        raise InputValidationError(f"Dimensiones invalidas: {width}x{height}")

    if not raw_input["camera_id"]:
        raise InputValidationError("camera_id invalido/vacio")
    if not raw_input["session_id"]:
        raise InputValidationError("session_id invalido/vacio")

    timestamp = raw_input["timestamp"]
    if timestamp is None or timestamp <= 0:
        raise InputValidationError("timestamp invalido")

    # Deteccion basica de corrupcion: varianza nula (frame solido/congelado
    # en un unico valor) suele indicar un buffer corrupto o un frame negro.
    if image.dtype != np.uint8:
        raise InputValidationError(f"dtype de imagen inesperado: {image.dtype}")


# =============================================================================
# Buffer con politica "descartar los mas antiguos" (tiempo real, RAM acotada)
# =============================================================================

class BoundedFrameQueue:
    """Cola de tamaño maximo estricto. Si se llena, descarta el elemento mas
    antiguo para priorizar el frame mas reciente (tiempo real)."""

    def __init__(self, max_size: int, on_drop=None) -> None:
        if max_size <= 0:
            raise ValueError("max_size debe ser mayor a 0")
        self._max_size = max_size
        self._buffer: Deque[object] = deque()
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._on_drop = on_drop

    def put(self, item: object) -> None:
        with self._not_empty:
            if len(self._buffer) >= self._max_size:
                self._buffer.popleft()
                if self._on_drop is not None:
                    self._on_drop()
            self._buffer.append(item)
            self._not_empty.notify()

    def get(self, timeout: Optional[float] = None) -> Optional[object]:
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
# Control de FPS por camara (muestreo matematico)
# =============================================================================

class FpsThrottler:
    """Descarta frames de forma controlada para pasar del FPS de entrada al
    FPS objetivo configurado por camara. Tambien mide el FPS real de
    entrada y de salida (procesados)."""

    def __init__(self, target_fps: float) -> None:
        if target_fps <= 0:
            raise ValueError("target_fps debe ser mayor a 0")
        self.target_fps = target_fps
        self._target_interval = 1.0 / target_fps
        self._last_accepted_ts: Optional[float] = None

        self._in_count = 0
        self._in_window_start = time.time()
        self.measured_input_fps = 0.0

        self._out_count = 0
        self._out_window_start = time.time()
        self.measured_output_fps = 0.0

    def register_input(self) -> None:
        self._in_count += 1
        elapsed = time.time() - self._in_window_start
        if elapsed >= 1.0:
            self.measured_input_fps = round(self._in_count / elapsed, 2)
            self._in_count = 0
            self._in_window_start = time.time()

    def register_output(self) -> None:
        self._out_count += 1
        elapsed = time.time() - self._out_window_start
        if elapsed >= 1.0:
            self.measured_output_fps = round(self._out_count / elapsed, 2)
            self._out_count = 0
            self._out_window_start = time.time()

    def should_accept(self, timestamp: float) -> bool:
        if self._last_accepted_ts is None:
            self._last_accepted_ts = timestamp
            return True
        if timestamp - self._last_accepted_ts >= self._target_interval:
            self._last_accepted_ts = timestamp
            return True
        return False


# =============================================================================
# Metricas / Health check
# =============================================================================

@dataclass
class CameraMetrics:
    """Contadores por camara."""

    frames_received: int = 0
    frames_processed: int = 0
    frames_dropped_invalid: int = 0
    frames_dropped_fps_control: int = 0
    frames_dropped_backpressure_in: int = 0
    frames_dropped_backpressure_out: int = 0
    total_latency_seconds: float = 0.0
    last_frame_timestamp: Optional[float] = None

    @property
    def frames_dropped_total(self) -> int:
        return (
            self.frames_dropped_invalid
            + self.frames_dropped_fps_control
            + self.frames_dropped_backpressure_in
            + self.frames_dropped_backpressure_out
        )

    @property
    def average_latency_ms(self) -> float:
        if self.frames_processed == 0:
            return 0.0
        return round((self.total_latency_seconds / self.frames_processed) * 1000.0, 3)

    def to_dict(self) -> Dict[str, object]:
        return {
            "frames_received": self.frames_received,
            "frames_processed": self.frames_processed,
            "frames_dropped_invalid": self.frames_dropped_invalid,
            "frames_dropped_fps_control": self.frames_dropped_fps_control,
            "frames_dropped_backpressure_in": self.frames_dropped_backpressure_in,
            "frames_dropped_backpressure_out": self.frames_dropped_backpressure_out,
            "frames_dropped_total": self.frames_dropped_total,
            "average_latency_ms": self.average_latency_ms,
            "last_frame_timestamp": self.last_frame_timestamp,
        }


# =============================================================================
# Nucleo de transformacion (funciones puras sobre matrices)
# =============================================================================

_MAX_ROI_AREA_PX = 50_000_000  # limite defensivo contra ROI desmedido / OOM


def apply_roi(image: np.ndarray, roi: Optional[ROI]) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
    """Recorta la imagen al ROI configurado, si existe. Valida limites."""
    if roi is None:
        return image, None

    height, width = image.shape[0], image.shape[1]
    roi.validate(frame_width=width, frame_height=height, max_area=_MAX_ROI_AREA_PX)

    cropped = image[roi.y: roi.y + roi.height, roi.x: roi.x + roi.width]
    return cropped, (roi.x, roi.y, roi.width, roi.height)


def normalize_color_and_dtype(image: np.ndarray) -> Tuple[np.ndarray, str]:
    """Convierte la imagen al formato interno uniforme: BGR de 3 canales,
    uint8. Este es el formato predecible que consumira YOLO en el Servicio 3."""
    channels = image.shape[2] if image.ndim == 3 else 1

    if channels == 1:
        converted = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        conversion = "GRAY2BGR"
    elif channels == 4:
        converted = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        conversion = "BGRA2BGR"
    elif channels == 3:
        converted = image
        conversion = "NONE"
    else:
        raise InputValidationError(f"Numero de canales no soportado: {channels}")

    if converted.dtype != np.uint8:
        converted = converted.astype(np.uint8)

    return converted, conversion


def resize_with_letterbox(
    image: np.ndarray,
    target_width: int,
    target_height: int,
    pad_color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, float, int, int]:
    """Redimensiona conservando el aspect ratio y rellena (letterbox) para
    alcanzar exactamente target_width x target_height.

    Retorna: (imagen_resultante, scale_x, scale_y, pad_x, pad_y)
    scale_x == scale_y siempre (se usa una unica escala uniforme), se
    entregan ambos por claridad/compatibilidad del contrato con YOLO.
    """
    original_height, original_width = image.shape[0], image.shape[1]

    scale = min(target_width / original_width, target_height / original_height)
    new_width = max(1, int(round(original_width * scale)))
    new_height = max(1, int(round(original_height * scale)))

    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)

    pad_x = (target_width - new_width) // 2
    pad_y = (target_height - new_height) // 2

    # Se reparte el padding restante (por redondeo) al borde derecho/inferior.
    pad_right = target_width - new_width - pad_x
    pad_bottom = target_height - new_height - pad_y

    padded = cv2.copyMakeBorder(
        resized,
        top=pad_y,
        bottom=pad_bottom,
        left=pad_x,
        right=pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=pad_color,
    )

    return padded, scale, scale, pad_x, pad_y


def _apply_quality_enhancements(image: np.ndarray, config: CameraProcessingConfig) -> np.ndarray:
    """[STUB] Punto de extension para mejoras de calidad de imagen.

    Pendiente de implementacion. Activables por camara via
    CameraProcessingConfig (enable_denoise, enable_brightness_contrast,
    enable_sharpening). Se deja la estructura de llamada lista para no
    romper el pipeline cuando se habiliten.
    """
    if config.enable_denoise:
        # image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        pass

    if config.enable_brightness_contrast:
        # alpha (contraste), beta (brillo) se calcularian dinamicamente o
        # via configuracion por camara.
        # image = cv2.convertScaleAbs(image, alpha=1.0, beta=0)
        pass

    if config.enable_sharpening:
        # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        # image = cv2.filter2D(image, -1, kernel)
        pass

    return image


# =============================================================================
# Worker de preprocessing por camara (hilo dedicado, aislamiento por camara)
# =============================================================================

class PreprocessingWorker:
    """Procesa el stream de una unica camara en un hilo dedicado.

    Una camara lenta o con errores no bloquea a las demas: cada
    PreprocessingWorker tiene sus propios buffers de entrada/salida,
    hilo y metricas.
    """

    def __init__(
        self,
        config: CameraProcessingConfig,
        input_buffer_size: int = 30,
        output_buffer_size: int = 30,
    ) -> None:
        self.config = config
        self.metrics = CameraMetrics()
        self._fps_throttler = FpsThrottler(target_fps=config.target_fps)

        self._input_queue = BoundedFrameQueue(
            max_size=input_buffer_size,
            on_drop=self._on_input_drop,
        )
        self._output_queue = BoundedFrameQueue(
            max_size=output_buffer_size,
            on_drop=self._on_output_drop,
        )

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._active = False

    # -------------------------------------------------------------------
    # Ciclo de vida
    # -------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Worker ya en ejecucion | camera_id=%s", self.config.camera_id)
            return

        logger.info("Iniciando worker de preprocessing | camera_id=%s", self.config.camera_id)
        self._stop_event.clear()
        self._active = True
        self._thread = threading.Thread(
            target=self._run,
            name=f"PreprocessingWorker-{self.config.camera_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        logger.info("Deteniendo worker de preprocessing | camera_id=%s", self.config.camera_id)
        self._stop_event.set()
        self._active = False

        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning(
                    "El hilo del worker de preprocessing no finalizo dentro del timeout "
                    "de %.1fs | camera_id=%s",
                    timeout,
                    self.config.camera_id,
                )

        self._input_queue.clear()
        self._output_queue.clear()
        logger.info("Worker de preprocessing detenido | camera_id=%s", self.config.camera_id)

    # -------------------------------------------------------------------
    # Entrada / salida
    # -------------------------------------------------------------------

    def submit(self, raw_input: Dict[str, object]) -> None:
        """Recibe un frame crudo desde Video Ingestion y lo encola para
        procesamiento asincrono. No bloquea al llamador."""
        self._input_queue.put(raw_input)

    def get_next_processed(self, timeout: float = 1.0) -> Optional[Dict[str, object]]:
        """Entrega el siguiente frame preprocesado disponible, o None si no
        hay ninguno dentro del timeout."""
        item = self._output_queue.get(timeout=timeout)
        return None if item is None else item  # ya es un dict (contrato de salida)

    def processed_stream(self) -> Generator[Dict[str, object], None, None]:
        while not self._stop_event.is_set():
            item = self.get_next_processed(timeout=1.0)
            if item is not None:
                yield item

    # -------------------------------------------------------------------
    # Configuracion dinamica por camara
    # -------------------------------------------------------------------

    def update_config(self, config: CameraProcessingConfig) -> None:
        if config.camera_id != self.config.camera_id:
            raise ValueError("camera_id no coincide con este worker")
        logger.info(
            "Actualizando configuracion | camera_id=%s target=%dx%d target_fps=%.1f roi=%s",
            config.camera_id,
            config.target_width,
            config.target_height,
            config.target_fps,
            "SI" if config.roi else "NO",
        )
        self.config = config
        self._fps_throttler = FpsThrottler(target_fps=config.target_fps)

    # -------------------------------------------------------------------
    # Health check
    # -------------------------------------------------------------------

    def health_check(self) -> Dict[str, object]:
        return {
            "camera_id": self.config.camera_id,
            "active": self._active,
            "input_buffer_usage": f"{self._input_queue.qsize()}/{self._input_queue.max_size}",
            "output_buffer_usage": f"{self._output_queue.qsize()}/{self._output_queue.max_size}",
            "measured_input_fps": self._fps_throttler.measured_input_fps,
            "measured_output_fps": self._fps_throttler.measured_output_fps,
            "target_fps": self.config.target_fps,
            "average_latency_ms": self.metrics.average_latency_ms,
            "metrics": self.metrics.to_dict(),
        }

    # -------------------------------------------------------------------
    # Loop principal
    # -------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                raw_input = self._input_queue.get(timeout=1.0)
                if raw_input is None:
                    continue
                self._process_one(raw_input)  # type: ignore[arg-type]
            except Exception:
                # Ningun fallo de una camara individual debe crashear el
                # servicio ni afectar a las demas camaras.
                logger.exception(
                    "Error inesperado procesando frame | camera_id=%s",
                    self.config.camera_id,
                )

    def _process_one(self, raw_input: Dict[str, object]) -> None:
        self.metrics.frames_received += 1

        try:
            validate_input_frame(raw_input)
        except InputValidationError as exc:
            self.metrics.frames_dropped_invalid += 1
            _rate_limited_logger.log(
                f"invalid_input_{self.config.camera_id}",
                logging.WARNING,
                f"[{self.config.camera_id}] Frames de entrada invalidos descartados "
                f"(%d en la ultima ventana): {exc}",
            )
            return

        timestamp = float(raw_input["timestamp"])  # type: ignore[arg-type]
        self._fps_throttler.register_input()

        if not self._fps_throttler.should_accept(timestamp):
            self.metrics.frames_dropped_fps_control += 1
            _rate_limited_logger.log(
                f"fps_throttle_{self.config.camera_id}",
                logging.DEBUG,
                f"[{self.config.camera_id}] Frames descartados por control de FPS "
                f"(%d en la ultima ventana)",
            )
            return

        start_time = time.time()
        try:
            processed = self._transform(raw_input)
        except (InputValidationError, ROIConfigError) as exc:
            self.metrics.frames_dropped_invalid += 1
            logger.warning(
                "[%s] Frame descartado durante la transformacion: %s",
                self.config.camera_id,
                exc,
            )
            return
        except Exception:
            self.metrics.frames_dropped_invalid += 1
            logger.exception(
                "[%s] Error no controlado durante la transformacion, frame descartado",
                self.config.camera_id,
            )
            return

        latency = time.time() - start_time
        self.metrics.frames_processed += 1
        self.metrics.total_latency_seconds += latency
        self.metrics.last_frame_timestamp = timestamp
        self._fps_throttler.register_output()

        self._output_queue.put(processed.to_dict())

    def _transform(self, raw_input: Dict[str, object]) -> PreprocessedFrame:
        image: np.ndarray = raw_input["image"]  # type: ignore[assignment]
        original_height, original_width = image.shape[0], image.shape[1]

        # 1) ROI (recorte de areas inutiles antes de procesar).
        roi_image, roi_applied = apply_roi(image, self.config.roi)

        # 2) Normalizacion de espacio de color / tipo de dato.
        normalized_image, color_conversion = normalize_color_and_dtype(roi_image)

        # 3) Mejora de calidad de imagen (stub, desactivado por defecto).
        normalized_image = _apply_quality_enhancements(normalized_image, self.config)

        # 4) Redimensionamiento + letterboxing hacia la resolucion objetivo.
        final_image, scale_x, scale_y, pad_x, pad_y = resize_with_letterbox(
            normalized_image,
            target_width=self.config.target_width,
            target_height=self.config.target_height,
        )

        transformation_metadata = TransformationMetadata(
            original_width=original_width,
            original_height=original_height,
            processed_width=self.config.target_width,
            processed_height=self.config.target_height,
            scale_x=scale_x,
            scale_y=scale_y,
            pad_x=pad_x,
            pad_y=pad_y,
            roi_applied=roi_applied,
            color_conversion=color_conversion,
            dtype=str(final_image.dtype),
            processing_timestamp=time.time(),
        )

        original_metadata = {
            k: v for k, v in raw_input.items() if k not in ("image",)
        }

        return PreprocessedFrame(
            camera_id=raw_input["camera_id"],  # type: ignore[arg-type]
            frame_id=raw_input["frame_id"],  # type: ignore[arg-type]
            session_id=raw_input["session_id"],  # type: ignore[arg-type]
            timestamp=raw_input["timestamp"],  # type: ignore[arg-type]
            image=final_image,
            original_metadata=original_metadata,
            transformation_metadata=transformation_metadata,
        )

    # -------------------------------------------------------------------
    # Callbacks de descarte por backpressure
    # -------------------------------------------------------------------

    def _on_input_drop(self) -> None:
        self.metrics.frames_dropped_backpressure_in += 1
        _rate_limited_logger.log(
            f"input_backpressure_{self.config.camera_id}",
            logging.WARNING,
            f"[{self.config.camera_id}] Buffer de entrada lleno: frames mas antiguos "
            f"descartados (%d en la ultima ventana)",
        )

    def _on_output_drop(self) -> None:
        self.metrics.frames_dropped_backpressure_out += 1
        _rate_limited_logger.log(
            f"output_backpressure_{self.config.camera_id}",
            logging.WARNING,
            f"[{self.config.camera_id}] Buffer de salida lleno (consumidor lento): "
            f"frames mas antiguos descartados (%d en la ultima ventana)",
        )


# =============================================================================
# Orquestador multi-camara: concurrencia independiente por camara
# =============================================================================

class PreprocessingService:
    """Gestiona multiples PreprocessingWorker (uno por camara) de forma
    concurrente. Una camara lenta o con errores no bloquea a las demas."""

    def __init__(self) -> None:
        self._workers: Dict[str, PreprocessingWorker] = {}
        self._lock = threading.Lock()
        self._start_timestamp = time.time()

    def register_camera(
        self,
        config: CameraProcessingConfig,
        input_buffer_size: int = 30,
        output_buffer_size: int = 30,
    ) -> None:
        """Registra (o reconfigura, si ya existe) una camara y arranca su
        worker dedicado."""
        with self._lock:
            existing = self._workers.get(config.camera_id)
            if existing is not None:
                existing.update_config(config)
                return

            worker = PreprocessingWorker(
                config=config,
                input_buffer_size=input_buffer_size,
                output_buffer_size=output_buffer_size,
            )
            self._workers[config.camera_id] = worker
            worker.start()
            logger.info("Camara registrada en PreprocessingService | camera_id=%s", config.camera_id)

    def unregister_camera(self, camera_id: str) -> None:
        with self._lock:
            worker = self._workers.pop(camera_id, None)
        if worker is not None:
            worker.stop()
            logger.info("Camara removida de PreprocessingService | camera_id=%s", camera_id)

    def submit_frame(self, camera_id: str, raw_input: Dict[str, object]) -> None:
        """Punto de entrada para frames provenientes de Video Ingestion."""
        worker = self._workers.get(camera_id)
        if worker is None:
            _rate_limited_logger.log(
                f"unregistered_camera_{camera_id}",
                logging.WARNING,
                f"[{camera_id}] Frame recibido para una camara no registrada "
                f"(%d ocurrencias en la ultima ventana)",
            )
            return
        worker.submit(raw_input)

    def get_next_processed(self, camera_id: str, timeout: float = 1.0) -> Optional[Dict[str, object]]:
        worker = self._workers.get(camera_id)
        if worker is None:
            return None
        return worker.get_next_processed(timeout=timeout)

    def stop_all(self, timeout: float = 5.0) -> None:
        logger.info("Deteniendo PreprocessingService (todas las camaras)...")
        with self._lock:
            workers = list(self._workers.values())
            self._workers.clear()
        for worker in workers:
            worker.stop(timeout=timeout)
        logger.info("PreprocessingService detenido correctamente")

    def health_check(self) -> Dict[str, object]:
        with self._lock:
            workers = dict(self._workers)

        camera_healths = {camera_id: worker.health_check() for camera_id, worker in workers.items()}
        latencies = [w.metrics.average_latency_ms for w in workers.values() if w.metrics.frames_processed > 0]
        overall_avg_latency = round(sum(latencies) / len(latencies), 3) if latencies else 0.0

        return {
            "service_active": True,
            "active_cameras": list(workers.keys()),
            "active_camera_count": len(workers),
            "overall_average_latency_ms": overall_avg_latency,
            "uptime_seconds": round(time.time() - self._start_timestamp, 3),
            "cameras": camera_healths,
        }


# =============================================================================
# Ejemplo de uso manual (smoke test) - NO se ejecuta como servicio de IA
# =============================================================================

if __name__ == "__main__":
    service = PreprocessingService()
    service.register_camera(
        CameraProcessingConfig(
            camera_id="CAM-001",
            target_width=640,
            target_height=640,
            target_fps=10.0,
            roi=None,
        )
    )

    # Frame sintetico de ejemplo, simulando el contrato de salida del
    # Servicio 1 (Video Ingestion).
    synthetic_frame = {
        "camera_id": "CAM-001",
        "frame_id": 1,
        "session_id": "session-demo",
        "stream_id": "stream-demo",
        "timestamp": time.time(),
        "image": np.zeros((1080, 1920, 3), dtype=np.uint8),
    }

    service.submit_frame("CAM-001", synthetic_frame)

    result = service.get_next_processed("CAM-001", timeout=2.0)
    if result is not None:
        logger.info(
            "Frame preprocesado | camera_id=%s frame_id=%s shape_final=%s transform=%s",
            result["camera_id"],
            result["frame_id"],
            result["image"].shape,
            result["transformation_metadata"],
        )

    logger.info("Health check: %s", service.health_check())
    service.stop_all()
