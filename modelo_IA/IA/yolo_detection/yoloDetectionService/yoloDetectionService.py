"""
yoloDetectionService.py

Servicio de YOLO / Detection (Servicio 3 del pipeline).

Responsabilidad UNICA: recibir frames preprocesados desde "Preprocessing"
(Servicio 2), ejecutar inferencia de deteccion de objetos sobre GPU/CPU con
batching dinamico multi-camara, proyectar las bounding boxes de vuelta a la
resolucion original de cada camara (usando la metadata de transformacion) y
emitir un DetectionResult estructurado (sin imagen) hacia "Tracking"
(Servicio 4).

YOLO OBSERVA, NO INTERPRETA: este modulo NO implementa tracking
(DeepSORT/ByteTrack), identificacion persistente, pose, analisis de
comportamiento ni logica de decision. Las confianzas del modelo se
entregan intactas, sin reglas de negocio aplicadas sobre ellas.
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
from typing import Deque, Dict, Generator, List, Optional, Sequence, Set, Tuple

import numpy as np


# =============================================================================
# Logging
# =============================================================================
# Logs estructurados. Jamas se loguean arrays de imagenes/frames ni metadata
# excesiva por cada frame individual; los eventos de alta frecuencia usan
# rate-limiting.

logger = logging.getLogger("yolo_detection_service")
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
    """Error de validacion del frame preprocesado de entrada."""


class ModelLoadError(Exception):
    """Error al cargar/inicializar el modelo de deteccion."""


class InferenceOOMError(Exception):
    """Error de memoria (VRAM) durante la inferencia. Recuperable."""


# =============================================================================
# Enums de configuracion
# =============================================================================

class Device(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"


class Precision(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"


class ModelBackend(str, Enum):
    """Motores de inferencia soportados (o preparados) por el servicio."""

    ULTRALYTICS = "ULTRALYTICS"       # Activo.
    ONNX_RUNTIME = "ONNX_RUNTIME"     # Reservado. Esqueleto listo.
    TENSORRT = "TENSORRT"             # Reservado. Esqueleto listo.


# =============================================================================
# Configuracion (100% desde variables de entorno, nada hardcodeado)
# =============================================================================

@dataclass(frozen=True)
class ModelConfig:
    """Configuracion del modelo y de la inferencia, leida de variables de
    entorno. Nunca se hardcodean rutas de modelos ni credenciales."""

    backend: ModelBackend
    model_path: str
    device: Device
    precision: Precision
    confidence_threshold: float
    iou_threshold: float
    class_whitelist: Optional[Set[int]]
    max_batch_size: int
    batch_timeout_ms: float
    input_queue_size: int
    output_queue_size_per_camera: int

    @staticmethod
    def from_env() -> "ModelConfig":
        whitelist_raw = os.getenv("MODEL_CLASS_WHITELIST", "").strip()
        class_whitelist: Optional[Set[int]] = None
        if whitelist_raw:
            try:
                class_whitelist = {int(x.strip()) for x in whitelist_raw.split(",") if x.strip()}
            except ValueError as exc:
                raise ValueError(
                    f"MODEL_CLASS_WHITELIST invalido, se esperaban ids numericos separados "
                    f"por coma: {whitelist_raw!r}"
                ) from exc

        return ModelConfig(
            backend=ModelBackend(os.getenv("MODEL_BACKEND", ModelBackend.ULTRALYTICS.value)),
            model_path=os.getenv("MODEL_PATH", "models/yolo.pt"),
            device=Device(os.getenv("MODEL_DEVICE", Device.CPU.value)),
            precision=Precision(os.getenv("MODEL_PRECISION", Precision.FP32.value)),
            confidence_threshold=float(os.getenv("MODEL_CONF_THRESHOLD", "0.25")),
            iou_threshold=float(os.getenv("MODEL_IOU_THRESHOLD", "0.45")),
            class_whitelist=class_whitelist,
            max_batch_size=int(os.getenv("MODEL_MAX_BATCH_SIZE", "8")),
            batch_timeout_ms=float(os.getenv("MODEL_BATCH_TIMEOUT_MS", "10")),
            input_queue_size=int(os.getenv("MODEL_INPUT_QUEUE_SIZE", "64")),
            output_queue_size_per_camera=int(os.getenv("MODEL_OUTPUT_QUEUE_SIZE", "30")),
        )

    def validate_model_path(self) -> None:
        """Valida que la ruta del modelo exista antes de intentar cargarlo.
        Nunca se registra el contenido del archivo, solo la ruta."""
        if not self.model_path:
            raise ModelLoadError("MODEL_PATH no configurado")
        if not os.path.isfile(self.model_path):
            raise ModelLoadError(f"No se encontro el archivo de modelo en: {self.model_path}")


@dataclass(frozen=True)
class ModelInfo:
    """Identificacion del modelo activo, expuesta en cada DetectionResult."""

    name: str
    version: str
    backend: str
    device: str
    precision: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "version": self.version,
            "backend": self.backend,
            "device": self.device,
            "precision": self.precision,
        }


# =============================================================================
# Contratos de datos: deteccion cruda, bbox proyectada, resultado final
# =============================================================================

@dataclass
class RawDetection:
    """Deteccion cruda devuelta por el motor de inferencia, en coordenadas
    del frame PRE-PROCESADO (ej. 640x640 con letterboxing), ya filtrada por
    el NMS nativo del modelo."""

    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: Tuple[float, float, float, float]  # coords en el frame procesado


@dataclass(frozen=True)
class BBoxOriginal:
    """Bounding box proyectada de vuelta a la resolucion original de la
    camara (deshecho el padding, la escala y el offset de ROI)."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    def to_dict(self) -> Dict[str, float]:
        return {
            "x1": round(self.x1, 2),
            "y1": round(self.y1, 2),
            "x2": round(self.x2, 2),
            "y2": round(self.y2, 2),
            "width": round(self.width, 2),
            "height": round(self.height, 2),
        }


@dataclass
class Detection:
    """Deteccion individual en el contrato de salida, con coordenadas ya
    proyectadas a la imagen original. La confianza del modelo se entrega
    intacta, sin reglas de negocio aplicadas."""

    detection_id: str
    class_id: int
    class_name: str
    confidence: float
    bbox_original_coords: BBoxOriginal

    def to_dict(self) -> Dict[str, object]:
        return {
            "detection_id": self.detection_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox_original_coords": self.bbox_original_coords.to_dict(),
        }


@dataclass
class DetectionResult:
    """Contrato de salida hacia el Servicio 4 (Tracking). El frame original
    NUNCA se incluye, solo metadata estructurada."""

    camera_id: str
    session_id: str
    frame_id: int
    timestamp: float
    model_info: ModelInfo
    inference_time_ms: float
    detections: List[Detection]

    def to_dict(self) -> Dict[str, object]:
        return {
            "camera_id": self.camera_id,
            "session_id": self.session_id,
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "model_info": self.model_info.to_dict(),
            "inference_time_ms": self.inference_time_ms,
            "detections": [d.to_dict() for d in self.detections],
        }


# =============================================================================
# Validacion del contrato de entrada (desde Preprocessing)
# =============================================================================

_REQUIRED_INPUT_KEYS = (
    "camera_id",
    "frame_id",
    "session_id",
    "timestamp",
    "image",
    "transformation_metadata",
)
_REQUIRED_TRANSFORM_KEYS = (
    "scale_x",
    "scale_y",
    "pad_x",
    "pad_y",
    "original_width",
    "original_height",
)


def validate_preprocessed_input(payload: Optional[Dict[str, object]]) -> None:
    """Valida el diccionario recibido desde Preprocessing antes de encolarlo
    para inferencia. Descarta (no propaga) cualquier entrada incompleta o
    corrupta."""

    if not payload:
        raise InputValidationError("Input nulo o vacio")

    for key in _REQUIRED_INPUT_KEYS:
        if key not in payload:
            raise InputValidationError(f"Falta campo obligatorio en el contrato de entrada: {key}")

    image = payload["image"]
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise InputValidationError("Campo 'image' invalido o vacio")
    if image.ndim != 3 or image.shape[2] != 3:
        raise InputValidationError(f"Formato de imagen incompatible: shape={image.shape}")

    transform = payload["transformation_metadata"]
    if not isinstance(transform, dict):
        raise InputValidationError("transformation_metadata invalida (se esperaba dict)")
    for key in _REQUIRED_TRANSFORM_KEYS:
        if key not in transform:
            raise InputValidationError(f"Falta campo obligatorio en transformation_metadata: {key}")

    if not payload["camera_id"]:
        raise InputValidationError("camera_id invalido/vacio")
    if not payload["session_id"]:
        raise InputValidationError("session_id invalido/vacio")
    timestamp = payload["timestamp"]
    if timestamp is None or timestamp <= 0:
        raise InputValidationError("timestamp invalido")


# =============================================================================
# Proyeccion inversa de coordenadas (letterbox + ROI -> imagen original)
# =============================================================================

def reproject_bbox(
    bbox_xyxy_processed: Tuple[float, float, float, float],
    transformation_metadata: Dict[str, object],
) -> BBoxOriginal:
    """Deshace, en orden inverso, las transformaciones aplicadas por el
    Servicio de Preprocessing (letterbox resize -> padding -> ROI) para
    devolver la bounding box en coordenadas de la imagen original de la
    camara.

    Orden de las transformaciones originales: ROI -> resize con escala ->
    padding. La proyeccion inversa deshace en orden contrario: padding ->
    escala -> offset de ROI.
    """

    x1, y1, x2, y2 = bbox_xyxy_processed

    pad_x = float(transformation_metadata["pad_x"])
    pad_y = float(transformation_metadata["pad_y"])
    scale_x = float(transformation_metadata["scale_x"])
    scale_y = float(transformation_metadata["scale_y"])

    if scale_x <= 0 or scale_y <= 0:
        raise InputValidationError(f"Factores de escala invalidos: scale_x={scale_x}, scale_y={scale_y}")

    # 1) Deshacer el padding del letterbox.
    x1 -= pad_x
    y1 -= pad_y
    x2 -= pad_x
    y2 -= pad_y

    # 2) Deshacer la escala del resize.
    x1 /= scale_x
    y1 /= scale_y
    x2 /= scale_x
    y2 /= scale_y

    # 3) Deshacer el offset del ROI, si se aplico uno antes del preprocessing.
    roi_applied = transformation_metadata.get("roi_applied")
    if roi_applied:
        roi_x, roi_y = float(roi_applied[0]), float(roi_applied[1])
        x1 += roi_x
        y1 += roi_y
        x2 += roi_x
        y2 += roi_y

    # 4) Recorte defensivo a los limites de la imagen original.
    original_width = float(transformation_metadata["original_width"])
    original_height = float(transformation_metadata["original_height"])

    x1 = min(max(x1, 0.0), original_width)
    y1 = min(max(y1, 0.0), original_height)
    x2 = min(max(x2, 0.0), original_width)
    y2 = min(max(y2, 0.0), original_height)

    return BBoxOriginal(x1=x1, y1=y1, x2=x2, y2=y2)


# =============================================================================
# Abstraccion del motor de inferencia (Singleton por proceso)
# =============================================================================

class InferenceEngine(abc.ABC):
    """Interfaz abstracta de motor de inferencia. El DynamicBatcher trabaja
    unicamente contra esta abstraccion, permitiendo intercambiar el backend
    (Ultralytics/PyTorch, ONNX Runtime, TensorRT) sin modificar el resto del
    servicio."""

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    @abc.abstractmethod
    def model_info(self) -> ModelInfo:
        raise NotImplementedError

    @abc.abstractmethod
    def load(self) -> None:
        """Carga el modelo en memoria/GPU. Se invoca una unica vez
        (Singleton) al iniciar el servicio."""
        raise NotImplementedError

    @abc.abstractmethod
    def infer_batch(self, batch_images: List[np.ndarray]) -> List[List[RawDetection]]:
        """Ejecuta inferencia sobre un batch de imagenes (todas del mismo
        tamaño, ya preprocesadas). Retorna una lista de detecciones por
        imagen, en el mismo orden de entrada, con NMS y umbrales de
        confianza/IoU ya aplicados por el motor."""
        raise NotImplementedError


class UltralyticsEngine(InferenceEngine):
    """Motor de inferencia activo, basado en Ultralytics (PyTorch)."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self._model = None
        self._model_name = os.path.basename(config.model_path)
        self._model_version = "unknown"

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(
            name=self._model_name,
            version=self._model_version,
            backend=ModelBackend.ULTRALYTICS.value,
            device=self._config.device.value,
            precision=self._config.precision.value,
        )

    def load(self) -> None:
        self._config.validate_model_path()

        try:
            from ultralytics import YOLO  # import diferido: dependencia pesada
        except ImportError as exc:
            raise ModelLoadError(
                "El paquete 'ultralytics' no esta instalado en el entorno"
            ) from exc

        try:
            model = YOLO(self._config.model_path)
        except Exception as exc:
            raise ModelLoadError(f"Fallo al cargar el modelo: {exc}") from exc

        try:
            model.to(self._config.device.value)
        except Exception as exc:
            raise ModelLoadError(
                f"Fallo al mover el modelo al device '{self._config.device.value}': {exc}"
            ) from exc

        self._model = model
        self._model_version = getattr(model, "ckpt_path", None) or self._model_version
        self._loaded = True

        logger.info(
            "Modelo YOLO cargado | backend=%s device=%s precision=%s model=%s",
            ModelBackend.ULTRALYTICS.value,
            self._config.device.value,
            self._config.precision.value,
            self._model_name,
        )

    def infer_batch(self, batch_images: List[np.ndarray]) -> List[List[RawDetection]]:
        if not self._loaded or self._model is None:
            raise ModelLoadError("El motor de inferencia no ha sido cargado (load() no invocado)")

        classes = list(self._config.class_whitelist) if self._config.class_whitelist else None
        half = self._config.precision == Precision.FP16 and self._config.device == Device.CUDA

        try:
            results = self._model.predict(
                source=batch_images,
                conf=self._config.confidence_threshold,
                iou=self._config.iou_threshold,
                classes=classes,
                device=self._config.device.value,
                half=half,
                verbose=False,
            )
        except RuntimeError as exc:
            if _is_cuda_oom_error(exc):
                raise InferenceOOMError(str(exc)) from exc
            raise

        batch_detections: List[List[RawDetection]] = []
        for result in results:
            frame_detections: List[RawDetection] = []
            boxes = getattr(result, "boxes", None)
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.tolist()
                confs = boxes.conf.tolist()
                cls_ids = boxes.cls.tolist()
                names = result.names if hasattr(result, "names") else {}
                for coords, conf, cls_id in zip(xyxy, confs, cls_ids):
                    class_id = int(cls_id)
                    frame_detections.append(
                        RawDetection(
                            class_id=class_id,
                            class_name=str(names.get(class_id, class_id)),
                            confidence=float(conf),
                            bbox_xyxy=(coords[0], coords[1], coords[2], coords[3]),
                        )
                    )
            batch_detections.append(frame_detections)

        return batch_detections


def _is_cuda_oom_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda oom" in message or "cublas_status_alloc_failed" in message


# -----------------------------------------------------------------------------
# NOTA ARQUITECTONICA:
# Los siguientes motores son STUBS preparados para futuros backends de
# inferencia optimizados. NO estan activos ni se instancian por defecto.
# Cuando se habiliten, deberan implementar load()/infer_batch()/model_info
# respetando el contrato de InferenceEngine, sin requerir cambios en
# DynamicBatcher ni en YoloDetectionService.
# -----------------------------------------------------------------------------

class ONNXRuntimeEngine(InferenceEngine):
    """[STUB] Motor de inferencia via ONNX Runtime.

    Pendiente de implementacion. Uso previsto:
        - session = onnxruntime.InferenceSession(model_path, providers=[...])
        - IOBinding para minimizar copias host/device en GPU.
        - Ejecutar session.run(...) sobre el batch normalizado (NCHW, float32/16).
        - Aplicar NMS (nativo del grafo ONNX o post-proceso vectorizado) y
          mapear las salidas al mismo formato List[List[RawDetection]].
    """

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(
            name=os.path.basename(self._config.model_path),
            version="unknown",
            backend=ModelBackend.ONNX_RUNTIME.value,
            device=self._config.device.value,
            precision=self._config.precision.value,
        )

    def load(self) -> None:
        raise NotImplementedError("ONNXRuntimeEngine aun no esta implementado.")

    def infer_batch(self, batch_images: List[np.ndarray]) -> List[List[RawDetection]]:
        raise NotImplementedError("ONNXRuntimeEngine aun no esta implementado.")


class TensorRTEngine(InferenceEngine):
    """[STUB] Motor de inferencia via NVIDIA TensorRT.

    Pendiente de implementacion. Uso previsto:
        - Cargar un engine .plan/.trt serializado (previamente compilado
          desde ONNX con trtexec o el builder de TensorRT).
        - Gestionar contexto de ejecucion, bindings de entrada/salida y
          buffers de GPU (pycuda o torch tensors) de forma explicita.
        - Ejecutar context.execute_async_v2(...) sobre el batch.
        - Mapear las salidas al mismo formato List[List[RawDetection]].
    """

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(
            name=os.path.basename(self._config.model_path),
            version="unknown",
            backend=ModelBackend.TENSORRT.value,
            device=self._config.device.value,
            precision=self._config.precision.value,
        )

    def load(self) -> None:
        raise NotImplementedError("TensorRTEngine aun no esta implementado.")

    def infer_batch(self, batch_images: List[np.ndarray]) -> List[List[RawDetection]]:
        raise NotImplementedError("TensorRTEngine aun no esta implementado.")


_ENGINE_REGISTRY = {
    ModelBackend.ULTRALYTICS: UltralyticsEngine,
    ModelBackend.ONNX_RUNTIME: ONNXRuntimeEngine,
    ModelBackend.TENSORRT: TensorRTEngine,
}

_engine_singleton: Optional[InferenceEngine] = None
_engine_singleton_lock = threading.Lock()


def get_inference_engine(config: ModelConfig) -> InferenceEngine:
    """Fabrica Singleton: el modelo se carga una unica vez por proceso,
    incluso si multiples hilos solicitan el motor concurrentemente."""
    global _engine_singleton
    if _engine_singleton is not None:
        return _engine_singleton

    with _engine_singleton_lock:
        if _engine_singleton is None:
            engine_cls = _ENGINE_REGISTRY[config.backend]
            engine = engine_cls(config)
            engine.load()
            _engine_singleton = engine

    return _engine_singleton


# =============================================================================
# Buffer con politica "descartar los mas antiguos" (tiempo real, RAM acotada)
# =============================================================================

class BoundedQueue:
    """Cola de tamaño maximo estricto. Si se llena, descarta el elemento mas
    antiguo para priorizar el mas reciente (tiempo real)."""

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
# Metricas / Health check
# =============================================================================

@dataclass
class DetectionMetrics:
    """Contadores globales del servicio de deteccion."""

    frames_received: int = 0
    frames_processed: int = 0
    frames_dropped_invalid: int = 0
    frames_dropped_backpressure: int = 0
    batches_processed: int = 0
    oom_events: int = 0
    total_inference_time_seconds: float = 0.0
    total_detections: int = 0
    last_result_timestamp: Optional[float] = None

    _fps_window_count: int = field(default=0, repr=False)
    _fps_window_start: float = field(default_factory=time.time, repr=False)
    measured_inference_fps: float = 0.0

    def register_processed_frame(self) -> None:
        self._fps_window_count += 1
        elapsed = time.time() - self._fps_window_start
        if elapsed >= 1.0:
            self.measured_inference_fps = round(self._fps_window_count / elapsed, 2)
            self._fps_window_count = 0
            self._fps_window_start = time.time()

    @property
    def avg_batch_latency_ms(self) -> float:
        if self.batches_processed == 0:
            return 0.0
        return round((self.total_inference_time_seconds / self.batches_processed) * 1000.0, 3)

    @property
    def avg_detections_per_frame(self) -> float:
        if self.frames_processed == 0:
            return 0.0
        return round(self.total_detections / self.frames_processed, 3)

    def to_dict(self) -> Dict[str, object]:
        return {
            "frames_received": self.frames_received,
            "frames_processed": self.frames_processed,
            "frames_dropped_invalid": self.frames_dropped_invalid,
            "frames_dropped_backpressure": self.frames_dropped_backpressure,
            "batches_processed": self.batches_processed,
            "oom_events": self.oom_events,
            "avg_batch_latency_ms": self.avg_batch_latency_ms,
            "avg_detections_per_frame": self.avg_detections_per_frame,
            "measured_inference_fps": self.measured_inference_fps,
            "last_result_timestamp": self.last_result_timestamp,
        }


def _gpu_health_snapshot(device: Device) -> Dict[str, object]:
    """Snapshot best-effort del estado de GPU/VRAM. Nunca lanza excepcion:
    si torch/CUDA no estan disponibles, reporta estado 'unavailable'."""
    if device != Device.CUDA:
        return {"gpu_available": False, "reason": "device configurado como CPU"}

    try:
        import torch  # import diferido
    except ImportError:
        return {"gpu_available": False, "reason": "torch no instalado"}

    if not torch.cuda.is_available():
        return {"gpu_available": False, "reason": "CUDA no disponible en este host"}

    try:
        index = torch.cuda.current_device()
        return {
            "gpu_available": True,
            "device_name": torch.cuda.get_device_name(index),
            "memory_allocated_mb": round(torch.cuda.memory_allocated(index) / (1024 ** 2), 2),
            "memory_reserved_mb": round(torch.cuda.memory_reserved(index) / (1024 ** 2), 2),
        }
    except Exception as exc:  # defensivo: health check nunca debe fallar
        return {"gpu_available": True, "error": str(exc)}


# =============================================================================
# Solicitud pendiente encolada para el Dynamic Batcher
# =============================================================================

@dataclass
class PendingRequest:
    """Unidad de trabajo interna: un frame preprocesado a la espera de ser
    agrupado en un batch."""

    camera_id: str
    frame_id: int
    session_id: str
    timestamp: float
    image: np.ndarray
    transformation_metadata: Dict[str, object]


# =============================================================================
# Dynamic Batcher: agrupa frames de multiples camaras en un solo batch GPU
# =============================================================================

class DynamicBatcher:
    """Agrupa frames entrantes de distintas camaras en un unico batch antes
    de enviarlos al motor de inferencia, maximizando el throughput de GPU.

    Politica de agrupamiento: se cierra el batch al alcanzar
    `max_batch_size` o al vencer `batch_timeout_ms` desde el primer frame
    recibido, lo que ocurra primero.

    Garantiza que los resultados se despachen al camera_id correcto:
    el orden de salida del motor de inferencia se preserva 1:1 con el
    orden de entrada del batch.
    """

    def __init__(
        self,
        engine: InferenceEngine,
        config: ModelConfig,
        metrics: DetectionMetrics,
    ) -> None:
        self._engine = engine
        self._config = config
        self._metrics = metrics

        self._input_queue = BoundedQueue(
            max_size=config.input_queue_size,
            on_drop=self._on_input_drop,
        )

        self._output_queues: Dict[str, BoundedQueue] = {}
        self._output_queues_lock = threading.Lock()

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -------------------------------------------------------------------
    # Ciclo de vida
    # -------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            logger.warning("DynamicBatcher ya en ejecucion")
            return
        logger.info(
            "Iniciando DynamicBatcher | max_batch_size=%d batch_timeout_ms=%.1f",
            self._config.max_batch_size,
            self._config.batch_timeout_ms,
        )
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="DynamicBatcher", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        logger.info("Deteniendo DynamicBatcher...")
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("DynamicBatcher no finalizo dentro del timeout de %.1fs", timeout)
        self._input_queue.clear()
        with self._output_queues_lock:
            for queue in self._output_queues.values():
                queue.clear()
        logger.info("DynamicBatcher detenido")

    # -------------------------------------------------------------------
    # Entrada / salida
    # -------------------------------------------------------------------

    def submit(self, request: PendingRequest) -> None:
        self._ensure_output_queue(request.camera_id)
        self._input_queue.put(request)

    def get_next_result(self, camera_id: str, timeout: float = 1.0) -> Optional[Dict[str, object]]:
        queue = self._ensure_output_queue(camera_id)
        return queue.get(timeout=timeout)

    def _ensure_output_queue(self, camera_id: str) -> BoundedQueue:
        with self._output_queues_lock:
            queue = self._output_queues.get(camera_id)
            if queue is None:
                queue = BoundedQueue(
                    max_size=self._config.output_queue_size_per_camera,
                    on_drop=lambda cam=camera_id: self._on_output_drop(cam),
                )
                self._output_queues[camera_id] = queue
            return queue

    # -------------------------------------------------------------------
    # Health / stats
    # -------------------------------------------------------------------

    def input_queue_usage(self) -> str:
        return f"{self._input_queue.qsize()}/{self._input_queue.max_size}"

    def output_queue_usage(self) -> Dict[str, str]:
        with self._output_queues_lock:
            return {
                camera_id: f"{queue.qsize()}/{queue.max_size}"
                for camera_id, queue in self._output_queues.items()
            }

    # -------------------------------------------------------------------
    # Loop principal
    # -------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.is_set():
            batch = self._collect_batch()
            if not batch:
                continue
            self._process_batch(batch)

    def _collect_batch(self) -> List[PendingRequest]:
        """Recolecta hasta max_batch_size requests, esperando como maximo
        batch_timeout_ms desde la llegada del primer elemento del batch."""
        first = self._input_queue.get(timeout=0.5)
        if first is None:
            return []

        batch: List[PendingRequest] = [first]  # type: ignore[list-item]
        deadline = time.time() + (self._config.batch_timeout_ms / 1000.0)

        while len(batch) < self._config.max_batch_size:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            item = self._input_queue.get(timeout=remaining)
            if item is None:
                break
            batch.append(item)  # type: ignore[arg-type]

        return batch

    def _process_batch(self, batch: List[PendingRequest]) -> None:
        images = [request.image for request in batch]

        start = time.time()
        try:
            raw_results = self._engine.infer_batch(images)
        except InferenceOOMError as exc:
            self._handle_oom(exc, batch)
            return
        except Exception:
            # Un fallo de inferencia no debe crashear el servicio; se
            # descarta el batch completo y se continua operando.
            self._metrics.frames_dropped_invalid += len(batch)
            logger.exception(
                "Error no controlado durante la inferencia, batch descartado (size=%d)",
                len(batch),
            )
            return

        inference_time_ms = (time.time() - start) * 1000.0
        self._metrics.batches_processed += 1
        self._metrics.total_inference_time_seconds += (time.time() - start)

        if len(raw_results) != len(batch):
            logger.error(
                "Discrepancia entre tamaño de batch (%d) y resultados del motor (%d); "
                "batch descartado para evitar mezclar resultados entre camaras",
                len(batch),
                len(raw_results),
            )
            self._metrics.frames_dropped_invalid += len(batch)
            return

        model_info = self._engine.model_info

        # Zip preserva el orden 1:1 entre la solicitud original y su
        # resultado correspondiente: garantiza que no se mezclen
        # detecciones entre camaras distintas.
        for request, frame_raw_detections in zip(batch, raw_results):
            self._dispatch_result(request, frame_raw_detections, model_info, inference_time_ms)

    def _dispatch_result(
        self,
        request: PendingRequest,
        raw_detections: List[RawDetection],
        model_info: ModelInfo,
        inference_time_ms: float,
    ) -> None:
        detections: List[Detection] = []
        for raw in raw_detections:
            # Filtrado defensivo adicional por umbral/whitelist (el motor ya
            # deberia aplicarlo, pero se refuerza aqui por seguridad).
            if raw.confidence < self._config.confidence_threshold:
                continue
            if self._config.class_whitelist and raw.class_id not in self._config.class_whitelist:
                continue

            try:
                bbox_original = reproject_bbox(raw.bbox_xyxy, request.transformation_metadata)
            except InputValidationError as exc:
                logger.warning(
                    "[%s] Deteccion descartada por error de proyeccion de coordenadas: %s",
                    request.camera_id,
                    exc,
                )
                continue

            detections.append(
                Detection(
                    detection_id=str(uuid.uuid4()),
                    class_id=raw.class_id,
                    class_name=raw.class_name,
                    confidence=raw.confidence,  # intacta, sin reglas de negocio
                    bbox_original_coords=bbox_original,
                )
            )

        result = DetectionResult(
            camera_id=request.camera_id,
            session_id=request.session_id,
            frame_id=request.frame_id,
            timestamp=request.timestamp,
            model_info=model_info,
            inference_time_ms=round(inference_time_ms, 3),
            detections=detections,
        )

        self._metrics.frames_processed += 1
        self._metrics.total_detections += len(detections)
        self._metrics.last_result_timestamp = request.timestamp
        self._metrics.register_processed_frame()

        output_queue = self._ensure_output_queue(request.camera_id)
        output_queue.put(result.to_dict())

    def _handle_oom(self, exc: InferenceOOMError, batch: List[PendingRequest]) -> None:
        """Prevencion de VRAM OOM: se vacia la cache de GPU, se limpian las
        colas para liberar presion de memoria y se reporta, pero el
        servicio continua activo."""
        self._metrics.oom_events += 1
        self._metrics.frames_dropped_invalid += len(batch)

        logger.critical(
            "CUDA Out-Of-Memory durante inferencia (batch_size=%d). Vaciando cache de GPU "
            "y limpiando colas para recuperar el servicio. error=%s",
            len(batch),
            str(exc),
        )

        try:
            import torch  # import diferido

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        except Exception:
            logger.exception("Error adicional al intentar liberar cache de GPU tras OOM")

        # Se limpia la cola de entrada para aliviar presion de memoria; los
        # frames en vuelo se pierden deliberadamente (prioridad: tiempo real
        # y estabilidad del servicio, no backlog historico).
        self._input_queue.clear()

    def _on_input_drop(self) -> None:
        self._metrics.frames_dropped_backpressure += 1
        _rate_limited_logger.log(
            "input_backpressure",
            logging.WARNING,
            "Cola de entrada de inferencia llena: frames mas antiguos descartados "
            "(%d en la ultima ventana)",
        )

    def _on_output_drop(self, camera_id: str) -> None:
        self._metrics.frames_dropped_backpressure += 1
        _rate_limited_logger.log(
            f"output_backpressure_{camera_id}",
            logging.WARNING,
            f"[{camera_id}] Cola de salida de resultados llena (consumidor lento): "
            f"resultados mas antiguos descartados (%d en la ultima ventana)",
        )


# =============================================================================
# Servicio principal
# =============================================================================

class YoloDetectionService:
    """Punto de entrada del Servicio 3. Recibe frames preprocesados de
    multiples camaras de forma concurrente y expone los DetectionResult
    correspondientes por camara."""

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        self.config = config or ModelConfig.from_env()
        self.metrics = DetectionMetrics()

        self._engine = get_inference_engine(self.config)
        self._batcher = DynamicBatcher(engine=self._engine, config=self.config, metrics=self.metrics)
        self._start_timestamp = time.time()

    def start(self) -> None:
        logger.info(
            "Iniciando YoloDetectionService | backend=%s model=%s device=%s precision=%s",
            self.config.backend.value,
            os.path.basename(self.config.model_path),
            self.config.device.value,
            self.config.precision.value,
        )
        self._batcher.start()

    def stop(self, timeout: float = 5.0) -> None:
        logger.info("Deteniendo YoloDetectionService...")
        self._batcher.stop(timeout=timeout)
        logger.info("YoloDetectionService detenido correctamente")

    # -------------------------------------------------------------------
    # Contrato de entrada / salida
    # -------------------------------------------------------------------

    def submit_frame(self, preprocessed_payload: Dict[str, object]) -> None:
        """Punto de entrada para frames provenientes de Preprocessing."""
        self.metrics.frames_received += 1

        try:
            validate_preprocessed_input(preprocessed_payload)
        except InputValidationError as exc:
            self.metrics.frames_dropped_invalid += 1
            _rate_limited_logger.log(
                "invalid_preprocessed_input",
                logging.WARNING,
                f"Frames preprocesados invalidos descartados (%d en la ultima ventana): {exc}",
            )
            return

        request = PendingRequest(
            camera_id=preprocessed_payload["camera_id"],  # type: ignore[arg-type]
            frame_id=preprocessed_payload["frame_id"],  # type: ignore[arg-type]
            session_id=preprocessed_payload["session_id"],  # type: ignore[arg-type]
            timestamp=preprocessed_payload["timestamp"],  # type: ignore[arg-type]
            image=preprocessed_payload["image"],  # type: ignore[arg-type]
            transformation_metadata=preprocessed_payload["transformation_metadata"],  # type: ignore[arg-type]
        )
        self._batcher.submit(request)

    def get_next_result(self, camera_id: str, timeout: float = 1.0) -> Optional[Dict[str, object]]:
        """Entrega el siguiente DetectionResult disponible para una camara
        especifica, o None si no hay ninguno dentro del timeout."""
        return self._batcher.get_next_result(camera_id, timeout=timeout)

    def result_stream(self, camera_id: str) -> Generator[Dict[str, object], None, None]:
        while True:
            result = self.get_next_result(camera_id, timeout=1.0)
            if result is not None:
                yield result

    # -------------------------------------------------------------------
    # Health check
    # -------------------------------------------------------------------

    def health_check(self) -> Dict[str, object]:
        return {
            "service_active": True,
            "model_info": self._engine.model_info.to_dict(),
            "model_loaded": self._engine.is_loaded,
            "gpu": _gpu_health_snapshot(self.config.device),
            "input_queue_usage": self._batcher.input_queue_usage(),
            "output_queue_usage": self._batcher.output_queue_usage(),
            "uptime_seconds": round(time.time() - self._start_timestamp, 3),
            "metrics": self.metrics.to_dict(),
        }


# =============================================================================
# Ejemplo de uso manual (smoke test) - motor sintetico, sin pesos reales
# =============================================================================

class _SyntheticDemoEngine(InferenceEngine):
    """Motor de demostracion SOLO para validar el cableado del
    DynamicBatcher/reproyeccion de coordenadas sin depender de pesos de
    modelo reales ni GPU. No debe usarse en produccion."""

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(
            name="synthetic-demo",
            version="0.0.0",
            backend="SYNTHETIC_DEMO",
            device=self._config.device.value,
            precision=self._config.precision.value,
        )

    def load(self) -> None:
        self._loaded = True

    def infer_batch(self, batch_images: List[np.ndarray]) -> List[List[RawDetection]]:
        results: List[List[RawDetection]] = []
        for image in batch_images:
            height, width = image.shape[0], image.shape[1]
            results.append(
                [
                    RawDetection(
                        class_id=0,
                        class_name="person",
                        confidence=0.87,
                        bbox_xyxy=(width * 0.25, height * 0.25, width * 0.75, height * 0.75),
                    )
                ]
            )
        return results


if __name__ == "__main__":
    demo_config = ModelConfig(
        backend=ModelBackend.ULTRALYTICS,
        model_path="models/yolo.pt",
        device=Device.CPU,
        precision=Precision.FP32,
        confidence_threshold=0.25,
        iou_threshold=0.45,
        class_whitelist=None,
        max_batch_size=8,
        batch_timeout_ms=10,
        input_queue_size=64,
        output_queue_size_per_camera=30,
    )

    with _engine_singleton_lock:
        _engine_singleton = _SyntheticDemoEngine(demo_config)
        _engine_singleton.load()

    service = YoloDetectionService(config=demo_config)
    service.start()

    synthetic_payload = {
        "camera_id": "CAM-001",
        "frame_id": 1,
        "session_id": "session-demo",
        "timestamp": time.time(),
        "image": np.zeros((640, 640, 3), dtype=np.uint8),
        "original_metadata": {},
        "transformation_metadata": {
            "original_width": 1920,
            "original_height": 1080,
            "processed_width": 640,
            "processed_height": 640,
            "scale_x": 0.3333333333333333,
            "scale_y": 0.3333333333333333,
            "pad_x": 0,
            "pad_y": 140,
            "roi_applied": None,
        },
    }

    service.submit_frame(synthetic_payload)

    result = service.get_next_result("CAM-001", timeout=2.0)
    if result is not None:
        logger.info("DetectionResult recibido: %s", result)

    logger.info("Health check: %s", service.health_check())
    service.stop()
