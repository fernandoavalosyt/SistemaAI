"""
poseService.py

Servicio de Pose / Specialized AI (Servicio 5 del pipeline).

Responsabilidad UNICA: recibir TrackingResult (bboxes + track_id) desde
"Tracking" (Servicio 4), recortar (crop) cada persona del frame original
(enfoque Top-Down), ejecutar un modelo especializado de pose sobre los
recortes, proyectar los keypoints de vuelta a coordenadas globales del frame
y calcular señales geometricas derivadas (angulos articulares, inclinacion de
torso, orientacion de cabeza) para el "Behavior Engine" (Servicio 6).

POSE OBSERVA, NO INTERPRETA: el output contiene exclusivamente valores
numericos (ej. torso_inclination_deg: 80.5). Nunca etiquetas de
comportamiento (is_sleeping, is_stealing, etc.). Este servicio NO crea
track_ids: la identidad pertenece al Tracking Service.
"""

from __future__ import annotations

import abc
import logging
import math
import os
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, FrozenSet, Generator, List, Optional, Set, Tuple

import numpy as np


# =============================================================================
# Logging
# =============================================================================
# Nunca se loguean imagenes, crops ni arrays; solo ids, conteos y tiempos.

logger = logging.getLogger("pose_service")
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


class _RateLimitedLogger:
    """Agrupa eventos de alta frecuencia y emite una linea por ventana."""

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
            if now - self._last_emit.get(key, 0.0) >= self._interval:
                self._logger.log(level, message_template, self._counts[key])
                self._counts[key] = 0
                self._last_emit[key] = now


_rate_limited_logger = _RateLimitedLogger(logger)


# =============================================================================
# Excepciones
# =============================================================================

class InputValidationError(Exception):
    """TrackingResult de entrada invalido."""


class ModelLoadError(Exception):
    """Fallo al cargar el modelo especializado."""


class InferenceOOMError(Exception):
    """Memoria de GPU agotada durante la inferencia. Recuperable."""


class InvalidCropError(Exception):
    """El bbox del track no produce un recorte valido."""


# =============================================================================
# Enums y constantes anatomicas
# =============================================================================

class PoseBackend(str, Enum):
    YOLO_POSE = "YOLO_POSE"      # GPU, soporta batching real.
    MEDIAPIPE = "MEDIAPIPE"      # CPU, un crop a la vez.


class Device(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"


class Precision(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"


class PoseStatus(str, Enum):
    OK = "OK"                  # Pose extraida.
    NO_POSE = "NO_POSE"        # El modelo no encontro esqueleto en el crop.
    FAILED = "FAILED"          # Error (crop invalido, fallo del modelo, OOM).


# Esquema COCO-17: orden de salida de YOLOv8-Pose.
COCO_KEYPOINTS: Tuple[str, ...] = (
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
)

# Indices de MediaPipe Pose (33 landmarks) mapeados al esquema COCO-17,
# para que ambos backends produzcan exactamente el mismo contrato.
_MEDIAPIPE_TO_COCO: Tuple[int, ...] = (0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)

_TRACKABLE_STATES: FrozenSet[str] = frozenset({"ACTIVE", "NEW"})


# =============================================================================
# Configuracion (variables de entorno)
# =============================================================================

@dataclass(frozen=True)
class PoseConfig:
    backend: PoseBackend = PoseBackend.YOLO_POSE
    model_path: str = "models/yolov8n-pose.pt"
    device: Device = Device.CPU
    precision: Precision = Precision.FP32
    keypoint_conf_threshold: float = 0.3
    target_classes: FrozenSet[str] = frozenset({"person"})
    max_batch_crops: int = 16
    batch_timeout_ms: float = 10.0
    crop_margin_ratio: float = 0.1
    min_crop_size_px: int = 16
    input_queue_size: int = 32
    output_queue_size_per_camera: int = 30
    frame_cache_size_per_camera: int = 60

    @staticmethod
    def from_env() -> "PoseConfig":
        classes_raw = os.getenv("POSE_TARGET_CLASSES", "person")
        return PoseConfig(
            backend=PoseBackend(os.getenv("POSE_BACKEND", PoseBackend.YOLO_POSE.value)),
            model_path=os.getenv("POSE_MODEL_PATH", "models/yolov8n-pose.pt"),
            device=Device(os.getenv("POSE_DEVICE", Device.CPU.value)),
            precision=Precision(os.getenv("POSE_PRECISION", Precision.FP32.value)),
            keypoint_conf_threshold=float(os.getenv("POSE_KPT_CONF_THRESHOLD", "0.3")),
            target_classes=frozenset(c.strip() for c in classes_raw.split(",") if c.strip()),
            max_batch_crops=int(os.getenv("POSE_MAX_BATCH_CROPS", "16")),
            batch_timeout_ms=float(os.getenv("POSE_BATCH_TIMEOUT_MS", "10")),
            crop_margin_ratio=float(os.getenv("POSE_CROP_MARGIN", "0.1")),
            min_crop_size_px=int(os.getenv("POSE_MIN_CROP_SIZE", "16")),
            input_queue_size=int(os.getenv("POSE_INPUT_QUEUE_SIZE", "32")),
            output_queue_size_per_camera=int(os.getenv("POSE_OUTPUT_QUEUE_SIZE", "30")),
            frame_cache_size_per_camera=int(os.getenv("POSE_FRAME_CACHE_SIZE", "60")),
        )

    def validate_model_path(self) -> None:
        # MediaPipe trae su modelo empaquetado; solo YOLO-Pose requiere archivo.
        if self.backend == PoseBackend.YOLO_POSE and not os.path.isfile(self.model_path):
            raise ModelLoadError(f"No se encontro el archivo de modelo en: {self.model_path}")


@dataclass(frozen=True)
class ModelInfo:
    model_name: str
    version: str
    backend: str
    device: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "model_name": self.model_name,
            "version": self.version,
            "backend": self.backend,
            "device": self.device,
        }


# =============================================================================
# Contratos de datos
# =============================================================================

@dataclass(frozen=True)
class CropRegion:
    """Recorte efectivo aplicado al frame. (x_min, y_min) es el offset que
    se suma a los keypoints para volver a coordenadas globales."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int


@dataclass
class RawPose:
    """Salida cruda del modelo: coordenadas RELATIVAS AL CROP."""

    xy: np.ndarray           # (17, 2)
    confidence: np.ndarray   # (17,)


@dataclass
class Keypoint:
    x: Optional[float]
    y: Optional[float]
    confidence: float
    visible: bool

    def to_dict(self) -> Dict[str, object]:
        return {
            "x": None if self.x is None else round(self.x, 2),
            "y": None if self.y is None else round(self.y, 2),
            "confidence": round(self.confidence, 4),
            "visible": self.visible,
        }


@dataclass
class PoseData:
    track_id: str
    class_name: str
    status: PoseStatus
    keypoints: Dict[str, Keypoint] = field(default_factory=dict)
    derived_signals: Dict[str, Optional[float]] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "status": self.status.value,
            "keypoints": {name: kp.to_dict() for name, kp in self.keypoints.items()},
            "derived_signals": self.derived_signals,
            "error": self.error,
        }


@dataclass
class SpecializedAIResult:
    """Contrato de salida hacia el Servicio 6 (Behavior Engine)."""

    camera_id: str
    session_id: str
    frame_id: int
    timestamp: float
    model_info: ModelInfo
    inference_time_ms: float
    poses: List[PoseData]

    def to_dict(self) -> Dict[str, object]:
        return {
            "camera_id": self.camera_id,
            "session_id": self.session_id,
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "model_info": {**self.model_info.to_dict(), "inference_time_ms": self.inference_time_ms},
            "poses": [p.to_dict() for p in self.poses],
        }


# =============================================================================
# Validacion del contrato de entrada (desde Tracking)
# =============================================================================

_REQUIRED_RESULT_KEYS = ("camera_id", "session_id", "frame_id", "timestamp", "tracks")
_REQUIRED_TRACK_KEYS = ("track_id", "class_name", "bbox", "track_state")
_REQUIRED_BBOX_KEYS = ("x1", "y1", "x2", "y2")


def validate_tracking_result(payload: Optional[Dict[str, object]]) -> None:
    if not payload or not isinstance(payload, dict):
        raise InputValidationError("Input nulo o vacio")
    for key in _REQUIRED_RESULT_KEYS:
        if key not in payload:
            raise InputValidationError(f"Falta campo obligatorio en TrackingResult: {key}")
    if not payload["camera_id"] or not payload["session_id"]:
        raise InputValidationError("camera_id/session_id invalido")
    timestamp = payload["timestamp"]
    if timestamp is None or timestamp <= 0:
        raise InputValidationError("timestamp invalido")
    if not isinstance(payload["tracks"], list):
        raise InputValidationError("El campo 'tracks' debe ser una lista")
    for track in payload["tracks"]:
        if not isinstance(track, dict):
            raise InputValidationError("Cada track debe ser un dict")
        for key in _REQUIRED_TRACK_KEYS:
            if key not in track:
                raise InputValidationError(f"Falta campo obligatorio en track: {key}")
        bbox = track["bbox"]
        if not isinstance(bbox, dict) or any(k not in bbox for k in _REQUIRED_BBOX_KEYS):
            raise InputValidationError("bbox de track invalido")


# =============================================================================
# Cache de frames (acceso al frame original por camera_id + frame_id)
# =============================================================================

class FrameCache:
    """Cache acotado por camara de frames originales. Video Ingestion (o un
    adaptador intermedio) deposita aqui el frame; Pose lo recupera al
    recibir el TrackingResult correspondiente. Los mas antiguos se expulsan
    automaticamente (RAM acotada)."""

    def __init__(self, max_frames_per_camera: int) -> None:
        self._max = max_frames_per_camera
        self._frames: Dict[str, "OrderedDict[int, np.ndarray]"] = {}
        self._lock = threading.Lock()

    def put(self, camera_id: str, frame_id: int, image: np.ndarray) -> None:
        with self._lock:
            per_camera = self._frames.setdefault(camera_id, OrderedDict())
            per_camera[frame_id] = image
            per_camera.move_to_end(frame_id)
            while len(per_camera) > self._max:
                per_camera.popitem(last=False)

    def get(self, camera_id: str, frame_id: int) -> Optional[np.ndarray]:
        with self._lock:
            per_camera = self._frames.get(camera_id)
            return None if per_camera is None else per_camera.get(frame_id)

    def evict_up_to(self, camera_id: str, frame_id: int) -> None:
        """Libera frames ya procesados (y anteriores) de una camara."""
        with self._lock:
            per_camera = self._frames.get(camera_id)
            if per_camera is None:
                return
            for fid in [f for f in per_camera if f <= frame_id]:
                del per_camera[fid]

    def usage(self) -> Dict[str, str]:
        with self._lock:
            return {cam: f"{len(frames)}/{self._max}" for cam, frames in self._frames.items()}

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()


# =============================================================================
# Recorte (Top-Down) y proyeccion de coordenadas
# =============================================================================

def compute_crop_region(
    bbox: Dict[str, float],
    frame_width: int,
    frame_height: int,
    margin_ratio: float,
    min_size: int,
) -> CropRegion:
    """Expande el bbox con un margen (evita cortar extremidades), lo recorta
    a los limites del frame y valida que el resultado sea utilizable."""
    x1, y1, x2, y2 = (float(bbox[k]) for k in ("x1", "y1", "x2", "y2"))
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        raise InvalidCropError("bbox con valores no finitos")

    width, height = x2 - x1, y2 - y1
    if width <= 0 or height <= 0:
        raise InvalidCropError(f"bbox degenerado: {width:.1f}x{height:.1f}")

    x_min = max(0, int(math.floor(x1 - width * margin_ratio)))
    y_min = max(0, int(math.floor(y1 - height * margin_ratio)))
    x_max = min(frame_width, int(math.ceil(x2 + width * margin_ratio)))
    y_max = min(frame_height, int(math.ceil(y2 + height * margin_ratio)))

    if x_max - x_min < min_size or y_max - y_min < min_size:
        raise InvalidCropError(
            f"crop fuera de limites o demasiado pequeño: {x_max - x_min}x{y_max - y_min}"
        )
    return CropRegion(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


def extract_crop(frame: np.ndarray, region: CropRegion) -> np.ndarray:
    return np.ascontiguousarray(frame[region.y_min:region.y_max, region.x_min:region.x_max])


def project_keypoints(
    raw_pose: RawPose,
    region: CropRegion,
    conf_threshold: float,
) -> Dict[str, Keypoint]:
    """Proyecta keypoints relativos al crop a coordenadas globales del frame
    sumando (x_min, y_min) del recorte. Los puntos bajo el umbral de
    confianza se marcan como ocluidos (x/y = None) para no propagar
    'keypoints fantasma' al Behavior Engine."""
    keypoints: Dict[str, Keypoint] = {}
    for idx, name in enumerate(COCO_KEYPOINTS):
        confidence = float(raw_pose.confidence[idx])
        local_x, local_y = float(raw_pose.xy[idx, 0]), float(raw_pose.xy[idx, 1])
        # Ultralytics devuelve (0,0) para puntos no localizados.
        located = math.isfinite(local_x) and math.isfinite(local_y) and not (local_x == 0.0 and local_y == 0.0)
        if confidence < conf_threshold or not located:
            keypoints[name] = Keypoint(x=None, y=None, confidence=confidence, visible=False)
            continue
        keypoints[name] = Keypoint(
            x=local_x + region.x_min,
            y=local_y + region.y_min,
            confidence=confidence,
            visible=True,
        )
    return keypoints


# =============================================================================
# Señales derivadas (geometria de postura) - solo valores numericos
# =============================================================================

def _point(keypoints: Dict[str, Keypoint], name: str) -> Optional[np.ndarray]:
    kp = keypoints.get(name)
    if kp is None or not kp.visible or kp.x is None or kp.y is None:
        return None
    return np.array([kp.x, kp.y], dtype=np.float64)


def _midpoint(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if a is not None and b is not None:
        return (a + b) / 2.0
    return a if a is not None else b


def joint_angle_deg(a: Optional[np.ndarray], vertex: Optional[np.ndarray], c: Optional[np.ndarray]) -> Optional[float]:
    """Angulo interior en 'vertex' formado por los segmentos vertex->a y
    vertex->c, via producto punto. 180 = extremidad extendida."""
    if a is None or vertex is None or c is None:
        return None
    v1, v2 = a - vertex, c - vertex
    norm = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if norm < 1e-9:
        return None
    cos_angle = float(np.clip(np.dot(v1, v2) / norm, -1.0, 1.0))
    return round(math.degrees(math.acos(cos_angle)), 2)


def torso_inclination_deg(keypoints: Dict[str, Keypoint]) -> Optional[float]:
    """Angulo con signo entre el vector cadera->hombros y la vertical de la
    imagen. 0 = erguido; + = inclinado hacia la derecha de la imagen;
    - = hacia la izquierda; |90| = horizontal. Se usa atan2 con el eje y
    invertido porque en imagen 'arriba' es y decreciente."""
    shoulders = _midpoint(_point(keypoints, "left_shoulder"), _point(keypoints, "right_shoulder"))
    hips = _midpoint(_point(keypoints, "left_hip"), _point(keypoints, "right_hip"))
    if shoulders is None or hips is None:
        return None
    dx, dy = shoulders - hips
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return None
    return round(math.degrees(math.atan2(dx, -dy)), 2)


def head_orientation(keypoints: Dict[str, Keypoint]) -> Dict[str, Optional[float]]:
    """Proxy geometrico del giro de cabeza (yaw) a partir de nariz vs orejas.

    head_yaw_ratio: desplazamiento horizontal de la nariz respecto al punto
    medio de las orejas, normalizado por la distancia entre orejas.
    ~0 = de frente; tiende a +/-0.5 o mas al girar. Con una sola oreja
    visible (perfil) se reporta +/-1.0 segun el lado visible.
    head_yaw_deg: estimacion angular derivada del ratio (arcsin acotado).
    """
    nose = _point(keypoints, "nose")
    left_ear = _point(keypoints, "left_ear")
    right_ear = _point(keypoints, "right_ear")
    visible_ears = int(left_ear is not None) + int(right_ear is not None)

    result: Dict[str, Optional[float]] = {
        "head_yaw_ratio": None,
        "head_yaw_deg": None,
        "head_visible_ears": float(visible_ears),
    }
    if nose is None:
        return result

    if left_ear is not None and right_ear is not None:
        ear_span = float(right_ear[0] - left_ear[0])
        if abs(ear_span) < 1e-6:
            return result
        mid_x = (left_ear[0] + right_ear[0]) / 2.0
        ratio = float((nose[0] - mid_x) / abs(ear_span))
    elif left_ear is not None or right_ear is not None:
        ear = left_ear if left_ear is not None else right_ear
        ratio = 1.0 if nose[0] > ear[0] else -1.0
    else:
        return result

    ratio = float(np.clip(ratio, -1.0, 1.0))
    result["head_yaw_ratio"] = round(ratio, 4)
    result["head_yaw_deg"] = round(math.degrees(math.asin(ratio)), 2)
    return result


def compute_derived_signals(keypoints: Dict[str, Keypoint]) -> Dict[str, Optional[float]]:
    """Señales geometricas por persona. None = no calculable por oclusion."""
    p = lambda name: _point(keypoints, name)  # noqa: E731

    signals: Dict[str, Optional[float]] = {
        "torso_inclination_deg": torso_inclination_deg(keypoints),
        "left_elbow_angle_deg": joint_angle_deg(p("left_shoulder"), p("left_elbow"), p("left_wrist")),
        "right_elbow_angle_deg": joint_angle_deg(p("right_shoulder"), p("right_elbow"), p("right_wrist")),
        "left_shoulder_angle_deg": joint_angle_deg(p("left_hip"), p("left_shoulder"), p("left_elbow")),
        "right_shoulder_angle_deg": joint_angle_deg(p("right_hip"), p("right_shoulder"), p("right_elbow")),
        "left_knee_angle_deg": joint_angle_deg(p("left_hip"), p("left_knee"), p("left_ankle")),
        "right_knee_angle_deg": joint_angle_deg(p("right_hip"), p("right_knee"), p("right_ankle")),
        "left_hip_angle_deg": joint_angle_deg(p("left_shoulder"), p("left_hip"), p("left_knee")),
        "right_hip_angle_deg": joint_angle_deg(p("right_shoulder"), p("right_hip"), p("right_knee")),
    }
    signals.update(head_orientation(keypoints))

    visible = sum(1 for kp in keypoints.values() if kp.visible)
    signals["visible_keypoints"] = float(visible)
    signals["visible_keypoints_ratio"] = round(visible / len(COCO_KEYPOINTS), 4)
    return signals


# =============================================================================
# Abstraccion del modelo especializado
# =============================================================================

class SpecializedModelInterface(abc.ABC):
    """Contrato de cualquier modelo de pose. Recibe crops (BGR uint8) y
    devuelve, en el mismo orden, un RawPose por crop (o None si no hay
    esqueleto). Coordenadas SIEMPRE relativas al crop."""

    supports_batching: bool = False

    def __init__(self, config: PoseConfig) -> None:
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
        raise NotImplementedError

    @abc.abstractmethod
    def infer_batch(self, crops: List[np.ndarray]) -> List[Optional[RawPose]]:
        raise NotImplementedError


def _is_cuda_oom_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cublas_status_alloc_failed" in message


class YoloPoseModel(SpecializedModelInterface):
    """Backend por defecto: YOLOv8-Pose (Ultralytics). Optimizado para GPU,
    acepta una lista de crops de tamaños distintos como un solo batch
    (Ultralytics aplica letterbox por imagen internamente)."""

    supports_batching = True

    def __init__(self, config: PoseConfig) -> None:
        super().__init__(config)
        self._model = None
        self._name = os.path.basename(config.model_path)
        self._version = "unknown"

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(
            model_name=self._name,
            version=self._version,
            backend=PoseBackend.YOLO_POSE.value,
            device=self._config.device.value,
        )

    def load(self) -> None:
        self._config.validate_model_path()
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ModelLoadError("El paquete 'ultralytics' no esta instalado") from exc
        try:
            model = YOLO(self._config.model_path)
            model.to(self._config.device.value)
        except Exception as exc:
            raise ModelLoadError(f"Fallo al cargar YOLO-Pose: {exc}") from exc

        task = getattr(model, "task", None)
        if task is not None and task != "pose":
            raise ModelLoadError(f"El modelo cargado no es de pose (task={task!r})")

        self._model = model
        try:
            import ultralytics
            self._version = f"ultralytics-{ultralytics.__version__}"
        except Exception:
            pass
        self._loaded = True
        logger.info(
            "Modelo de pose cargado | backend=YOLO_POSE model=%s device=%s precision=%s",
            self._name, self._config.device.value, self._config.precision.value,
        )

    def infer_batch(self, crops: List[np.ndarray]) -> List[Optional[RawPose]]:
        if self._model is None:
            raise ModelLoadError("YoloPoseModel no cargado")
        half = self._config.precision == Precision.FP16 and self._config.device == Device.CUDA
        try:
            results = self._model.predict(
                source=crops,
                device=self._config.device.value,
                half=half,
                verbose=False,
            )
        except RuntimeError as exc:
            if _is_cuda_oom_error(exc):
                raise InferenceOOMError(str(exc)) from exc
            raise

        poses: List[Optional[RawPose]] = []
        for result in results:
            poses.append(self._select_primary_pose(result))
        return poses

    @staticmethod
    def _select_primary_pose(result) -> Optional[RawPose]:
        """Un crop puede contener a mas de una persona (solapamientos). Se
        elige el esqueleto con mayor area de caja: la persona del track es
        la que domina el recorte."""
        keypoints = getattr(result, "keypoints", None)
        boxes = getattr(result, "boxes", None)
        if keypoints is None or boxes is None or len(boxes) == 0:
            return None

        xyxy = boxes.xyxy.cpu().numpy()
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        best = int(np.argmax(areas))

        xy = keypoints.xy.cpu().numpy()[best].astype(np.float64)
        if keypoints.conf is not None:
            conf = keypoints.conf.cpu().numpy()[best].astype(np.float64)
        else:
            conf = np.ones(len(COCO_KEYPOINTS), dtype=np.float64)

        if xy.shape[0] != len(COCO_KEYPOINTS):
            return None
        return RawPose(xy=xy, confidence=conf)


class MediaPipePoseModel(SpecializedModelInterface):
    """Backend alternativo: MediaPipe Pose. Ideal para CPU. No soporta
    batching: procesa los crops secuencialmente. Sus 33 landmarks se
    remapean al esquema COCO-17 para mantener el mismo contrato."""

    supports_batching = False

    def __init__(self, config: PoseConfig) -> None:
        super().__init__(config)
        self._pose = None
        self._lock = threading.Lock()  # la instancia de MediaPipe no es thread-safe

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(
            model_name="mediapipe-pose",
            version="solutions.pose",
            backend=PoseBackend.MEDIAPIPE.value,
            device=Device.CPU.value,
        )

    def load(self) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise ModelLoadError("El paquete 'mediapipe' no esta instalado") from exc
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=True,  # cada crop es independiente
            model_complexity=1,
            enable_segmentation=False,
        )
        self._loaded = True
        logger.info("Modelo de pose cargado | backend=MEDIAPIPE device=cpu")

    def infer_batch(self, crops: List[np.ndarray]) -> List[Optional[RawPose]]:
        if self._pose is None:
            raise ModelLoadError("MediaPipePoseModel no cargado")
        import cv2

        poses: List[Optional[RawPose]] = []
        with self._lock:
            for crop in crops:
                height, width = crop.shape[:2]
                result = self._pose.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                if result.pose_landmarks is None:
                    poses.append(None)
                    continue
                landmarks = result.pose_landmarks.landmark
                xy = np.array(
                    [[landmarks[i].x * width, landmarks[i].y * height] for i in _MEDIAPIPE_TO_COCO],
                    dtype=np.float64,
                )
                conf = np.array([landmarks[i].visibility for i in _MEDIAPIPE_TO_COCO], dtype=np.float64)
                poses.append(RawPose(xy=xy, confidence=conf))
        return poses


_MODEL_REGISTRY = {
    PoseBackend.YOLO_POSE: YoloPoseModel,
    PoseBackend.MEDIAPIPE: MediaPipePoseModel,
}

_model_singleton: Optional[SpecializedModelInterface] = None
_model_singleton_lock = threading.Lock()


def get_pose_model(config: PoseConfig) -> SpecializedModelInterface:
    """El modelo se carga una unica vez por proceso (Singleton thread-safe)."""
    global _model_singleton
    if _model_singleton is not None:
        return _model_singleton
    with _model_singleton_lock:
        if _model_singleton is None:
            model = _MODEL_REGISTRY[config.backend](config)
            model.load()
            _model_singleton = model
    return _model_singleton


# =============================================================================
# Cola acotada (descarta los mas antiguos)
# =============================================================================

class BoundedQueue:
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
# Metricas
# =============================================================================

@dataclass
class PoseMetrics:
    results_received: int = 0
    results_processed: int = 0
    results_dropped_invalid: int = 0
    results_dropped_stale: int = 0
    results_dropped_no_frame: int = 0
    results_dropped_backpressure: int = 0
    crops_inferred: int = 0
    crops_failed: int = 0
    crops_no_pose: int = 0
    batches_processed: int = 0
    oom_events: int = 0
    total_inference_seconds: float = 0.0

    _fps_count: int = field(default=0, repr=False)
    _fps_window_start: float = field(default_factory=time.time, repr=False)
    measured_fps: float = 0.0

    def register_processed(self) -> None:
        self._fps_count += 1
        elapsed = time.time() - self._fps_window_start
        if elapsed >= 1.0:
            self.measured_fps = round(self._fps_count / elapsed, 2)
            self._fps_count = 0
            self._fps_window_start = time.time()

    @property
    def avg_batch_latency_ms(self) -> float:
        if self.batches_processed == 0:
            return 0.0
        return round(self.total_inference_seconds / self.batches_processed * 1000.0, 3)

    def to_dict(self) -> Dict[str, object]:
        return {
            "results_received": self.results_received,
            "results_processed": self.results_processed,
            "results_dropped_invalid": self.results_dropped_invalid,
            "results_dropped_stale": self.results_dropped_stale,
            "results_dropped_no_frame": self.results_dropped_no_frame,
            "results_dropped_backpressure": self.results_dropped_backpressure,
            "crops_inferred": self.crops_inferred,
            "crops_failed": self.crops_failed,
            "crops_no_pose": self.crops_no_pose,
            "batches_processed": self.batches_processed,
            "oom_events": self.oom_events,
            "avg_batch_latency_ms": self.avg_batch_latency_ms,
            "measured_fps": self.measured_fps,
        }


# =============================================================================
# Unidades de trabajo internas
# =============================================================================

@dataclass
class _PendingFrame:
    """Un TrackingResult validado + su frame, listo para agruparse."""

    camera_id: str
    session_id: str
    frame_id: int
    timestamp: float
    frame: np.ndarray
    tracks: List[Dict[str, object]]


@dataclass
class _CropJob:
    """Un crop individual dentro de un batch, con su trazabilidad completa
    (a que frame y track pertenece) para no mezclar resultados."""

    frame_index: int
    track_id: str
    class_name: str
    region: CropRegion
    crop: np.ndarray


# =============================================================================
# Servicio principal
# =============================================================================

class PoseService:
    """Punto de entrada del Servicio 5.

    Flujo: submit_tracking_result() -> cola acotada -> batcher (hilo) que
    agrupa crops de varios frames/camaras -> inferencia en un solo batch ->
    proyeccion + señales derivadas -> cola de salida por camara.
    """

    def __init__(
        self,
        config: Optional[PoseConfig] = None,
        model: Optional[SpecializedModelInterface] = None,
    ) -> None:
        self.config = config or PoseConfig.from_env()
        self.metrics = PoseMetrics()
        self.frame_cache = FrameCache(self.config.frame_cache_size_per_camera)

        self._model = model if model is not None else get_pose_model(self.config)

        self._input_queue = BoundedQueue(self.config.input_queue_size, on_drop=self._on_input_drop)
        self._output_queues: Dict[str, BoundedQueue] = {}
        self._output_lock = threading.Lock()

        self._last_frame_id: Dict[str, int] = {}
        self._last_frame_lock = threading.Lock()

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_timestamp = time.time()

    # -------------------------------------------------------------------
    # Ciclo de vida
    # -------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        logger.info(
            "Iniciando PoseService | backend=%s batching=%s max_batch_crops=%d",
            self._model.model_info.backend,
            self._model.supports_batching,
            self.config.max_batch_crops,
        )
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="PoseBatcher", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        logger.info("Deteniendo PoseService...")
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._input_queue.clear()
        with self._output_lock:
            for queue in self._output_queues.values():
                queue.clear()
        self.frame_cache.clear()
        logger.info("PoseService detenido")

    # -------------------------------------------------------------------
    # Entrada
    # -------------------------------------------------------------------

    def cache_frame(self, camera_id: str, frame_id: int, image: np.ndarray) -> None:
        """Deposita el frame ORIGINAL (salida de Video Ingestion). Los bbox
        del Tracking estan en coordenadas originales, asi que el crop debe
        hacerse sobre este frame, no sobre el preprocesado."""
        self.frame_cache.put(camera_id, frame_id, image)

    def submit_tracking_result(
        self,
        payload: Dict[str, object],
        frame: Optional[np.ndarray] = None,
    ) -> None:
        """Recibe un TrackingResult. El frame puede pasarse directamente o
        recuperarse del FrameCache por (camera_id, frame_id)."""
        self.metrics.results_received += 1
        try:
            validate_tracking_result(payload)
        except InputValidationError as exc:
            self.metrics.results_dropped_invalid += 1
            _rate_limited_logger.log(
                "invalid_input", logging.WARNING,
                f"TrackingResult invalidos descartados (%d en la ultima ventana): {exc}",
            )
            return

        camera_id = str(payload["camera_id"])
        frame_id = int(payload["frame_id"])  # type: ignore[arg-type]

        if frame is None:
            frame = self.frame_cache.get(camera_id, frame_id)
        if frame is None or not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.size == 0:
            self.metrics.results_dropped_no_frame += 1
            _rate_limited_logger.log(
                f"no_frame_{camera_id}", logging.WARNING,
                f"[{camera_id}] TrackingResult sin frame disponible en cache, descartado "
                f"(%d en la ultima ventana)",
            )
            return

        tracks = [
            t for t in payload["tracks"]  # type: ignore[union-attr]
            if t["track_state"] in _TRACKABLE_STATES and t["class_name"] in self.config.target_classes
        ]

        self._ensure_output_queue(camera_id)
        self._input_queue.put(
            _PendingFrame(
                camera_id=camera_id,
                session_id=str(payload["session_id"]),
                frame_id=frame_id,
                timestamp=float(payload["timestamp"]),  # type: ignore[arg-type]
                frame=frame,
                tracks=tracks,
            )
        )

    # -------------------------------------------------------------------
    # Salida
    # -------------------------------------------------------------------

    def get_next_result(self, camera_id: str, timeout: float = 1.0) -> Optional[Dict[str, object]]:
        return self._ensure_output_queue(camera_id).get(timeout=timeout)  # type: ignore[return-value]

    def result_stream(self, camera_id: str) -> Generator[Dict[str, object], None, None]:
        while not self._stop_event.is_set():
            result = self.get_next_result(camera_id, timeout=1.0)
            if result is not None:
                yield result

    def health_check(self) -> Dict[str, object]:
        with self._output_lock:
            output_usage = {cam: f"{q.qsize()}/{q.max_size}" for cam, q in self._output_queues.items()}
        return {
            "service_active": self._thread is not None and self._thread.is_alive(),
            "model_info": self._model.model_info.to_dict(),
            "model_loaded": self._model.is_loaded,
            "batching_enabled": self._model.supports_batching,
            "active_cameras": list(output_usage.keys()),
            "input_queue_usage": f"{self._input_queue.qsize()}/{self._input_queue.max_size}",
            "output_queue_usage": output_usage,
            "frame_cache_usage": self.frame_cache.usage(),
            "uptime_seconds": round(time.time() - self._start_timestamp, 3),
            "metrics": self.metrics.to_dict(),
        }

    # -------------------------------------------------------------------
    # Loop de batching
    # -------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                frames = self._collect_frames()
                if frames:
                    self._process_frames(frames)
            except Exception:
                logger.exception("Error inesperado en el loop de PoseService (se continua)")

    def _collect_frames(self) -> List[_PendingFrame]:
        """Agrupa frames hasta acumular max_batch_crops crops o vencer el
        timeout desde el primer frame. Un frame nunca se parte entre batches."""
        first = self._input_queue.get(timeout=0.5)
        if first is None:
            return []
        frames: List[_PendingFrame] = [first]  # type: ignore[list-item]
        crop_count = len(first.tracks)  # type: ignore[attr-defined]

        if not self._model.supports_batching:
            return frames

        deadline = time.time() + self.config.batch_timeout_ms / 1000.0
        while crop_count < self.config.max_batch_crops:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            item = self._input_queue.get(timeout=remaining)
            if item is None:
                break
            frames.append(item)  # type: ignore[arg-type]
            crop_count += len(item.tracks)  # type: ignore[attr-defined]
        return frames

    def _process_frames(self, frames: List[_PendingFrame]) -> None:
        frames = [f for f in frames if self._accept_frame_order(f)]
        if not frames:
            return

        poses_by_frame: List[Dict[str, PoseData]] = [{} for _ in frames]
        jobs: List[_CropJob] = []

        for frame_index, pending in enumerate(frames):
            height, width = pending.frame.shape[:2]
            for track in pending.tracks:
                track_id = str(track["track_id"])
                class_name = str(track["class_name"])
                try:
                    region = compute_crop_region(
                        track["bbox"], width, height,  # type: ignore[arg-type]
                        self.config.crop_margin_ratio, self.config.min_crop_size_px,
                    )
                    jobs.append(
                        _CropJob(frame_index, track_id, class_name, region, extract_crop(pending.frame, region))
                    )
                except (InvalidCropError, KeyError, TypeError, ValueError) as exc:
                    self.metrics.crops_failed += 1
                    poses_by_frame[frame_index][track_id] = PoseData(
                        track_id=track_id, class_name=class_name,
                        status=PoseStatus.FAILED, error=f"invalid_crop: {exc}",
                    )

        inference_ms = 0.0
        if jobs:
            inference_ms = self._infer_jobs(jobs, poses_by_frame)

        model_info = self._model.model_info
        for frame_index, pending in enumerate(frames):
            # Se preserva el orden original de los tracks del TrackingResult.
            ordered = [
                poses_by_frame[frame_index][str(t["track_id"])]
                for t in pending.tracks
                if str(t["track_id"]) in poses_by_frame[frame_index]
            ]
            result = SpecializedAIResult(
                camera_id=pending.camera_id,
                session_id=pending.session_id,
                frame_id=pending.frame_id,
                timestamp=pending.timestamp,
                model_info=model_info,
                inference_time_ms=round(inference_ms, 3),
                poses=ordered,
            )
            self._ensure_output_queue(pending.camera_id).put(result.to_dict())
            self.frame_cache.evict_up_to(pending.camera_id, pending.frame_id)
            self.metrics.results_processed += 1
            self.metrics.register_processed()

    def _accept_frame_order(self, pending: _PendingFrame) -> bool:
        """Descarta frames atrasados/fuera de orden por camara: priorizar
        tiempo real y mantener coherencia temporal para Behavior Engine."""
        with self._last_frame_lock:
            last = self._last_frame_id.get(pending.camera_id)
            if last is not None and pending.frame_id <= last:
                self.metrics.results_dropped_stale += 1
                _rate_limited_logger.log(
                    f"stale_{pending.camera_id}", logging.WARNING,
                    f"[{pending.camera_id}] Frames atrasados descartados (%d en la ultima ventana)",
                )
                return False
            self._last_frame_id[pending.camera_id] = pending.frame_id
            return True

    def _infer_jobs(self, jobs: List[_CropJob], poses_by_frame: List[Dict[str, PoseData]]) -> float:
        """Inferencia de todos los crops en un solo batch. Si el batch falla
        por un error no-OOM, se reintenta crop por crop para aislar el crop
        problematico y no perder la pose del resto de personas."""
        start = time.time()
        try:
            raw_poses = self._model.infer_batch([job.crop for job in jobs])
            if len(raw_poses) != len(jobs):
                raise RuntimeError(
                    f"El modelo devolvio {len(raw_poses)} resultados para {len(jobs)} crops"
                )
        except InferenceOOMError as exc:
            self._handle_oom(exc, jobs, poses_by_frame)
            return (time.time() - start) * 1000.0
        except Exception as exc:
            logger.warning(
                "Fallo la inferencia en batch (%d crops), reintentando crop por crop: %s",
                len(jobs), exc,
            )
            raw_poses = self._infer_individually(jobs, poses_by_frame)

        elapsed = time.time() - start
        self.metrics.batches_processed += 1
        self.metrics.total_inference_seconds += elapsed

        for job, raw in zip(jobs, raw_poses):
            if job.track_id in poses_by_frame[job.frame_index]:
                continue  # ya marcado FAILED en el reintento individual
            poses_by_frame[job.frame_index][job.track_id] = self._build_pose_data(job, raw)
        return elapsed * 1000.0

    def _infer_individually(
        self,
        jobs: List[_CropJob],
        poses_by_frame: List[Dict[str, PoseData]],
    ) -> List[Optional[RawPose]]:
        raw_poses: List[Optional[RawPose]] = []
        for job in jobs:
            try:
                raw_poses.append(self._model.infer_batch([job.crop])[0])
            except Exception as exc:
                self.metrics.crops_failed += 1
                poses_by_frame[job.frame_index][job.track_id] = PoseData(
                    track_id=job.track_id, class_name=job.class_name,
                    status=PoseStatus.FAILED, error=f"inference_error: {type(exc).__name__}",
                )
                raw_poses.append(None)
        return raw_poses

    def _build_pose_data(self, job: _CropJob, raw: Optional[RawPose]) -> PoseData:
        if raw is None:
            self.metrics.crops_no_pose += 1
            return PoseData(track_id=job.track_id, class_name=job.class_name, status=PoseStatus.NO_POSE)
        try:
            keypoints = project_keypoints(raw, job.region, self.config.keypoint_conf_threshold)
            signals = compute_derived_signals(keypoints)
        except Exception as exc:
            self.metrics.crops_failed += 1
            return PoseData(
                track_id=job.track_id, class_name=job.class_name,
                status=PoseStatus.FAILED, error=f"postprocess_error: {type(exc).__name__}",
            )
        self.metrics.crops_inferred += 1
        return PoseData(
            track_id=job.track_id,
            class_name=job.class_name,
            status=PoseStatus.OK,
            keypoints=keypoints,
            derived_signals=signals,
        )

    def _handle_oom(
        self,
        exc: InferenceOOMError,
        jobs: List[_CropJob],
        poses_by_frame: List[Dict[str, PoseData]],
    ) -> None:
        self.metrics.oom_events += 1
        self.metrics.crops_failed += len(jobs)
        logger.critical(
            "CUDA Out-Of-Memory en inferencia de pose (crops=%d). Liberando cache de GPU "
            "y vaciando la cola de entrada. error=%s", len(jobs), exc,
        )
        for job in jobs:
            poses_by_frame[job.frame_index][job.track_id] = PoseData(
                track_id=job.track_id, class_name=job.class_name,
                status=PoseStatus.FAILED, error="gpu_out_of_memory",
            )
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        self._input_queue.clear()

    # -------------------------------------------------------------------
    # Colas / backpressure
    # -------------------------------------------------------------------

    def _ensure_output_queue(self, camera_id: str) -> BoundedQueue:
        with self._output_lock:
            queue = self._output_queues.get(camera_id)
            if queue is None:
                queue = BoundedQueue(
                    self.config.output_queue_size_per_camera,
                    on_drop=lambda cam=camera_id: self._on_output_drop(cam),
                )
                self._output_queues[camera_id] = queue
            return queue

    def _on_input_drop(self) -> None:
        self.metrics.results_dropped_backpressure += 1
        _rate_limited_logger.log(
            "input_backpressure", logging.WARNING,
            "Cola de entrada de pose llena: frames mas antiguos descartados (%d en la ultima ventana)",
        )

    def _on_output_drop(self, camera_id: str) -> None:
        self.metrics.results_dropped_backpressure += 1
        _rate_limited_logger.log(
            f"output_backpressure_{camera_id}", logging.WARNING,
            f"[{camera_id}] Cola de salida de pose llena (consumidor lento): resultados "
            f"mas antiguos descartados (%d en la ultima ventana)",
        )


# =============================================================================
# Smoke test manual (modelo sintetico, sin pesos reales)
# =============================================================================

class _SyntheticPoseModel(SpecializedModelInterface):
    """SOLO para validar crop -> proyeccion -> señales sin pesos reales.
    Devuelve un esqueleto erguido con brazos flexionados a 90 grados y
    oculta los tobillos (confianza baja) para ejercitar la oclusion."""

    supports_batching = True

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo("synthetic-pose", "0.0.0", "SYNTHETIC", "cpu")

    def load(self) -> None:
        self._loaded = True

    def infer_batch(self, crops: List[np.ndarray]) -> List[Optional[RawPose]]:
        skeleton = {
            "nose": (50, 20), "left_eye": (48, 18), "right_eye": (52, 18),
            "left_ear": (44, 20), "right_ear": (56, 20),
            "left_shoulder": (40, 40), "right_shoulder": (60, 40),
            "left_elbow": (40, 60), "right_elbow": (60, 60),
            "left_wrist": (20, 60), "right_wrist": (80, 60),
            "left_hip": (42, 90), "right_hip": (58, 90),
            "left_knee": (42, 120), "right_knee": (58, 120),
            "left_ankle": (42, 150), "right_ankle": (58, 150),
        }
        xy = np.array([skeleton[name] for name in COCO_KEYPOINTS], dtype=np.float64)
        conf = np.full(len(COCO_KEYPOINTS), 0.9)
        conf[COCO_KEYPOINTS.index("left_ankle")] = 0.1
        conf[COCO_KEYPOINTS.index("right_ankle")] = 0.1
        return [RawPose(xy=xy.copy(), confidence=conf.copy()) for _ in crops]


if __name__ == "__main__":
    demo_config = PoseConfig(crop_margin_ratio=0.0)
    service = PoseService(config=demo_config, model=_SyntheticPoseModel(demo_config))
    service._model.load()
    service.start()

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    service.cache_frame("CAM-001", 7, frame)
    service.submit_tracking_result(
        {
            "camera_id": "CAM-001",
            "session_id": "session-demo",
            "frame_id": 7,
            "timestamp": time.time(),
            "tracks": [
                {"track_id": "CAM-001-1", "class_name": "person", "track_state": "ACTIVE",
                 "bbox": {"x1": 500, "y1": 300, "x2": 600, "y2": 500}},
                {"track_id": "CAM-001-2", "class_name": "person", "track_state": "NEW",
                 "bbox": {"x1": 1915, "y1": 10, "x2": 1930, "y2": 20}},   # crop invalido
                {"track_id": "CAM-001-3", "class_name": "person", "track_state": "LOST",
                 "bbox": {"x1": 10, "y1": 10, "x2": 110, "y2": 210}},     # ignorado por estado
            ],
        }
    )

    result = service.get_next_result("CAM-001", timeout=2.0)
    if result is not None:
        for pose in result["poses"]:  # type: ignore[union-attr]
            logger.info(
                "track=%s status=%s error=%s left_shoulder=%s left_ankle=%s signals=%s",
                pose["track_id"], pose["status"], pose["error"],
                pose["keypoints"].get("left_shoulder"), pose["keypoints"].get("left_ankle"),
                pose["derived_signals"],
            )
    logger.info("Health: %s", service.health_check()["metrics"])
    service.stop()
