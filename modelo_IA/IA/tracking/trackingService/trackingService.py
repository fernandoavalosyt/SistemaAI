"""
trackingService.py

Servicio de Tracking / Seguimiento temporal (Servicio 4 del pipeline).

Responsabilidad UNICA: recibir DetectionResult (detecciones "crudas" y
estaticas por frame) desde "YOLO/Detection" (Servicio 3), asociarlas
matematicamente a traves del tiempo (IoU + Algoritmo Hungaro + prediccion
de Kalman) y emitir un TrackingResult con identidad temporal (track_id) y
trayectoria hacia "Pose/Modelos" (Servicio 5).

Tracking SOLO responde una pregunta matematica: "La caja X en el frame N es
la misma entidad que la caja Y en el frame N+1?". Este modulo NO interpreta
comportamiento, NO genera alertas, NO implementa reglas del tipo "si esta
mucho tiempo es sospechoso" y NO implementa Re-Identificacion (ReID) entre
camaras distintas. Cada camara mantiene su propio espacio de identidades,
completamente aislado.
"""

from __future__ import annotations

import abc
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, Generator, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


# =============================================================================
# Logging
# =============================================================================

logger = logging.getLogger("tracking_service")
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
    """Error de validacion del DetectionResult de entrada."""


# =============================================================================
# Maquina de estados del track
# =============================================================================

class TrackState(str, Enum):
    NEW = "NEW"
    ACTIVE = "ACTIVE"
    LOST = "LOST"
    RECOVERING = "RECOVERING"
    TERMINATED = "TERMINATED"


# =============================================================================
# Contratos de datos
# =============================================================================

@dataclass(frozen=True)
class BBox:
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

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_dict(self) -> Dict[str, float]:
        return {
            "x1": round(self.x1, 2),
            "y1": round(self.y1, 2),
            "x2": round(self.x2, 2),
            "y2": round(self.y2, 2),
            "width": round(self.width, 2),
            "height": round(self.height, 2),
        }


@dataclass(frozen=True)
class InputDetection:
    """Deteccion cruda, tal como la entrega YOLO (Servicio 3), ya parseada
    y validada."""

    detection_id: str
    class_id: int
    class_name: str
    confidence: float
    bbox: BBox


@dataclass(frozen=True)
class TrackConfig:
    """Configuracion del ciclo de vida y de la asociacion de tracks."""

    iou_match_threshold: float = 0.3
    max_lost_frames: int = 30
    min_hits_to_confirm: int = 3
    history_max_len: int = 50
    process_noise: float = 1.0
    measurement_noise: float = 10.0
    id_switch_soft_iou_margin: float = 0.15


@dataclass
class TrackSummary:
    """Registro de un track en un instante dado, listo para el contrato
    de salida (TrackingResult.tracks)."""

    track_id: str
    class_name: str
    bbox: BBox
    center: Tuple[float, float]
    detection_confidence: float
    track_state: TrackState
    first_seen: float
    duration_seconds: float
    frames_seen: int
    frames_missing: int
    position_history: List[Tuple[float, float]]

    def to_dict(self) -> Dict[str, object]:
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "bbox": self.bbox.to_dict(),
            "center": {"x": round(self.center[0], 2), "y": round(self.center[1], 2)},
            "detection_confidence": self.detection_confidence,
            "track_state": self.track_state.value,
            "first_seen": self.first_seen,
            "duration_seconds": round(self.duration_seconds, 3),
            "frames_seen": self.frames_seen,
            "frames_missing": self.frames_missing,
            "position_history": [
                {"x": round(x, 2), "y": round(y, 2)} for x, y in self.position_history
            ],
        }


@dataclass
class TrackingResult:
    """Contrato de salida hacia el Servicio 5 (Pose/Modelos)."""

    camera_id: str
    session_id: str
    frame_id: int
    timestamp: float
    tracks: List[TrackSummary]

    def to_dict(self) -> Dict[str, object]:
        return {
            "camera_id": self.camera_id,
            "session_id": self.session_id,
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "tracks": [t.to_dict() for t in self.tracks],
        }


# =============================================================================
# Validacion del contrato de entrada (desde YOLO/Detection)
# =============================================================================

_REQUIRED_RESULT_KEYS = ("camera_id", "session_id", "frame_id", "timestamp", "detections")
_REQUIRED_DETECTION_KEYS = ("class_id", "class_name", "confidence", "bbox_original_coords")
_REQUIRED_BBOX_KEYS = ("x1", "y1", "x2", "y2")


def validate_detection_result(payload: Optional[Dict[str, object]]) -> None:
    """Valida el DetectionResult recibido desde YOLO/Detection. Descarta
    (no propaga) cualquier entrada incompleta, corrupta o de una
    camara/sesion invalida."""

    if not payload:
        raise InputValidationError("Input nulo o vacio")

    for key in _REQUIRED_RESULT_KEYS:
        if key not in payload:
            raise InputValidationError(f"Falta campo obligatorio en DetectionResult: {key}")

    if not payload["camera_id"]:
        raise InputValidationError("camera_id invalido/vacio")
    if not payload["session_id"]:
        raise InputValidationError("session_id invalido/vacio")

    timestamp = payload["timestamp"]
    if timestamp is None or timestamp <= 0:
        raise InputValidationError("timestamp invalido")

    detections = payload["detections"]
    if not isinstance(detections, list):
        raise InputValidationError("El campo 'detections' debe ser una lista")

    for det in detections:
        if not isinstance(det, dict):
            raise InputValidationError("Cada deteccion debe ser un dict")
        for key in _REQUIRED_DETECTION_KEYS:
            if key not in det:
                raise InputValidationError(f"Falta campo obligatorio en deteccion: {key}")
        bbox = det["bbox_original_coords"]
        if not isinstance(bbox, dict):
            raise InputValidationError("bbox_original_coords invalido (se esperaba dict)")
        for key in _REQUIRED_BBOX_KEYS:
            if key not in bbox:
                raise InputValidationError(f"Falta campo obligatorio en bbox: {key}")


def parse_detections(payload: Dict[str, object]) -> List[InputDetection]:
    """Convierte la lista cruda de detecciones del payload en objetos
    InputDetection tipados. Asume que validate_detection_result() ya paso."""
    parsed: List[InputDetection] = []
    for det in payload["detections"]:  # type: ignore[index]
        bbox_raw = det["bbox_original_coords"]
        bbox = BBox(
            x1=float(bbox_raw["x1"]),
            y1=float(bbox_raw["y1"]),
            x2=float(bbox_raw["x2"]),
            y2=float(bbox_raw["y2"]),
        )
        if bbox.width <= 0 or bbox.height <= 0:
            continue  # bbox degenerada, se ignora silenciosamente (no rompe el batch)
        parsed.append(
            InputDetection(
                detection_id=str(det.get("detection_id", "")),
                class_id=int(det["class_id"]),
                class_name=str(det["class_name"]),
                confidence=float(det["confidence"]),
                bbox=bbox,
            )
        )
    return parsed


# =============================================================================
# Metrica de similitud: IoU (Intersection over Union)
# =============================================================================

def iou(a: BBox, b: BBox) -> float:
    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)

    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    union_area = a.area + b.area - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def iou_matrix(tracks_bboxes: List[BBox], detections_bboxes: List[BBox]) -> np.ndarray:
    matrix = np.zeros((len(tracks_bboxes), len(detections_bboxes)), dtype=np.float64)
    for i, track_bbox in enumerate(tracks_bboxes):
        for j, det_bbox in enumerate(detections_bboxes):
            matrix[i, j] = iou(track_bbox, det_bbox)
    return matrix


# =============================================================================
# Filtro de Kalman de velocidad constante (modelo clasico estilo SORT)
# =============================================================================

class KalmanBoxTracker:
    """Filtro de Kalman lineal de velocidad constante sobre el estado
    [cx, cy, s, r, vx, vy, vs], donde s=area del bbox y r=aspect ratio
    (asumido aproximadamente constante entre frames consecutivos).

    Permite predecir la posicion esperada de un track incluso cuando la
    deteccion de YOLO falla temporalmente (oclusion), habilitando el
    estado LOST -> RECOVERING del ciclo de vida del track.
    """

    _STATE_DIM = 7
    _MEASURE_DIM = 4

    def __init__(self, initial_bbox: BBox, process_noise: float, measurement_noise: float) -> None:
        cx, cy = initial_bbox.center
        w, h = initial_bbox.width, initial_bbox.height
        s = max(w * h, 1e-6)
        r = w / h if h > 1e-6 else 1.0

        self.state = np.array([cx, cy, s, r, 0.0, 0.0, 0.0], dtype=np.float64)
        self.covariance = np.eye(self._STATE_DIM, dtype=np.float64) * 10.0
        self.covariance[4:, 4:] *= 100.0  # incertidumbre inicial alta sobre velocidades

        self._H = np.zeros((self._MEASURE_DIM, self._STATE_DIM), dtype=np.float64)
        self._H[0, 0] = 1.0
        self._H[1, 1] = 1.0
        self._H[2, 2] = 1.0
        self._H[3, 3] = 1.0

        self._Q_scale = process_noise
        self._R = np.eye(self._MEASURE_DIM, dtype=np.float64) * measurement_noise

    def predict(self, dt: float) -> BBox:
        dt = max(dt, 1e-3)
        F = np.eye(self._STATE_DIM, dtype=np.float64)
        F[0, 4] = dt
        F[1, 5] = dt
        F[2, 6] = dt

        Q = np.eye(self._STATE_DIM, dtype=np.float64) * self._Q_scale * dt

        self.state = F @ self.state
        self.covariance = F @ self.covariance @ F.T + Q

        self.state[2] = max(self.state[2], 1e-6)  # el area no puede ser negativa
        return self._state_to_bbox()

    def update(self, measured_bbox: BBox) -> None:
        cx, cy = measured_bbox.center
        w, h = measured_bbox.width, measured_bbox.height
        s = max(w * h, 1e-6)
        r = w / h if h > 1e-6 else 1.0
        z = np.array([cx, cy, s, r], dtype=np.float64)

        y = z - self._H @ self.state
        S = self._H @ self.covariance @ self._H.T + self._R
        kalman_gain = self.covariance @ self._H.T @ np.linalg.inv(S)

        self.state = self.state + kalman_gain @ y
        identity = np.eye(self._STATE_DIM, dtype=np.float64)
        self.covariance = (identity - kalman_gain @ self._H) @ self.covariance
        self.state[2] = max(self.state[2], 1e-6)

    def current_bbox(self) -> BBox:
        return self._state_to_bbox()

    def _state_to_bbox(self) -> BBox:
        cx, cy = float(self.state[0]), float(self.state[1])
        s = max(float(self.state[2]), 1e-6)
        r = max(float(self.state[3]), 1e-6)
        w = float(np.sqrt(s * r))
        h = float(s / w) if w > 1e-6 else 0.0
        return BBox(x1=cx - w / 2.0, y1=cy - h / 2.0, x2=cx + w / 2.0, y2=cy + h / 2.0)


# =============================================================================
# Track individual: identidad temporal + historial + ciclo de vida
# =============================================================================

class Track:
    """Representa una unica identidad temporal dentro de UNA camara.
    Nunca se comparte ni se referencia entre camaras distintas."""

    def __init__(
        self,
        track_id: str,
        detection: InputDetection,
        timestamp: float,
        config: TrackConfig,
    ) -> None:
        self.track_id = track_id
        self.class_name = detection.class_name
        self.state = TrackState.NEW
        self.config = config

        self.kalman = KalmanBoxTracker(
            initial_bbox=detection.bbox,
            process_noise=config.process_noise,
            measurement_noise=config.measurement_noise,
        )

        self.first_seen = timestamp
        self.last_seen = timestamp
        self.last_update_timestamp = timestamp
        self.frames_seen = 1
        self.frames_missing = 0
        self.consecutive_hits = 1
        self.detection_confidence = detection.confidence
        self.marked_for_removal = False

        self.history: Deque[Tuple[float, float]] = deque(maxlen=config.history_max_len)
        self.history.append(detection.bbox.center)

    def predict(self, timestamp: float) -> BBox:
        dt = max(timestamp - self.last_update_timestamp, 1e-3)
        return self.kalman.predict(dt)

    def mark_matched(self, detection: InputDetection, timestamp: float) -> None:
        self.kalman.update(detection.bbox)
        self.detection_confidence = detection.confidence
        self.last_seen = timestamp
        self.last_update_timestamp = timestamp
        self.frames_seen += 1
        self.frames_missing = 0
        self.consecutive_hits += 1
        self.history.append(self.kalman.current_bbox().center)

        if self.state in (TrackState.NEW,):
            if self.consecutive_hits >= self.config.min_hits_to_confirm:
                self.state = TrackState.ACTIVE
        elif self.state == TrackState.LOST:
            self.state = TrackState.RECOVERING
            logger.info("Track recuperado (LOST -> RECOVERING) | track_id=%s", self.track_id)
        elif self.state == TrackState.RECOVERING:
            self.state = TrackState.ACTIVE
            logger.info("Track confirmado (RECOVERING -> ACTIVE) | track_id=%s", self.track_id)
        # Si ya estaba ACTIVE, permanece ACTIVE.

    def mark_unmatched(self, timestamp: float, max_lost_frames: int) -> None:
        self.frames_missing += 1
        self.consecutive_hits = 0
        self.last_update_timestamp = timestamp

        if self.state in (TrackState.NEW, TrackState.ACTIVE, TrackState.RECOVERING):
            self.state = TrackState.LOST

        if self.state == TrackState.LOST and self.frames_missing > max_lost_frames:
            self.state = TrackState.TERMINATED

    def to_summary(self, now: float) -> TrackSummary:
        bbox = self.kalman.current_bbox()
        return TrackSummary(
            track_id=self.track_id,
            class_name=self.class_name,
            bbox=bbox,
            center=bbox.center,
            detection_confidence=self.detection_confidence,
            track_state=self.state,
            first_seen=self.first_seen,
            duration_seconds=max(0.0, self.last_seen - self.first_seen),
            frames_seen=self.frames_seen,
            frames_missing=self.frames_missing,
            position_history=list(self.history),
        )


# =============================================================================
# Abstraccion del algoritmo de tracking (Pro-Tip: intercambiable)
# =============================================================================

class TrackerInterface(abc.ABC):
    """Interfaz abstracta de un algoritmo de tracking para UNA camara. El
    TrackingService trabaja unicamente contra esta abstraccion, permitiendo
    inyectar en el futuro un DeepSORT o BoT-SORT sin modificar el resto del
    servicio."""

    def __init__(self, camera_id: str, config: TrackConfig) -> None:
        self.camera_id = camera_id
        self.config = config

    @abc.abstractmethod
    def update(
        self,
        detections: List[InputDetection],
        frame_id: int,
        timestamp: float,
    ) -> List[TrackSummary]:
        """Asocia las detecciones del frame actual con los tracks
        existentes, avanza el ciclo de vida de cada track y retorna el
        snapshot de todos los tracks vigentes (incluye el ultimo evento de
        los recien TERMINATED)."""
        raise NotImplementedError

    @abc.abstractmethod
    def active_track_count(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self) -> None:
        raise NotImplementedError


class SortLikeTracker(TrackerInterface):
    """Implementacion de referencia: SORT (Simple Online and Realtime
    Tracking) - IoU + Algoritmo Hungaro (asignacion optima) + Filtro de
    Kalman de velocidad constante para prediccion durante oclusiones."""

    def __init__(self, camera_id: str, config: TrackConfig) -> None:
        super().__init__(camera_id, config)
        self._tracks: Dict[str, Track] = {}
        self._next_track_seq = 0
        self._lock = threading.Lock()
        self.last_id_switch_estimate = 0

    def tracks_snapshot(self) -> Dict[str, TrackState]:
        """Copia superficial (track_id -> estado) para diagnostico externo,
        sin exponer los objetos Track internos."""
        with self._lock:
            return {tid: track.state for tid, track in self._tracks.items()}

    def update(
        self,
        detections: List[InputDetection],
        frame_id: int,
        timestamp: float,
    ) -> List[TrackSummary]:
        with self._lock:
            self._cleanup_terminated()

            track_ids = list(self._tracks.keys())
            predicted_bboxes = [self._tracks[tid].predict(timestamp) for tid in track_ids]
            detection_bboxes = [d.bbox for d in detections]

            matched_pairs, unmatched_track_idx, unmatched_det_idx = self._associate(
                predicted_bboxes, detection_bboxes
            )

            weak_match_count = 0
            for track_idx, det_idx in matched_pairs:
                matched_iou = iou(predicted_bboxes[track_idx], detection_bboxes[det_idx])
                if matched_iou < (self.config.iou_match_threshold + self.config.id_switch_soft_iou_margin):
                    # Heuristica: una asociacion valida pero "debil" (IoU
                    # apenas por encima del umbral) es una señal indirecta
                    # de posible ID switch. No es una deteccion exacta
                    # (requeriria Re-ID), solo una estimacion agregada.
                    weak_match_count += 1
                track = self._tracks[track_ids[track_idx]]
                track.mark_matched(detections[det_idx], timestamp)
            self.last_id_switch_estimate = weak_match_count

            for track_idx in unmatched_track_idx:
                track = self._tracks[track_ids[track_idx]]
                previous_state = track.state
                track.mark_unmatched(timestamp, self.config.max_lost_frames)
                if track.state == TrackState.TERMINATED and previous_state != TrackState.TERMINATED:
                    logger.info(
                        "Track terminado por timeout (max_lost_frames=%d) | camera_id=%s track_id=%s",
                        self.config.max_lost_frames,
                        self.camera_id,
                        track.track_id,
                    )
                    track.marked_for_removal = True

            for det_idx in unmatched_det_idx:
                self._create_track(detections[det_idx], timestamp)

            summaries = [track.to_summary(timestamp) for track in self._tracks.values()]
            return summaries

    def active_track_count(self) -> int:
        with self._lock:
            return sum(
                1
                for track in self._tracks.values()
                if track.state in (TrackState.NEW, TrackState.ACTIVE, TrackState.RECOVERING, TrackState.LOST)
            )

    def reset(self) -> None:
        with self._lock:
            self._tracks.clear()
            self._next_track_seq = 0

    # -------------------------------------------------------------------
    # Internos
    # -------------------------------------------------------------------

    def _create_track(self, detection: InputDetection, timestamp: float) -> None:
        self._next_track_seq += 1
        track_id = f"{self.camera_id}-{self._next_track_seq}"
        self._tracks[track_id] = Track(track_id, detection, timestamp, self.config)
        logger.info("Track ID creado | camera_id=%s track_id=%s class=%s", self.camera_id, track_id, detection.class_name)

    def _cleanup_terminated(self) -> None:
        """Recolector de basura: elimina de memoria los tracks TERMINATED
        que ya emitieron su evento final en el ciclo anterior, previniendo
        memory leaks por acumulacion indefinida."""
        to_remove = [tid for tid, track in self._tracks.items() if track.marked_for_removal]
        for tid in to_remove:
            del self._tracks[tid]

    def _associate(
        self,
        predicted_bboxes: List[BBox],
        detection_bboxes: List[BBox],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Asociacion optima track<->deteccion via Algoritmo Hungaro sobre
        una matriz de costos derivada de IoU, con gating por
        iou_match_threshold para evitar asociaciones matematicamente
        invalidas."""

        if not predicted_bboxes or not detection_bboxes:
            return [], list(range(len(predicted_bboxes))), list(range(len(detection_bboxes)))

        cost_matrix = 1.0 - iou_matrix(predicted_bboxes, detection_bboxes)
        track_indices, det_indices = linear_sum_assignment(cost_matrix)

        matched_pairs: List[Tuple[int, int]] = []
        unmatched_tracks = set(range(len(predicted_bboxes)))
        unmatched_dets = set(range(len(detection_bboxes)))

        for track_idx, det_idx in zip(track_indices, det_indices):
            matched_iou = 1.0 - cost_matrix[track_idx, det_idx]
            if matched_iou < self.config.iou_match_threshold:
                continue  # gating: asociacion matematicamente insuficiente, se descarta
            matched_pairs.append((int(track_idx), int(det_idx)))
            unmatched_tracks.discard(track_idx)
            unmatched_dets.discard(det_idx)

        return matched_pairs, sorted(unmatched_tracks), sorted(unmatched_dets)


# -----------------------------------------------------------------------------
# NOTA ARQUITECTONICA:
# Los siguientes trackers son STUBS preparados para algoritmos mas
# avanzados (con apariencia visual / Re-ID intra-camara). NO estan activos
# por defecto. Cuando se habiliten, deberan implementar update() /
# active_track_count() / reset() respetando TrackerInterface, sin requerir
# cambios en TrackingService.
# -----------------------------------------------------------------------------

class DeepSortTracker(TrackerInterface):
    """[STUB] Tracker con embeddings de apariencia (Re-ID intra-camara).

    Pendiente de implementacion. Uso previsto: combinar el costo de IoU con
    una distancia de embeddings (coseno/euclidiana) extraidos por una red
    de Re-ID, para mejorar la asociacion en escenas con cruces frecuentes.
    Mantiene el mismo aislamiento estricto por camera_id.
    """

    def update(self, detections: List[InputDetection], frame_id: int, timestamp: float) -> List[TrackSummary]:
        raise NotImplementedError("DeepSortTracker aun no esta implementado.")

    def active_track_count(self) -> int:
        raise NotImplementedError("DeepSortTracker aun no esta implementado.")

    def reset(self) -> None:
        raise NotImplementedError("DeepSortTracker aun no esta implementado.")


class BotSortTracker(TrackerInterface):
    """[STUB] Tracker con compensacion de movimiento de camara + Re-ID.

    Pendiente de implementacion. Uso previsto: BoT-SORT combina Kalman +
    Hungaro (como SortLikeTracker) con compensacion global de movimiento de
    camara (GMC) y un modulo de Re-ID opcional.
    """

    def update(self, detections: List[InputDetection], frame_id: int, timestamp: float) -> List[TrackSummary]:
        raise NotImplementedError("BotSortTracker aun no esta implementado.")

    def active_track_count(self) -> int:
        raise NotImplementedError("BotSortTracker aun no esta implementado.")

    def reset(self) -> None:
        raise NotImplementedError("BotSortTracker aun no esta implementado.")


def create_tracker(camera_id: str, config: TrackConfig) -> TrackerInterface:
    """Factory del algoritmo de tracking activo. Punto unico de extension
    para inyectar en el futuro DeepSortTracker/BotSortTracker."""
    return SortLikeTracker(camera_id=camera_id, config=config)


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
# Metricas
# =============================================================================

@dataclass
class CameraTrackingMetrics:
    results_received: int = 0
    results_processed: int = 0
    results_dropped_invalid: int = 0
    results_dropped_stale: int = 0
    results_dropped_backpressure: int = 0
    tracks_created: int = 0
    tracks_recovered: int = 0
    tracks_terminated: int = 0
    id_switches_estimated: int = 0
    total_assignment_latency_seconds: float = 0.0

    _fps_window_count: int = field(default=0, repr=False)
    _fps_window_start: float = field(default_factory=time.time, repr=False)
    measured_tracking_fps: float = 0.0

    def register_processed(self) -> None:
        self._fps_window_count += 1
        elapsed = time.time() - self._fps_window_start
        if elapsed >= 1.0:
            self.measured_tracking_fps = round(self._fps_window_count / elapsed, 2)
            self._fps_window_count = 0
            self._fps_window_start = time.time()

    @property
    def avg_assignment_latency_ms(self) -> float:
        if self.results_processed == 0:
            return 0.0
        return round((self.total_assignment_latency_seconds / self.results_processed) * 1000.0, 3)

    def to_dict(self) -> Dict[str, object]:
        return {
            "results_received": self.results_received,
            "results_processed": self.results_processed,
            "results_dropped_invalid": self.results_dropped_invalid,
            "results_dropped_stale": self.results_dropped_stale,
            "results_dropped_backpressure": self.results_dropped_backpressure,
            "tracks_created": self.tracks_created,
            "tracks_recovered": self.tracks_recovered,
            "tracks_terminated": self.tracks_terminated,
            "id_switches_estimated": self.id_switches_estimated,
            "avg_assignment_latency_ms": self.avg_assignment_latency_ms,
            "measured_tracking_fps": self.measured_tracking_fps,
        }


# =============================================================================
# Worker por camara: aislamiento total de memoria e hilos entre camaras
# =============================================================================

class CameraTrackingWorker:
    """Procesa el stream de DetectionResult de UNA unica camara en un hilo
    dedicado, con su propio TrackerInterface (espacio de track_id aislado),
    buffers de entrada/salida y metricas."""

    def __init__(
        self,
        camera_id: str,
        config: TrackConfig,
        input_buffer_size: int = 30,
        output_buffer_size: int = 30,
    ) -> None:
        self.camera_id = camera_id
        self.config = config
        self.metrics = CameraTrackingMetrics()
        self.tracker = create_tracker(camera_id, config)

        self._input_queue = BoundedQueue(max_size=input_buffer_size, on_drop=self._on_input_drop)
        self._output_queue = BoundedQueue(max_size=output_buffer_size, on_drop=self._on_output_drop)

        self._last_processed_frame_id: Optional[int] = None
        self._last_processed_timestamp: Optional[float] = None

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        logger.info("Tracker inicializado | camera_id=%s", self.camera_id)
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"CameraTrackingWorker-{self.camera_id}", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._input_queue.clear()
        self._output_queue.clear()
        logger.info("Worker de tracking detenido | camera_id=%s", self.camera_id)

    def submit(self, payload: Dict[str, object]) -> None:
        self._input_queue.put(payload)

    def get_next_result(self, timeout: float = 1.0) -> Optional[Dict[str, object]]:
        return self._output_queue.get(timeout=timeout)  # type: ignore[return-value]

    def health_check(self) -> Dict[str, object]:
        return {
            "camera_id": self.camera_id,
            "active_tracks": self.tracker.active_track_count(),
            "input_buffer_usage": f"{self._input_queue.qsize()}/{self._input_queue.max_size}",
            "output_buffer_usage": f"{self._output_queue.qsize()}/{self._output_queue.max_size}",
            "metrics": self.metrics.to_dict(),
        }

    def _run(self) -> None:
        while not self._stop_event.is_set():
            payload = self._input_queue.get(timeout=1.0)
            if payload is None:
                continue
            try:
                self._process_one(payload)  # type: ignore[arg-type]
            except Exception:
                logger.exception(
                    "Error inesperado procesando DetectionResult | camera_id=%s", self.camera_id
                )

    def _process_one(self, payload: Dict[str, object]) -> None:
        self.metrics.results_received += 1

        try:
            validate_detection_result(payload)
        except InputValidationError as exc:
            self.metrics.results_dropped_invalid += 1
            _rate_limited_logger.log(
                f"invalid_input_{self.camera_id}",
                logging.WARNING,
                f"[{self.camera_id}] DetectionResult invalidos descartados "
                f"(%d en la ultima ventana): {exc}",
            )
            return

        frame_id = int(payload["frame_id"])  # type: ignore[arg-type]
        timestamp = float(payload["timestamp"])  # type: ignore[arg-type]

        # Descarte de frames atrasados: rompen la coherencia matematica de
        # la velocidad estimada por el filtro de Kalman (saltos espaciales).
        if self._last_processed_frame_id is not None and frame_id <= self._last_processed_frame_id:
            self.metrics.results_dropped_stale += 1
            _rate_limited_logger.log(
                f"stale_frame_{self.camera_id}",
                logging.WARNING,
                f"[{self.camera_id}] Frames atrasados/fuera de orden descartados "
                f"(%d en la ultima ventana)",
            )
            return

        detections = parse_detections(payload)

        tracks_before = (
            self.tracker.tracks_snapshot() if isinstance(self.tracker, SortLikeTracker) else {}
        )

        start = time.time()
        summaries = self.tracker.update(detections, frame_id, timestamp)
        latency = time.time() - start

        self._update_lifecycle_metrics(tracks_before)
        if isinstance(self.tracker, SortLikeTracker):
            self.metrics.id_switches_estimated += self.tracker.last_id_switch_estimate

        self.metrics.total_assignment_latency_seconds += latency
        self.metrics.results_processed += 1
        self.metrics.register_processed()
        self._last_processed_frame_id = frame_id
        self._last_processed_timestamp = timestamp

        result = TrackingResult(
            camera_id=payload["camera_id"],  # type: ignore[arg-type]
            session_id=payload["session_id"],  # type: ignore[arg-type]
            frame_id=frame_id,
            timestamp=timestamp,
            tracks=summaries,
        )
        self._output_queue.put(result.to_dict())

    def _update_lifecycle_metrics(self, tracks_before: Dict[str, TrackState]) -> None:
        if not isinstance(self.tracker, SortLikeTracker):
            return
        tracks_after = self.tracker.tracks_snapshot()
        for tid, state_after in tracks_after.items():
            previous = tracks_before.get(tid)
            if previous is None:
                self.metrics.tracks_created += 1
            elif previous == TrackState.LOST and state_after == TrackState.RECOVERING:
                self.metrics.tracks_recovered += 1
        for tid in tracks_before:
            if tid not in tracks_after:
                self.metrics.tracks_terminated += 1

    def _on_input_drop(self) -> None:
        self.metrics.results_dropped_backpressure += 1
        _rate_limited_logger.log(
            f"input_backpressure_{self.camera_id}",
            logging.WARNING,
            f"[{self.camera_id}] Buffer de entrada de tracking lleno: resultados mas "
            f"antiguos descartados (%d en la ultima ventana)",
        )

    def _on_output_drop(self) -> None:
        self.metrics.results_dropped_backpressure += 1
        _rate_limited_logger.log(
            f"output_backpressure_{self.camera_id}",
            logging.WARNING,
            f"[{self.camera_id}] Buffer de salida de tracking lleno (consumidor lento): "
            f"resultados mas antiguos descartados (%d en la ultima ventana)",
        )


# =============================================================================
# Servicio principal: orquesta un worker aislado por camara
# =============================================================================

class TrackingService:
    """Punto de entrada del Servicio 4. Procesa DetectionResults de
    multiples camaras de forma concurrente; cada camara tiene su propio
    espacio de track_id, hilo, buffers y metricas — sin cruces posibles de
    identidad entre camaras."""

    def __init__(self, config: Optional[TrackConfig] = None) -> None:
        self.config = config or TrackConfig()
        self._workers: Dict[str, CameraTrackingWorker] = {}
        self._lock = threading.Lock()
        self._start_timestamp = time.time()

    def submit_detection_result(self, payload: Dict[str, object]) -> None:
        camera_id = payload.get("camera_id") if isinstance(payload, dict) else None
        if not camera_id:
            _rate_limited_logger.log(
                "missing_camera_id",
                logging.WARNING,
                "DetectionResult sin camera_id recibido, descartado (%d en la ultima ventana)",
            )
            return

        worker = self._ensure_worker(str(camera_id))
        worker.submit(payload)

    def get_next_result(self, camera_id: str, timeout: float = 1.0) -> Optional[Dict[str, object]]:
        with self._lock:
            worker = self._workers.get(camera_id)
        if worker is None:
            return None
        return worker.get_next_result(timeout=timeout)

    def result_stream(self, camera_id: str) -> Generator[Dict[str, object], None, None]:
        while True:
            result = self.get_next_result(camera_id, timeout=1.0)
            if result is not None:
                yield result

    def stop_all(self, timeout: float = 5.0) -> None:
        with self._lock:
            workers = list(self._workers.values())
            self._workers.clear()
        for worker in workers:
            worker.stop(timeout=timeout)

    def health_check(self) -> Dict[str, object]:
        with self._lock:
            workers = dict(self._workers)

        cameras_health = {camera_id: worker.health_check() for camera_id, worker in workers.items()}
        total_active_tracks = sum(w.tracker.active_track_count() for w in workers.values())
        latencies = [w.metrics.avg_assignment_latency_ms for w in workers.values() if w.metrics.results_processed > 0]
        overall_latency = round(sum(latencies) / len(latencies), 3) if latencies else 0.0

        return {
            "service_active": True,
            "active_cameras": list(workers.keys()),
            "active_camera_count": len(workers),
            "total_active_tracks": total_active_tracks,
            "overall_avg_assignment_latency_ms": overall_latency,
            "uptime_seconds": round(time.time() - self._start_timestamp, 3),
            "cameras": cameras_health,
        }

    def _ensure_worker(self, camera_id: str) -> CameraTrackingWorker:
        with self._lock:
            worker = self._workers.get(camera_id)
            if worker is None:
                worker = CameraTrackingWorker(camera_id=camera_id, config=self.config)
                self._workers[camera_id] = worker
                worker.start()
            return worker


# =============================================================================
# Ejemplo de uso manual (smoke test): 3 frames sinteticos, una camara
# =============================================================================

if __name__ == "__main__":
    service = TrackingService(config=TrackConfig(min_hits_to_confirm=2, max_lost_frames=5))

    base_ts = time.time()
    frames_payloads = [
        {
            "camera_id": "CAM-001",
            "session_id": "session-demo",
            "frame_id": frame_id,
            "timestamp": base_ts + frame_id * 0.1,
            "detections": [
                {
                    "detection_id": f"det-{frame_id}",
                    "class_id": 0,
                    "class_name": "person",
                    "confidence": 0.9,
                    "bbox_original_coords": {
                        "x1": 100 + frame_id * 5,
                        "y1": 100 + frame_id * 2,
                        "x2": 200 + frame_id * 5,
                        "y2": 300 + frame_id * 2,
                    },
                }
            ],
        }
        for frame_id in range(1, 4)
    ]

    for payload in frames_payloads:
        service.submit_detection_result(payload)
        time.sleep(0.05)
        result = service.get_next_result("CAM-001", timeout=2.0)
        if result is not None:
            logger.info("TrackingResult recibido: %s", result)

    logger.info("Health check: %s", service.health_check())
    service.stop_all()
