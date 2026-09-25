"""
behaviorService.py

Servicio de Behavior Analysis (Servicio 6 del pipeline).

Responsabilidad UNICA: fusionar a traves del tiempo las trayectorias
(TrackingResult, Servicio 4) y las señales posturales (SpecializedAIResult,
Servicio 5) de cada individuo, y producir EVIDENCIA temporal de patrones de
comportamiento (somnolencia, merodeo, interaccion con producto) hacia el
"Decision Engine" (Servicio 7).

Este servicio produce EVIDENCIA, NO ALERTAS. No se conecta a bases de datos,
no envia notificaciones y no toma decisiones de negocio. Su salida dice
"Evidencia de somnolencia sostenida durante 20.0 s", nunca "el empleado se
durmio, enviar alerta".

Convencion temporal: toda la logica de comportamiento usa el TIEMPO DE LOS
DATOS (timestamp del frame), no el reloj de pared. Esto hace el analisis
determinista y reproducible. El reloj de pared solo se usa para la ventana
de fusion tracking/pose y para detectar streams detenidos.
"""

from __future__ import annotations

import abc
import json
import logging
import math
import os
import threading
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, FrozenSet, Generator, List, Optional, Sequence, Set, Tuple

import numpy as np


# =============================================================================
# Logging
# =============================================================================
# Solo se loguean transiciones de estado importantes y eventos operativos.
# Nunca cada frame.

logger = logging.getLogger("behavior_service")
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
# Excepciones y enums
# =============================================================================

class InputValidationError(Exception):
    """Payload de entrada invalido."""


class PresenceState(str, Enum):
    """Presencia del sujeto desde la perspectiva de Behavior."""

    PRESENT = "PRESENT"
    RECOVERING = "RECOVERING"  # Tracking lo reporto LOST/TERMINATED: periodo de gracia.


class BehaviorPhase(str, Enum):
    """Maquina de estados de cada comportamiento. NORMAL es interno y nunca
    se emite; FINISHED es transitorio (se emite una vez y vuelve a NORMAL)."""

    NORMAL = "NORMAL"
    POSSIBLE = "POSSIBLE"
    SUSTAINED = "SUSTAINED"
    FINISHED = "FINISHED"


_OBSERVABLE_TRACK_STATES: FrozenSet[str] = frozenset({"NEW", "ACTIVE", "RECOVERING"})
_LOST_TRACK_STATES: FrozenSet[str] = frozenset({"LOST", "TERMINATED"})


# =============================================================================
# Utilidades numericas y geometricas
# =============================================================================

Point = Tuple[float, float]


def ramp(value: Optional[float], low: float, high: float) -> Optional[float]:
    """Funcion de pertenencia difusa lineal: 0 en 'low', 1 en 'high'.
    Si low > high la rampa es descendente (valores menores => mas score)."""
    if value is None or not math.isfinite(value):
        return None
    if high == low:
        return 1.0 if value >= high else 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def _round(value: Optional[float], digits: int = 3) -> Optional[float]:
    return None if value is None else round(float(value), digits)


def _median(values: Sequence[float]) -> Optional[float]:
    return float(np.median(values)) if values else None


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Ray casting."""
    x, y = point
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def _point_segment_distance(p: Point, a: Point, b: Point) -> float:
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-12:
        return _distance(p, a)
    t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / length_sq))
    return _distance(p, (ax + t * dx, ay + t * dy))


def point_polygon_distance(point: Point, polygon: Sequence[Point]) -> float:
    """0 si el punto esta dentro; si no, distancia al borde mas cercano."""
    if point_in_polygon(point, polygon):
        return 0.0
    return min(
        _point_segment_distance(point, polygon[i], polygon[(i + 1) % len(polygon)])
        for i in range(len(polygon))
    )


# =============================================================================
# Zonas y contexto espacial
# =============================================================================

@dataclass(frozen=True)
class Zone:
    """Poligono en coordenadas originales de la camara.

    kind="area": zona de piso (pasillo, zona_caja). Se asigna segun el punto
    de apoyo (pies) de la persona.
    kind="product": ROI de producto/anaquel para ProductInteractionBehavior.

    behavior_overrides permite que el MISMO comportamiento tenga parametros
    distintos segun el contexto, ej:
        {"loitering": {"enabled": False}}            # zona_caja: esperar es normal
        {"somnolence": {"sustain_seconds": 30.0}}    # sala de descanso
    """

    zone_id: str
    name: str
    polygon: Tuple[Point, ...]
    kind: str = "area"
    behavior_overrides: Dict[str, Dict[str, object]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.polygon) < 3:
            raise ValueError(f"La zona {self.zone_id} requiere al menos 3 vertices")
        if self.kind not in ("area", "product"):
            raise ValueError(f"Tipo de zona desconocido: {self.kind}")
        for x, y in self.polygon:
            if not (math.isfinite(x) and math.isfinite(y)):
                raise ValueError(f"La zona {self.zone_id} tiene vertices no finitos")


class ZoneRegistry:
    """Configuracion de zonas por camara, inyectable en caliente."""

    def __init__(self) -> None:
        self._zones: Dict[str, List[Zone]] = {}
        self._lock = threading.Lock()

    def set_zones(self, camera_id: str, zones: List[Zone]) -> None:
        with self._lock:
            self._zones[camera_id] = list(zones)
        logger.info("Zonas configuradas | camera_id=%s zonas=%s", camera_id, [z.zone_id for z in zones])

    def area_zone_for(self, camera_id: str, point: Point) -> Optional[Zone]:
        with self._lock:
            zones = self._zones.get(camera_id, [])
        for zone in zones:
            if zone.kind == "area" and point_in_polygon(point, zone.polygon):
                return zone
        return None

    def product_zones(self, camera_id: str) -> List[Zone]:
        with self._lock:
            return [z for z in self._zones.get(camera_id, []) if z.kind == "product"]

    def get(self, camera_id: str, zone_id: Optional[str]) -> Optional[Zone]:
        if zone_id is None:
            return None
        with self._lock:
            for zone in self._zones.get(camera_id, []):
                if zone.zone_id == zone_id:
                    return zone
        return None

    def load_json(self, path: str) -> None:
        """Formato: {"CAM-001": [{"zone_id", "name", "kind", "polygon": [[x,y],...],
        "behavior_overrides": {...}}]}"""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Archivo de zonas no encontrado: {path}")
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        for camera_id, zones_raw in raw.items():
            zones = [
                Zone(
                    zone_id=str(z["zone_id"]),
                    name=str(z.get("name", z["zone_id"])),
                    polygon=tuple((float(p[0]), float(p[1])) for p in z["polygon"]),
                    kind=str(z.get("kind", "area")),
                    behavior_overrides=dict(z.get("behavior_overrides", {})),
                )
                for z in zones_raw
            ]
            self.set_zones(str(camera_id), zones)


# =============================================================================
# Observaciones (memoria temporal)
# =============================================================================

@dataclass
class PoseSnapshot:
    """Rasgos posturales de un frame, derivados del SpecializedAIResult.
    None = no observable (oclusion / angulo de camara)."""

    keypoints: Dict[str, Point]
    torso_inclination_deg: Optional[float]
    head_angle_deg: Optional[float]         # angulo cuello vs eje del torso (0 = erguido)
    neck_height_ratio: Optional[float]      # altura de cabeza sobre hombros / largo de torso
    head_reference: Optional[str]           # "nose" | "ears" (vista de espalda)
    eye_aspect_ratio: Optional[float]       # solo si un modulo facial externo lo provee


def _midpoint(points: List[Optional[Point]]) -> Optional[Point]:
    available = [p for p in points if p is not None]
    if not available:
        return None
    return (sum(p[0] for p in available) / len(available), sum(p[1] for p in available) / len(available))


def build_pose_snapshot(pose: Dict[str, object]) -> Optional[PoseSnapshot]:
    """Convierte un PoseData (status OK) en rasgos posturales. La cabeza se
    referencia con la nariz o, si no es visible (vista de espalda), con el
    punto medio de las orejas: asi la caida de cabeza sigue siendo medible
    aunque no se vea el rostro."""
    if pose.get("status") != "OK":
        return None

    keypoints: Dict[str, Point] = {}
    for name, kp in (pose.get("keypoints") or {}).items():  # type: ignore[union-attr]
        if isinstance(kp, dict) and kp.get("visible") and kp.get("x") is not None and kp.get("y") is not None:
            keypoints[name] = (float(kp["x"]), float(kp["y"]))

    derived = pose.get("derived_signals") or {}
    shoulders = _midpoint([keypoints.get("left_shoulder"), keypoints.get("right_shoulder")])
    hips = _midpoint([keypoints.get("left_hip"), keypoints.get("right_hip")])

    head_reference: Optional[str] = None
    head_point: Optional[Point] = None
    if "nose" in keypoints:
        head_point, head_reference = keypoints["nose"], "nose"
    else:
        ears = _midpoint([keypoints.get("left_ear"), keypoints.get("right_ear")])
        if ears is not None:
            head_point, head_reference = ears, "ears"

    head_angle: Optional[float] = None
    neck_ratio: Optional[float] = None
    if shoulders is not None and hips is not None and head_point is not None:
        torso = np.array(shoulders) - np.array(hips)
        torso_len = float(np.linalg.norm(torso))
        neck = np.array(head_point) - np.array(shoulders)
        neck_len = float(np.linalg.norm(neck))
        if torso_len > 1e-6:
            axis = torso / torso_len
            neck_ratio = float(np.dot(neck, axis) / torso_len)
            if neck_len > 1e-6:
                cos_angle = float(np.clip(np.dot(neck, axis) / neck_len, -1.0, 1.0))
                head_angle = math.degrees(math.acos(cos_angle))

    torso_inclination = derived.get("torso_inclination_deg") if isinstance(derived, dict) else None
    ear_value = derived.get("eye_aspect_ratio") if isinstance(derived, dict) else None

    return PoseSnapshot(
        keypoints=keypoints,
        torso_inclination_deg=None if torso_inclination is None else float(torso_inclination),
        head_angle_deg=head_angle,
        neck_height_ratio=neck_ratio,
        head_reference=head_reference,
        eye_aspect_ratio=None if ear_value is None else float(ear_value),
    )


@dataclass
class Observation:
    timestamp: float
    frame_id: int
    camera_id: str
    track_id: str
    center: Point
    feet: Point
    body_height: float
    zone_id: Optional[str]
    pose: Optional[PoseSnapshot] = None


# =============================================================================
# Episodios y contexto por sujeto
# =============================================================================

@dataclass
class BehaviorEpisode:
    episode_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    phase: BehaviorPhase = BehaviorPhase.NORMAL
    start_time: Optional[float] = None
    last_active_time: Optional[float] = None
    last_emit_time: Optional[float] = None
    peak_score: float = 0.0
    trigger_evidence: Dict[str, object] = field(default_factory=dict)
    last_score: float = 0.0
    last_coverage: float = 0.0
    recoveries: int = 0

    @property
    def is_open(self) -> bool:
        return self.phase in (BehaviorPhase.POSSIBLE, BehaviorPhase.SUSTAINED)


class SubjectContext:
    """Memoria temporal de UN individuo (subject_id).

    Sin ReID, subject_id == "camera_id:track_id". Si un sistema externo de
    ReID unifica tracks de varias camaras, todos apuntan al mismo contexto y
    los modulos evaluan las señales combinadas. Las señales de MOVIMIENTO se
    calculan solo con observaciones de la camara mas reciente (las
    coordenadas de camaras distintas no son comparables); las señales
    POSTURALES (ratios/angulos) si son combinables entre camaras.
    """

    def __init__(self, subject_id: str, class_name: str, window_seconds: float, max_observations: int) -> None:
        self.subject_id = subject_id
        self.class_name = class_name
        self.window_seconds = window_seconds
        self.observations: Deque[Observation] = deque(maxlen=max_observations)
        self.presence = PresenceState.PRESENT
        self.lost_since: Optional[float] = None
        self.terminated = False
        self.last_seen: float = 0.0
        self.last_wall_update: float = time.time()
        self.camera_id: str = ""
        self.session_id: str = ""
        self.track_id: str = ""
        self.linked_tracks: Set[Tuple[str, str]] = set()
        self.zone_id: Optional[str] = None
        self.zone_entry_time: Optional[float] = None
        self.episodes: Dict[str, BehaviorEpisode] = {}

    # -------------------------------------------------------------------
    # Escritura
    # -------------------------------------------------------------------

    def add_observation(self, obs: Observation, session_id: str) -> None:
        if obs.camera_id != self.camera_id or obs.zone_id != self.zone_id or self.zone_entry_time is None:
            self.zone_id = obs.zone_id
            self.zone_entry_time = obs.timestamp
        self.camera_id = obs.camera_id
        self.track_id = obs.track_id
        self.session_id = session_id
        self.linked_tracks.add((obs.camera_id, obs.track_id))
        self.last_seen = obs.timestamp
        self.last_wall_update = time.time()
        self.observations.append(obs)
        self._prune(obs.timestamp)

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self.observations and self.observations[0].timestamp < cutoff:
            self.observations.popleft()

    def attach_late_pose(self, camera_id: str, track_id: str, frame_id: int, snapshot: PoseSnapshot) -> bool:
        for obs in reversed(self.observations):
            if obs.camera_id == camera_id and obs.track_id == track_id and obs.frame_id == frame_id:
                obs.pose = snapshot
                return True
            if obs.frame_id < frame_id and obs.camera_id == camera_id:
                break
        return False

    # -------------------------------------------------------------------
    # Lectura: ventanas
    # -------------------------------------------------------------------

    def recent(self, now: float, seconds: float, same_camera: bool) -> List[Observation]:
        cutoff = now - seconds
        result: List[Observation] = []
        for obs in reversed(self.observations):
            if obs.timestamp < cutoff:
                break
            if same_camera and obs.camera_id != self.camera_id:
                break
            result.append(obs)
        result.reverse()
        return result

    def latest(self) -> Optional[Observation]:
        return self.observations[-1] if self.observations else None

    def latest_pose(self, now: float, max_age: float) -> Optional[PoseSnapshot]:
        for obs in reversed(self.observations):
            if now - obs.timestamp > max_age:
                break
            if obs.pose is not None:
                return obs.pose
        return None

    def posture_median(self, attribute: str, now: float, seconds: float) -> Optional[float]:
        """Mediana robusta de un rasgo postural (filtra jitter de keypoints)."""
        values = [
            getattr(obs.pose, attribute)
            for obs in self.recent(now, seconds, same_camera=False)
            if obs.pose is not None and getattr(obs.pose, attribute) is not None
        ]
        return _median(values)

    # -------------------------------------------------------------------
    # Lectura: movimiento (normalizado por altura corporal, invariante a escala)
    # -------------------------------------------------------------------

    def _reference_height(self, observations: List[Observation]) -> float:
        heights = [o.body_height for o in observations if o.body_height > 1.0]
        return _median(heights) or 1.0

    def immobility_duration(self, now: float, tolerance_body_heights: float) -> float:
        """Segundos continuos en los que el centro se mantuvo dentro de una
        tolerancia (en alturas corporales) respecto a la posicion actual."""
        observations = self.recent(now, self.window_seconds, same_camera=True)
        if len(observations) < 2:
            return 0.0
        reference = observations[-1].center
        tolerance = tolerance_body_heights * self._reference_height(observations)
        earliest = observations[-1].timestamp
        for obs in reversed(observations):
            if _distance(obs.center, reference) > tolerance:
                break
            earliest = obs.timestamp
        return observations[-1].timestamp - earliest

    def motion_variance(self, now: float, seconds: float) -> Optional[float]:
        observations = self.recent(now, seconds, same_camera=True)
        if len(observations) < 2:
            return None
        height = self._reference_height(observations)
        centers = np.array([o.center for o in observations], dtype=np.float64) / height
        return float(np.linalg.norm(np.std(centers, axis=0)))

    def dispersion_radius(self, now: float, seconds: float) -> Optional[float]:
        """Radio de giro de la trayectoria (alturas corporales)."""
        observations = self.recent(now, seconds, same_camera=True)
        if len(observations) < 2:
            return None
        height = self._reference_height(observations)
        centers = np.array([o.center for o in observations], dtype=np.float64) / height
        return float(np.sqrt(np.mean(np.sum((centers - centers.mean(axis=0)) ** 2, axis=1))))

    def dwell_seconds(self, now: float) -> float:
        return 0.0 if self.zone_entry_time is None else max(0.0, now - self.zone_entry_time)

    # -------------------------------------------------------------------
    # Fusion de contextos (ReID externo)
    # -------------------------------------------------------------------

    def absorb(self, other: "SubjectContext") -> None:
        merged = sorted(list(self.observations) + list(other.observations), key=lambda o: o.timestamp)
        self.observations = deque(merged, maxlen=self.observations.maxlen)
        self.linked_tracks |= other.linked_tracks
        if other.last_seen > self.last_seen:
            self.last_seen = other.last_seen
            self.camera_id, self.track_id, self.session_id = other.camera_id, other.track_id, other.session_id
            self.zone_id, self.zone_entry_time = other.zone_id, other.zone_entry_time
        if other.presence == PresenceState.PRESENT:
            self.presence, self.lost_since = PresenceState.PRESENT, None
        for behavior_type, episode in other.episodes.items():
            mine = self.episodes.get(behavior_type)
            if mine is None or not mine.is_open or (
                episode.is_open and (episode.start_time or math.inf) < (mine.start_time or math.inf)
            ):
                self.episodes[behavior_type] = episode


class IdentityResolver:
    """Mapea (camera_id, track_id) -> subject_id. Por defecto la identidad
    es local a la camara. Un sistema externo de ReID puede invocar link()
    para declarar que varios tracks son el mismo individuo; Behavior NO
    implementa ReID, solo consume la asociacion."""

    def __init__(self) -> None:
        self._links: Dict[Tuple[str, str], str] = {}

    @staticmethod
    def local_id(camera_id: str, track_id: str) -> str:
        return f"{camera_id}:{track_id}"

    def resolve(self, camera_id: str, track_id: str) -> str:
        return self._links.get((camera_id, track_id), self.local_id(camera_id, track_id))

    def link(self, camera_id: str, track_id: str, subject_id: str) -> None:
        self._links[(camera_id, track_id)] = subject_id

    def forget(self, subject_id: str) -> None:
        for key in [k for k, v in self._links.items() if v == subject_id]:
            del self._links[key]


# =============================================================================
# Fusion de señales con pesos dinamicos (observabilidad parcial)
# =============================================================================

@dataclass
class SignalReading:
    name: str
    raw_value: Optional[float]
    score: Optional[float]  # membresia difusa [0,1]; None = señal no observable
    weight: float


@dataclass
class FusionResult:
    score: float       # promedio ponderado SOLO de las señales disponibles
    coverage: float    # fraccion del peso total que fue observable
    used: List[str]
    missing: List[str]
    per_signal: Dict[str, Optional[float]]


def fuse_signals(readings: List[SignalReading]) -> FusionResult:
    """Dynamic weighting: los pesos de las señales ausentes se redistribuyen
    entre las presentes (renormalizacion). 'coverage' cuantifica cuanta de
    la evidencia ideal fue realmente observable, y los modulos la usan para
    limitar cuan lejos puede escalar un comportamiento."""
    total_weight = sum(r.weight for r in readings) or 1.0
    available = [r for r in readings if r.score is not None]
    available_weight = sum(r.weight for r in available)
    score = sum(r.weight * r.score for r in available) / available_weight if available_weight > 0 else 0.0  # type: ignore[operator]
    return FusionResult(
        score=float(score),
        coverage=float(available_weight / total_weight),
        used=[r.name for r in available],
        missing=[r.name for r in readings if r.score is None],
        per_signal={r.name: _round(r.score) for r in readings},
    )


# =============================================================================
# Contrato de salida
# =============================================================================

@dataclass
class BehaviorResult:
    camera_id: str
    session_id: str
    track_id: str
    subject_id: str
    timestamp: float
    behavior_type: str
    state: BehaviorPhase
    start_time: float
    duration_seconds: float
    last_update: float
    evidence: Dict[str, object]
    evidence_strength: float
    observability: float
    episode_id: str
    zone_id: Optional[str]
    description: str
    result_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict[str, object]:
        return {
            "result_id": self.result_id,
            "episode_id": self.episode_id,
            "camera_id": self.camera_id,
            "session_id": self.session_id,
            "track_id": self.track_id,
            "subject_id": self.subject_id,
            "timestamp": self.timestamp,
            "behavior_type": self.behavior_type,
            "state": self.state.value,
            "start_time": self.start_time,
            "duration": round(self.duration_seconds, 3),
            "last_update": self.last_update,
            "zone_id": self.zone_id,
            "evidence_strength": round(self.evidence_strength, 4),
            "observability": round(self.observability, 4),
            "evidence": self.evidence,
            "description": self.description,
        }


# =============================================================================
# Modulos de comportamiento
# =============================================================================

@dataclass
class Evaluation:
    score: float = 0.0
    coverage: float = 0.0
    gate_ok: bool = True
    applicable: bool = True
    evidence: Dict[str, object] = field(default_factory=dict)

    @staticmethod
    def not_applicable(reason: str) -> "Evaluation":
        return Evaluation(applicable=False, gate_ok=False, evidence={"not_applicable_reason": reason})


@dataclass(frozen=True)
class ModuleContext:
    """Contexto espacial que el servicio entrega a cada modulo."""

    zone: Optional[Zone]
    product_zones: List[Zone]


class BaseBehaviorModule(abc.ABC):
    """Base de todos los comportamientos. Las subclases SOLO implementan
    evaluate() (que señales mirar y como puntuarlas). La maquina de estados
    NORMAL -> POSSIBLE -> SUSTAINED -> FINISHED, la histeresis, el
    debounce de salida, los heartbeats y la construccion del contrato de
    salida viven aqui y son identicos para todos los modulos."""

    behavior_type: str = "base"
    label: str = "comportamiento"
    feminine: bool = False
    applicable_classes: FrozenSet[str] = frozenset({"person"})

    common_defaults: Dict[str, object] = {
        "enabled": True,
        "enter_threshold": 0.6,      # score para abrir un episodio
        "exit_threshold": 0.45,      # histeresis: score minimo para mantenerlo
        "min_coverage": 0.4,         # observabilidad minima para considerar la evidencia
        "sustain_min_coverage": 0.5, # observabilidad minima para escalar a SUSTAINED
        "sustain_seconds": 15.0,
        "release_seconds": 3.0,      # inactividad continua requerida para cerrar
        "emit_interval_seconds": 1.0,
    }
    module_defaults: Dict[str, object] = {}

    def __init__(self, overrides: Optional[Dict[str, object]] = None) -> None:
        self._params: Dict[str, object] = {**self.common_defaults, **self.module_defaults, **(overrides or {})}

    def resolve_params(self, zone: Optional[Zone]) -> Dict[str, object]:
        if zone is None or self.behavior_type not in zone.behavior_overrides:
            return self._params
        return {**self._params, **zone.behavior_overrides[self.behavior_type]}

    @abc.abstractmethod
    def evaluate(self, ctx: SubjectContext, now: float, params: Dict[str, object], mctx: ModuleContext) -> Evaluation:
        raise NotImplementedError

    # -------------------------------------------------------------------
    # Maquina de estados (comun)
    # -------------------------------------------------------------------

    def step(self, ctx: SubjectContext, now: float, mctx: ModuleContext) -> List[BehaviorResult]:
        params = self.resolve_params(mctx.zone)
        episode = ctx.episodes.setdefault(self.behavior_type, BehaviorEpisode())

        if ctx.class_name not in self.applicable_classes:
            evaluation = Evaluation.not_applicable("class_not_applicable")
        elif not params["enabled"]:
            evaluation = Evaluation.not_applicable("disabled_in_zone")
        else:
            evaluation = self.evaluate(ctx, now, params, mctx)

        threshold = float(params["enter_threshold"] if episode.phase == BehaviorPhase.NORMAL else params["exit_threshold"])  # type: ignore[arg-type]
        active = (
            evaluation.applicable
            and evaluation.gate_ok
            and evaluation.coverage >= float(params["min_coverage"])  # type: ignore[arg-type]
            and evaluation.score >= threshold
        )
        episode.last_score, episode.last_coverage = evaluation.score, evaluation.coverage

        results: List[BehaviorResult] = []
        if episode.phase == BehaviorPhase.NORMAL:
            if active:
                episode.phase = BehaviorPhase.POSSIBLE
                episode.start_time = now
                episode.last_active_time = now
                episode.peak_score = evaluation.score
                episode.trigger_evidence = evaluation.evidence
                results.append(self._emit(ctx, episode, now, evaluation.evidence, transition=True))
            return results

        if active:
            episode.last_active_time = now
            episode.peak_score = max(episode.peak_score, evaluation.score)
            episode.trigger_evidence = evaluation.evidence
            can_sustain = evaluation.coverage >= float(params["sustain_min_coverage"])  # type: ignore[arg-type]
            if (
                episode.phase == BehaviorPhase.POSSIBLE
                and now - (episode.start_time or now) >= float(params["sustain_seconds"])  # type: ignore[arg-type]
                and can_sustain
            ):
                episode.phase = BehaviorPhase.SUSTAINED
                results.append(self._emit(ctx, episode, now, evaluation.evidence, transition=True))
                return results
        elif now - (episode.last_active_time or now) >= float(params["release_seconds"]):  # type: ignore[arg-type]
            results.append(self.finish(ctx, episode, now, "signals_released", evaluation.evidence))
            return results

        if now - (episode.last_emit_time or 0.0) >= float(params["emit_interval_seconds"]):  # type: ignore[arg-type]
            evidence = evaluation.evidence if active else {**episode.trigger_evidence, "currently_active": False}
            results.append(self._emit(ctx, episode, now, evidence, transition=False))
        return results

    def finish(
        self,
        ctx: SubjectContext,
        episode: BehaviorEpisode,
        now: float,
        reason: str,
        current_evidence: Optional[Dict[str, object]] = None,
    ) -> BehaviorResult:
        """Cierra el episodio (FINISHED) y lo reinicia a NORMAL."""
        episode.phase = BehaviorPhase.FINISHED
        evidence = {
            **episode.trigger_evidence,
            "finish_reason": reason,
            "peak_evidence_strength": _round(episode.peak_score),
        }
        if current_evidence:
            evidence["evidence_at_finish"] = current_evidence
        result = self._emit(ctx, episode, now, evidence, transition=True)
        ctx.episodes[self.behavior_type] = BehaviorEpisode()
        return result

    def _emit(
        self,
        ctx: SubjectContext,
        episode: BehaviorEpisode,
        now: float,
        evidence: Dict[str, object],
        transition: bool,
    ) -> BehaviorResult:
        start = episode.start_time if episode.start_time is not None else now
        end = episode.last_active_time if episode.phase == BehaviorPhase.FINISHED and episode.last_active_time else now
        duration = max(0.0, end - start)
        episode.last_emit_time = now

        if transition:
            logger.info(
                "Track %s entro en %s (%s) | subject=%s duracion=%.1fs",
                ctx.track_id, episode.phase.value, self.behavior_type, ctx.subject_id, duration,
            )

        return BehaviorResult(
            camera_id=ctx.camera_id,
            session_id=ctx.session_id,
            track_id=ctx.track_id,
            subject_id=ctx.subject_id,
            timestamp=now,
            behavior_type=self.behavior_type,
            state=episode.phase,
            start_time=start,
            duration_seconds=duration,
            last_update=now,
            evidence={**evidence, "track_recoveries": episode.recoveries, "linked_tracks": len(ctx.linked_tracks)},
            evidence_strength=episode.last_score,
            observability=episode.last_coverage,
            episode_id=episode.episode_id,
            zone_id=ctx.zone_id,
            description=self._describe(episode.phase, duration),
        )

    def _describe(self, phase: BehaviorPhase, duration: float) -> str:
        suffix = "a" if self.feminine else "o"
        qualifier = {
            BehaviorPhase.POSSIBLE: f"posible",
            BehaviorPhase.SUSTAINED: f"sostenid{suffix}",
            BehaviorPhase.FINISHED: f"finalizad{suffix}",
        }.get(phase, "")
        return f"Evidencia de {self.label} {qualifier} durante {duration:.1f} s"


class SleepingBehavior(BaseBehaviorModule):
    """Somnolencia: postura de cabeza caida / torso vencido + inmovilidad
    sostenida. Diseñado para observabilidad parcial:

    - eye_closure (ojos) suele faltar (vista de espalda / sin modulo
      facial). Su peso se redistribuye automaticamente.
    - head_angle / neck_height usan la nariz o, en su defecto, las orejas.
    - La inmovilidad es ademas una compuerta (gate): sin quietud minima
      no hay evidencia, por mas que la postura parezca de sueño.
    - Con observabilidad baja (ej. solo inmovilidad + torso) la evidencia
      nunca escala a SUSTAINED.
    """

    behavior_type = "somnolence"
    label = "somnolencia"
    feminine = True
    module_defaults = {
        "sustain_seconds": 15.0,
        "release_seconds": 3.0,
        "min_coverage": 0.4,
        "sustain_min_coverage": 0.45,
        "immobility_gate_seconds": 3.0,
        "immobility_tolerance_body_heights": 0.15,
        "posture_window_seconds": 3.0,
        "weight_eye_closure": 0.30,
        "weight_head_angle": 0.20,
        "weight_neck_height": 0.15,
        "weight_torso_inclination": 0.10,
        "weight_immobility": 0.25,
    }

    def evaluate(self, ctx: SubjectContext, now: float, params: Dict[str, object], mctx: ModuleContext) -> Evaluation:
        window = float(params["posture_window_seconds"])  # type: ignore[arg-type]
        eye_ratio = ctx.posture_median("eye_aspect_ratio", now, window)
        head_angle = ctx.posture_median("head_angle_deg", now, window)
        neck_ratio = ctx.posture_median("neck_height_ratio", now, window)
        torso = ctx.posture_median("torso_inclination_deg", now, window)
        immobility = ctx.immobility_duration(now, float(params["immobility_tolerance_body_heights"]))  # type: ignore[arg-type]
        latest_pose = ctx.latest_pose(now, window)

        fusion = fuse_signals([
            SignalReading("eye_closure", eye_ratio, ramp(eye_ratio, 0.25, 0.15), float(params["weight_eye_closure"])),  # type: ignore[arg-type]
            SignalReading("head_angle", head_angle, ramp(head_angle, 25.0, 60.0), float(params["weight_head_angle"])),  # type: ignore[arg-type]
            SignalReading("neck_height", neck_ratio, ramp(neck_ratio, 0.30, 0.10), float(params["weight_neck_height"])),  # type: ignore[arg-type]
            SignalReading(
                "torso_inclination", torso,
                ramp(None if torso is None else abs(torso), 20.0, 60.0),
                float(params["weight_torso_inclination"]),  # type: ignore[arg-type]
            ),
            SignalReading("immobility", immobility, ramp(immobility, 2.0, 10.0), float(params["weight_immobility"])),  # type: ignore[arg-type]
        ])

        evidence: Dict[str, object] = {
            "head_angle_deg": _round(head_angle, 2),
            "neck_height_ratio": _round(neck_ratio),
            "torso_inclination_deg": _round(torso, 2),
            "eye_aspect_ratio": _round(eye_ratio),
            "eyes_visible": eye_ratio is not None,
            "head_reference": latest_pose.head_reference if latest_pose else None,
            "immobility_duration": round(immobility, 2),
            "motion_variance": _round(ctx.motion_variance(now, 10.0), 4),
            "signal_scores": fusion.per_signal,
            "signals_used": fusion.used,
            "signals_missing": fusion.missing,
        }
        return Evaluation(
            score=fusion.score,
            coverage=fusion.coverage,
            gate_ok=immobility >= float(params["immobility_gate_seconds"]),  # type: ignore[arg-type]
            evidence=evidence,
        )


class LoiteringBehavior(BaseBehaviorModule):
    """Merodeo: permanencia prolongada en una misma zona con poco
    desplazamiento. Depende del contexto: una zona puede desactivarlo
    (ej. zona_caja, donde esperar es normal) o cambiar sus umbrales."""

    behavior_type = "loitering"
    label = "merodeo"
    feminine = False
    module_defaults = {
        "min_dwell_seconds": 45.0,
        "sustain_seconds": 60.0,
        "release_seconds": 5.0,
        "min_coverage": 0.5,
        "sustain_min_coverage": 0.5,
        "max_dispersion_body_heights": 1.5,
        "still_dispersion_body_heights": 0.5,
        "dispersion_window_seconds": 20.0,
        "emit_interval_seconds": 2.0,
    }

    def evaluate(self, ctx: SubjectContext, now: float, params: Dict[str, object], mctx: ModuleContext) -> Evaluation:
        min_dwell = float(params["min_dwell_seconds"])  # type: ignore[arg-type]
        dwell = ctx.dwell_seconds(now)
        # Desplazamiento RECIENTE: si la persona empieza a caminar, el
        # episodio se libera aunque siga dentro de la misma zona.
        dispersion_window = min(max(dwell, 1.0), float(params["dispersion_window_seconds"]))  # type: ignore[arg-type]
        dispersion = ctx.dispersion_radius(now, dispersion_window)

        fusion = fuse_signals([
            SignalReading("dwell_time", dwell, ramp(dwell, 0.5 * min_dwell, min_dwell), 0.5),
            SignalReading(
                "low_displacement", dispersion,
                ramp(dispersion, float(params["max_dispersion_body_heights"]), float(params["still_dispersion_body_heights"])),  # type: ignore[arg-type]
                0.5,
            ),
        ])
        evidence: Dict[str, object] = {
            "zone_id": ctx.zone_id,
            "zone_name": mctx.zone.name if mctx.zone else None,
            "dwell_time": round(dwell, 2),
            "dispersion_body_heights": _round(dispersion),
            "motion_variance": _round(ctx.motion_variance(now, 10.0), 4),
            "signal_scores": fusion.per_signal,
        }
        return Evaluation(score=fusion.score, coverage=fusion.coverage, gate_ok=dwell >= min_dwell, evidence=evidence)


class ProductInteractionBehavior(BaseBehaviorModule):
    """Interaccion con producto: proximidad de las MANOS (muñecas) a un ROI
    de producto, con apoyo secundario en la proximidad del cuerpo. Sin
    manos visibles la cobertura queda por debajo del minimo: no se genera
    evidencia de interaccion solo porque alguien este parado cerca."""

    behavior_type = "product_interaction"
    label = "interaccion con producto"
    feminine = True
    module_defaults = {
        "sustain_seconds": 2.0,
        "release_seconds": 1.5,
        "min_coverage": 0.5,
        "sustain_min_coverage": 0.7,
        "hand_contact_margin_body_heights": 0.15,
        "body_near_body_heights": 0.3,
        "body_far_body_heights": 1.0,
        "pose_max_age_seconds": 1.0,
        "emit_interval_seconds": 1.0,
    }

    def evaluate(self, ctx: SubjectContext, now: float, params: Dict[str, object], mctx: ModuleContext) -> Evaluation:
        if not mctx.product_zones:
            return Evaluation.not_applicable("no_product_roi_configured")
        latest = ctx.latest()
        if latest is None:
            return Evaluation.not_applicable("no_observations")

        height = max(latest.body_height, 1.0)
        pose = ctx.latest_pose(now, float(params["pose_max_age_seconds"]))  # type: ignore[arg-type]
        wrists = {}
        if pose is not None:
            for side in ("left", "right"):
                point = pose.keypoints.get(f"{side}_wrist")
                if point is not None:
                    wrists[side] = point

        best: Optional[Tuple[FusionResult, Dict[str, object]]] = None
        for zone in mctx.product_zones:
            hand_distance: Optional[float] = None
            hand_side: Optional[str] = None
            for side, point in wrists.items():
                d = point_polygon_distance(point, zone.polygon) / height
                if hand_distance is None or d < hand_distance:
                    hand_distance, hand_side = d, side
            body_distance = point_polygon_distance(latest.center, zone.polygon) / height

            fusion = fuse_signals([
                SignalReading(
                    "hand_contact", hand_distance,
                    ramp(hand_distance, float(params["hand_contact_margin_body_heights"]), 0.0),  # type: ignore[arg-type]
                    0.75,
                ),
                SignalReading(
                    "body_proximity", body_distance,
                    ramp(body_distance, float(params["body_far_body_heights"]), float(params["body_near_body_heights"])),  # type: ignore[arg-type]
                    0.25,
                ),
            ])
            evidence: Dict[str, object] = {
                "roi_id": zone.zone_id,
                "roi_name": zone.name,
                "hand": hand_side,
                "hands_visible": bool(wrists),
                "hand_distance_body_heights": _round(hand_distance),
                "body_distance_body_heights": _round(body_distance),
                "signal_scores": fusion.per_signal,
            }
            if best is None or (fusion.score * fusion.coverage) > (best[0].score * best[0].coverage):
                best = (fusion, evidence)

        assert best is not None
        return Evaluation(score=best[0].score, coverage=best[0].coverage, evidence=best[1])


# =============================================================================
# Validacion de entradas
# =============================================================================

def validate_tracking_result(payload: Optional[Dict[str, object]]) -> None:
    if not payload or not isinstance(payload, dict):
        raise InputValidationError("TrackingResult nulo o vacio")
    for key in ("camera_id", "session_id", "frame_id", "timestamp", "tracks"):
        if key not in payload:
            raise InputValidationError(f"Falta campo en TrackingResult: {key}")
    if not payload["camera_id"] or not payload["session_id"]:
        raise InputValidationError("camera_id/session_id invalido")
    if payload["timestamp"] is None or payload["timestamp"] <= 0:  # type: ignore[operator]
        raise InputValidationError("timestamp invalido")
    if not isinstance(payload["tracks"], list):
        raise InputValidationError("'tracks' debe ser una lista")
    for track in payload["tracks"]:
        if not isinstance(track, dict):
            raise InputValidationError("track invalido")
        for key in ("track_id", "class_name", "bbox", "track_state"):
            if key not in track:
                raise InputValidationError(f"Falta campo en track: {key}")
        bbox = track["bbox"]
        if not isinstance(bbox, dict) or any(k not in bbox for k in ("x1", "y1", "x2", "y2")):
            raise InputValidationError("bbox invalido")


def validate_pose_result(payload: Optional[Dict[str, object]]) -> None:
    if not payload or not isinstance(payload, dict):
        raise InputValidationError("SpecializedAIResult nulo o vacio")
    for key in ("camera_id", "frame_id", "poses"):
        if key not in payload:
            raise InputValidationError(f"Falta campo en SpecializedAIResult: {key}")
    if not isinstance(payload["poses"], list):
        raise InputValidationError("'poses' debe ser una lista")
    for pose in payload["poses"]:
        if not isinstance(pose, dict) or "track_id" not in pose or "status" not in pose:
            raise InputValidationError("pose invalida")


# =============================================================================
# Cola acotada (descarta lo mas antiguo)
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
# Configuracion y metricas
# =============================================================================

@dataclass(frozen=True)
class BehaviorConfig:
    window_seconds: float = 60.0            # ventana deslizante de memoria por sujeto
    max_observations: int = 1200            # tope duro por sujeto (RAM acotada)
    lost_grace_seconds: float = 5.0         # gracia ante LOST antes de descartar
    fusion_wait_ms: float = 150.0           # espera maxima de la pose de un frame
    stale_wall_timeout_seconds: float = 120.0
    target_classes: FrozenSet[str] = frozenset({"person"})
    input_queue_size: int = 256
    output_queue_size: int = 1000
    max_pending_frames: int = 64
    pose_buffer_size: int = 256

    @staticmethod
    def from_env() -> "BehaviorConfig":
        classes = os.getenv("BEHAVIOR_TARGET_CLASSES", "person")
        return BehaviorConfig(
            window_seconds=float(os.getenv("BEHAVIOR_WINDOW_SECONDS", "60")),
            max_observations=int(os.getenv("BEHAVIOR_MAX_OBSERVATIONS", "1200")),
            lost_grace_seconds=float(os.getenv("BEHAVIOR_LOST_GRACE_SECONDS", "5")),
            fusion_wait_ms=float(os.getenv("BEHAVIOR_FUSION_WAIT_MS", "150")),
            stale_wall_timeout_seconds=float(os.getenv("BEHAVIOR_STALE_TIMEOUT_SECONDS", "120")),
            target_classes=frozenset(c.strip() for c in classes.split(",") if c.strip()),
            input_queue_size=int(os.getenv("BEHAVIOR_INPUT_QUEUE_SIZE", "256")),
            output_queue_size=int(os.getenv("BEHAVIOR_OUTPUT_QUEUE_SIZE", "1000")),
        )


@dataclass
class BehaviorMetrics:
    tracking_received: int = 0
    pose_received: int = 0
    frames_processed: int = 0
    frames_fused_with_pose: int = 0
    frames_tracking_only: int = 0
    late_poses_attached: int = 0
    dropped_invalid: int = 0
    dropped_stale: int = 0
    dropped_backpressure: int = 0
    results_emitted: int = 0
    transitions: Dict[str, int] = field(default_factory=dict)
    subjects_collected: int = 0
    total_processing_seconds: float = 0.0

    @property
    def avg_processing_latency_ms(self) -> float:
        if self.frames_processed == 0:
            return 0.0
        return round(self.total_processing_seconds / self.frames_processed * 1000.0, 3)

    def to_dict(self) -> Dict[str, object]:
        return {
            "tracking_received": self.tracking_received,
            "pose_received": self.pose_received,
            "frames_processed": self.frames_processed,
            "frames_fused_with_pose": self.frames_fused_with_pose,
            "frames_tracking_only": self.frames_tracking_only,
            "late_poses_attached": self.late_poses_attached,
            "dropped_invalid": self.dropped_invalid,
            "dropped_stale": self.dropped_stale,
            "dropped_backpressure": self.dropped_backpressure,
            "results_emitted": self.results_emitted,
            "transitions": dict(self.transitions),
            "subjects_collected": self.subjects_collected,
            "avg_processing_latency_ms": self.avg_processing_latency_ms,
        }


# =============================================================================
# Servicio principal
# =============================================================================

class BehaviorService:
    """Punto de entrada del Servicio 6.

    Flujo (hilo worker unico, mutador exclusivo del estado):
      submit_tracking_result / submit_pose_result -> cola acotada ->
      fusion por (camera_id, frame_id) con espera maxima -> actualizacion de
      la ventana temporal del sujeto -> step() de cada modulo ->
      BehaviorResult (evidencia) en la cola de salida.
    """

    def __init__(
        self,
        config: Optional[BehaviorConfig] = None,
        modules: Optional[List[BaseBehaviorModule]] = None,
        zones: Optional[ZoneRegistry] = None,
    ) -> None:
        self.config = config or BehaviorConfig.from_env()
        self.zones = zones or ZoneRegistry()
        self.modules: List[BaseBehaviorModule] = modules or [
            SleepingBehavior(),
            LoiteringBehavior(),
            ProductInteractionBehavior(),
        ]
        self.metrics = BehaviorMetrics()
        self.identity = IdentityResolver()

        self._subjects: Dict[str, SubjectContext] = {}
        self._state_lock = threading.RLock()

        self._input_queue = BoundedQueue(self.config.input_queue_size, on_drop=self._on_input_drop)
        self._output_queue = BoundedQueue(self.config.output_queue_size, on_drop=self._on_output_drop)

        self._pending: "OrderedDict[Tuple[str, int], Tuple[Dict[str, object], float]]" = OrderedDict()
        self._pose_buffer: "OrderedDict[Tuple[str, int], Dict[str, object]]" = OrderedDict()
        self._last_frame_id: Dict[str, int] = {}
        self._last_sweep_wall = 0.0

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_wall = time.time()

        zones_file = os.getenv("BEHAVIOR_ZONES_FILE")
        if zones_file:
            self.zones.load_json(zones_file)

    # -------------------------------------------------------------------
    # Ciclo de vida
    # -------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        logger.info("Iniciando BehaviorService | modulos=%s", [m.behavior_type for m in self.modules])
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="BehaviorWorker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        logger.info("Deteniendo BehaviorService...")
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        with self._state_lock:
            self._pending.clear()
            self._pose_buffer.clear()
            self._subjects.clear()
        self._input_queue.clear()
        logger.info("BehaviorService detenido")

    # -------------------------------------------------------------------
    # Entrada
    # -------------------------------------------------------------------

    def submit_tracking_result(self, payload: Dict[str, object]) -> None:
        self.metrics.tracking_received += 1
        self._input_queue.put(("tracking", payload))

    def submit_pose_result(self, payload: Dict[str, object]) -> None:
        self.metrics.pose_received += 1
        self._input_queue.put(("pose", payload))

    def set_zones(self, camera_id: str, zones: List[Zone]) -> None:
        self.zones.set_zones(camera_id, zones)

    def link_identity(self, camera_id: str, track_id: str, subject_id: str) -> None:
        """Hook para un sistema EXTERNO de ReID: declara que (camera_id,
        track_id) pertenece al individuo subject_id. Si ya existia memoria
        bajo la identidad local, se fusiona con la del sujeto global."""
        with self._state_lock:
            previous_id = self.identity.resolve(camera_id, track_id)
            self.identity.link(camera_id, track_id, subject_id)
            if previous_id == subject_id:
                return
            previous_ctx = self._subjects.pop(previous_id, None)
            if previous_ctx is None:
                return
            target = self._subjects.get(subject_id)
            if target is None:
                previous_ctx.subject_id = subject_id
                self._subjects[subject_id] = previous_ctx
            else:
                target.absorb(previous_ctx)
            logger.info("Identidad vinculada por ReID externo | %s -> %s", previous_id, subject_id)

    # -------------------------------------------------------------------
    # Salida
    # -------------------------------------------------------------------

    def get_next_result(self, timeout: float = 1.0) -> Optional[Dict[str, object]]:
        return self._output_queue.get(timeout=timeout)  # type: ignore[return-value]

    def result_stream(self) -> Generator[Dict[str, object], None, None]:
        while not self._stop_event.is_set():
            result = self.get_next_result(timeout=1.0)
            if result is not None:
                yield result

    def health_check(self) -> Dict[str, object]:
        with self._state_lock:
            subjects = list(self._subjects.values())
            open_episodes: Dict[str, int] = {}
            for ctx in subjects:
                for behavior_type, episode in ctx.episodes.items():
                    if episode.is_open:
                        open_episodes[behavior_type] = open_episodes.get(behavior_type, 0) + 1
            pending = len(self._pending)
        return {
            "service_active": self._thread is not None and self._thread.is_alive(),
            "modules_loaded": [m.behavior_type for m in self.modules],
            "subjects_analyzed": len(subjects),
            "subjects_recovering": sum(1 for s in subjects if s.presence == PresenceState.RECOVERING),
            "open_episodes": open_episodes,
            "avg_processing_latency_ms": self.metrics.avg_processing_latency_ms,
            "input_queue_usage": f"{self._input_queue.qsize()}/{self._input_queue.max_size}",
            "output_queue_usage": f"{self._output_queue.qsize()}/{self._output_queue.max_size}",
            "pending_fusion_frames": pending,
            "uptime_seconds": round(time.time() - self._start_wall, 3),
            "metrics": self.metrics.to_dict(),
        }

    # -------------------------------------------------------------------
    # Worker
    # -------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._input_queue.get(timeout=0.02)
                with self._state_lock:
                    if item is not None:
                        kind, payload = item  # type: ignore[misc]
                        if kind == "tracking":
                            self._enqueue_tracking(payload)
                        else:
                            self._handle_pose(payload)
                    self._flush_pending()
                    if time.time() - self._last_sweep_wall >= 1.0:
                        self._sweep_stale_streams()
                        self._last_sweep_wall = time.time()
            except Exception:
                logger.exception("Error inesperado en BehaviorService (se continua)")

    # -------------------------------------------------------------------
    # Fusion tracking + pose
    # -------------------------------------------------------------------

    def _enqueue_tracking(self, payload: Dict[str, object]) -> None:
        try:
            validate_tracking_result(payload)
        except InputValidationError as exc:
            self.metrics.dropped_invalid += 1
            _rate_limited_logger.log("invalid_tracking", logging.WARNING, f"TrackingResult invalidos (%d): {exc}")
            return

        camera_id, frame_id = str(payload["camera_id"]), int(payload["frame_id"])  # type: ignore[arg-type]
        last = self._last_frame_id.get(camera_id)
        if last is not None and frame_id <= last:
            self.metrics.dropped_stale += 1
            _rate_limited_logger.log(f"stale_{camera_id}", logging.WARNING, f"[{camera_id}] Frames atrasados descartados (%d)")
            return
        self._pending[(camera_id, frame_id)] = (payload, time.time())

        while len(self._pending) > self.config.max_pending_frames:
            (cam, fid), (oldest, _) = next(iter(self._pending.items()))
            del self._pending[(cam, fid)]
            self._process_frame(oldest, self._pose_buffer.pop((cam, fid), None))

    def _handle_pose(self, payload: Dict[str, object]) -> None:
        try:
            validate_pose_result(payload)
        except InputValidationError as exc:
            self.metrics.dropped_invalid += 1
            _rate_limited_logger.log("invalid_pose", logging.WARNING, f"SpecializedAIResult invalidos (%d): {exc}")
            return

        camera_id, frame_id = str(payload["camera_id"]), int(payload["frame_id"])  # type: ignore[arg-type]
        last = self._last_frame_id.get(camera_id)
        if (camera_id, frame_id) not in self._pending and last is not None and frame_id <= last:
            self._attach_late_pose(payload)
            return

        self._pose_buffer[(camera_id, frame_id)] = payload
        while len(self._pose_buffer) > self.config.pose_buffer_size:
            self._pose_buffer.popitem(last=False)

    def _attach_late_pose(self, payload: Dict[str, object]) -> None:
        """La pose llego despues de procesar su frame: se adjunta
        retroactivamente a la observacion correspondiente para que las
        siguientes evaluaciones la aprovechen."""
        camera_id, frame_id = str(payload["camera_id"]), int(payload["frame_id"])  # type: ignore[arg-type]
        for pose in payload["poses"]:  # type: ignore[union-attr]
            snapshot = build_pose_snapshot(pose)
            if snapshot is None:
                continue
            track_id = str(pose["track_id"])
            ctx = self._subjects.get(self.identity.resolve(camera_id, track_id))
            if ctx is not None and ctx.attach_late_pose(camera_id, track_id, frame_id, snapshot):
                self.metrics.late_poses_attached += 1

    def _flush_pending(self) -> None:
        """Procesa en orden los frames cuya pose ya llego o cuya espera
        vencio. Un frame no listo bloquea solo a los frames posteriores de
        SU camara (preserva el orden temporal por camara)."""
        now_wall = time.time()
        wait_seconds = self.config.fusion_wait_ms / 1000.0
        blocked: Set[str] = set()
        for key in list(self._pending.keys()):
            camera_id = key[0]
            if camera_id in blocked:
                continue
            payload, arrived = self._pending[key]
            if key in self._pose_buffer or now_wall - arrived >= wait_seconds:
                del self._pending[key]
                self._process_frame(payload, self._pose_buffer.pop(key, None))
            else:
                blocked.add(camera_id)

    # -------------------------------------------------------------------
    # Procesamiento de un frame
    # -------------------------------------------------------------------

    def _process_frame(self, payload: Dict[str, object], pose_payload: Optional[Dict[str, object]]) -> None:
        start = time.time()
        camera_id = str(payload["camera_id"])
        session_id = str(payload["session_id"])
        frame_id = int(payload["frame_id"])  # type: ignore[arg-type]
        now = float(payload["timestamp"])  # type: ignore[arg-type]
        self._last_frame_id[camera_id] = frame_id

        poses: Dict[str, Dict[str, object]] = {}
        if pose_payload is not None:
            poses = {str(p["track_id"]): p for p in pose_payload["poses"]}  # type: ignore[union-attr]
            self.metrics.frames_fused_with_pose += 1
        else:
            self.metrics.frames_tracking_only += 1

        product_zones = self.zones.product_zones(camera_id)
        results: List[BehaviorResult] = []

        for track in payload["tracks"]:  # type: ignore[union-attr]
            class_name = str(track["class_name"])
            if class_name not in self.config.target_classes:
                continue
            track_id = str(track["track_id"])
            state = str(track["track_state"])
            subject_id = self.identity.resolve(camera_id, track_id)
            ctx = self._subjects.get(subject_id)

            if state in _LOST_TRACK_STATES:
                if ctx is not None:
                    if state == "TERMINATED":
                        ctx.terminated = True
                    if ctx.presence == PresenceState.PRESENT:
                        ctx.presence, ctx.lost_since = PresenceState.RECOVERING, now
                        logger.info("Track %s en RECOVERING (tracking=%s), gracia de %.1fs",
                                    track_id, state, self.config.lost_grace_seconds)
                continue

            if state not in _OBSERVABLE_TRACK_STATES:
                continue

            observation = self._build_observation(track, camera_id, frame_id, now, poses.get(track_id))
            if observation is None:
                continue

            if ctx is None:
                ctx = SubjectContext(subject_id, class_name, self.config.window_seconds, self.config.max_observations)
                self._subjects[subject_id] = ctx
            elif ctx.presence == PresenceState.RECOVERING:
                self._resume(ctx, now)

            ctx.add_observation(observation, session_id)
            mctx = ModuleContext(zone=self.zones.get(camera_id, ctx.zone_id), product_zones=product_zones)
            for module in self.modules:
                try:
                    results.extend(module.step(ctx, now, mctx))
                except Exception:
                    logger.exception("Fallo el modulo %s para %s (se continua)", module.behavior_type, subject_id)

        results.extend(self._sweep_camera(camera_id, now))
        self._publish(results)

        self.metrics.frames_processed += 1
        self.metrics.total_processing_seconds += time.time() - start

    def _build_observation(
        self,
        track: Dict[str, object],
        camera_id: str,
        frame_id: int,
        now: float,
        pose: Optional[Dict[str, object]],
    ) -> Optional[Observation]:
        bbox = track["bbox"]
        try:
            x1, y1, x2, y2 = (float(bbox[k]) for k in ("x1", "y1", "x2", "y2"))  # type: ignore[index]
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
            return None
        feet = ((x1 + x2) / 2.0, y2)
        zone = self.zones.area_zone_for(camera_id, feet)
        return Observation(
            timestamp=now,
            frame_id=frame_id,
            camera_id=camera_id,
            track_id=str(track["track_id"]),
            center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
            feet=feet,
            body_height=y2 - y1,
            zone_id=zone.zone_id if zone else None,
            pose=build_pose_snapshot(pose) if pose is not None else None,
        )

    def _resume(self, ctx: SubjectContext, now: float) -> None:
        """El sujeto reaparecio dentro del periodo de gracia: los episodios
        abiertos continuan (no se resetean) y reciben una ventana de
        liberacion fresca."""
        logger.info("Track %s recuperado tras %.1fs, comportamientos preservados",
                    ctx.track_id, now - (ctx.lost_since or now))
        ctx.presence, ctx.lost_since, ctx.terminated = PresenceState.PRESENT, None, False
        for episode in ctx.episodes.values():
            if episode.is_open:
                episode.last_active_time = now
                episode.recoveries += 1

    # -------------------------------------------------------------------
    # Garbage collection
    # -------------------------------------------------------------------

    def _sweep_camera(self, camera_id: str, now: float) -> List[BehaviorResult]:
        """Aplica el periodo de gracia usando el tiempo de datos de la
        camara. Tracks que desaparecen sin pasar por LOST se tratan igual."""
        results: List[BehaviorResult] = []
        grace = self.config.lost_grace_seconds
        for subject_id, ctx in list(self._subjects.items()):
            if ctx.camera_id != camera_id:
                continue
            if ctx.presence == PresenceState.PRESENT and now - ctx.last_seen > grace:
                ctx.presence, ctx.lost_since = PresenceState.RECOVERING, ctx.last_seen
            if ctx.presence == PresenceState.RECOVERING and now - (ctx.lost_since or now) >= grace:
                reason = "track_terminated" if ctx.terminated else "track_lost_timeout"
                results.extend(self._collect(subject_id, ctx, now, reason))
        return results

    def _sweep_stale_streams(self) -> None:
        """Camaras que dejaron de enviar datos: el tiempo de datos no avanza,
        asi que se usa el reloj de pared como ultimo recurso."""
        now_wall = time.time()
        results: List[BehaviorResult] = []
        for subject_id, ctx in list(self._subjects.items()):
            if now_wall - ctx.last_wall_update > self.config.stale_wall_timeout_seconds:
                results.extend(self._collect(subject_id, ctx, ctx.last_seen, "stream_stale"))
        self._publish(results)

    def _collect(self, subject_id: str, ctx: SubjectContext, now: float, reason: str) -> List[BehaviorResult]:
        results: List[BehaviorResult] = []
        modules = {m.behavior_type: m for m in self.modules}
        for behavior_type, episode in list(ctx.episodes.items()):
            module = modules.get(behavior_type)
            if module is not None and episode.is_open:
                results.append(module.finish(ctx, episode, now, reason))
        del self._subjects[subject_id]
        self.identity.forget(subject_id)
        self.metrics.subjects_collected += 1
        logger.debug("Memoria liberada | subject=%s motivo=%s", subject_id, reason)
        return results

    # -------------------------------------------------------------------
    # Publicacion / backpressure
    # -------------------------------------------------------------------

    def _publish(self, results: List[BehaviorResult]) -> None:
        for result in results:
            key = f"{result.behavior_type}:{result.state.value}"
            self.metrics.transitions[key] = self.metrics.transitions.get(key, 0) + 1
            self.metrics.results_emitted += 1
            self._output_queue.put(result.to_dict())

    def _on_input_drop(self) -> None:
        self.metrics.dropped_backpressure += 1
        _rate_limited_logger.log("input_backpressure", logging.WARNING,
                                 "Cola de entrada de Behavior llena: descartando lo mas antiguo (%d)")

    def _on_output_drop(self) -> None:
        self.metrics.dropped_backpressure += 1
        _rate_limited_logger.log("output_backpressure", logging.WARNING,
                                 "Cola de salida de Behavior llena (consumidor lento): descartando lo mas antiguo (%d)")


# =============================================================================
# Smoke test manual: escenario sintetico con tiempo de datos simulado
# =============================================================================

def _demo_keypoints(x: float, y: float, head_dropped: bool, back_view: bool, wrist: Optional[Point] = None) -> Dict[str, object]:
    """Esqueleto sintetico relativo a (x, y) = hombro izquierdo."""
    raw = {
        "left_shoulder": (x, y), "right_shoulder": (x + 40, y),
        "left_hip": (x + 5, y + 80), "right_hip": (x + 35, y + 80),
        "left_elbow": (x - 5, y + 40), "right_elbow": (x + 45, y + 40),
        "left_wrist": (x - 5, y + 75), "right_wrist": wrist or (x + 45, y + 75),
    }
    head_y = y + 12 if head_dropped else y - 30
    raw["left_ear"], raw["right_ear"] = (x + 12, head_y), (x + 28, head_y)
    if not back_view:
        raw["nose"] = (x + 20, head_y + 2)
    return {
        name: {"x": raw[name][0], "y": raw[name][1], "confidence": 0.9, "visible": True} if name in raw
        else {"x": None, "y": None, "confidence": 0.1, "visible": False}
        for name in ("nose", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow",
                     "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip")
    }


if __name__ == "__main__":
    demo_modules: List[BaseBehaviorModule] = [
        SleepingBehavior(),
        LoiteringBehavior({"min_dwell_seconds": 8.0, "sustain_seconds": 6.0}),
        ProductInteractionBehavior(),
    ]
    service = BehaviorService(config=BehaviorConfig(), modules=demo_modules)
    service.set_zones("CAM-001", [
        Zone("pasillo", "Pasillo", ((0, 0), (960, 0), (960, 1080), (0, 1080))),
        Zone("zona_caja", "Zona caja", ((960, 0), (1920, 0), (1920, 1080), (960, 1080)),
             behavior_overrides={"loitering": {"enabled": False}}),
        Zone("anaquel_1", "Anaquel 1", ((1500, 100), (1700, 100), (1700, 300), (1500, 300)), kind="product"),
    ])
    service.start()

    base = 1_000_000.0
    fps = 5
    for i in range(int(40 * fps)):
        t = base + i / fps
        frame_id = i + 1
        tracks: List[Dict[str, object]] = []
        poses: List[Dict[str, object]] = []

        # T1: pasillo, vista de espalda, cabeza caida e inmovil hasta t=30s; luego camina.
        moving = i >= 30 * fps
        x1 = 300 + (i - 30 * fps) * 8 if moving else 300
        tracks.append({"track_id": "CAM-001-1", "class_name": "person", "track_state": "ACTIVE",
                       "bbox": {"x1": x1, "y1": 400, "x2": x1 + 80, "y2": 600}})
        poses.append({"track_id": "CAM-001-1", "status": "OK",
                      "keypoints": _demo_keypoints(x1 + 20, 430, head_dropped=not moving, back_view=True),
                      "derived_signals": {"torso_inclination_deg": 5.0}})

        # T2: zona_caja, erguido e inmovil (esperando) -> sin merodeo por zona.
        tracks.append({"track_id": "CAM-001-2", "class_name": "person", "track_state": "ACTIVE",
                       "bbox": {"x1": 1100, "y1": 400, "x2": 1180, "y2": 600}})
        poses.append({"track_id": "CAM-001-2", "status": "OK",
                      "keypoints": _demo_keypoints(1120, 430, head_dropped=False, back_view=False),
                      "derived_signals": {"torso_inclination_deg": 2.0}})

        # T3: mano en el anaquel entre t=5s y t=10s; LOST desde t=20s (se descarta tras la gracia).
        state3 = "LOST" if i >= 20 * fps else "ACTIVE"
        wrist = (1550.0, 250.0) if 5 * fps <= i < 10 * fps else None
        tracks.append({"track_id": "CAM-001-3", "class_name": "person", "track_state": state3,
                       "bbox": {"x1": 1400, "y1": 200, "x2": 1480, "y2": 400}})
        if state3 == "ACTIVE":
            poses.append({"track_id": "CAM-001-3", "status": "OK",
                          "keypoints": _demo_keypoints(1420, 230, head_dropped=False, back_view=False, wrist=wrist),
                          "derived_signals": {"torso_inclination_deg": 3.0}})

        service.submit_tracking_result({"camera_id": "CAM-001", "session_id": "s-demo", "frame_id": frame_id,
                                        "timestamp": t, "tracks": tracks})
        service.submit_pose_result({"camera_id": "CAM-001", "session_id": "s-demo", "frame_id": frame_id,
                                    "timestamp": t, "poses": poses})
        time.sleep(0.003)

    time.sleep(0.5)
    transitions = []
    while True:
        result = service.get_next_result(timeout=0.2)
        if result is None:
            break
        if result["state"] in ("POSSIBLE", "SUSTAINED", "FINISHED"):
            transitions.append(result)

    seen: Set[Tuple[object, object, object]] = set()
    for r in transitions:
        key = (r["track_id"], r["behavior_type"], r["state"])
        if key in seen:
            continue
        seen.add(key)
        ev = r["evidence"]
        print(
            f"t={r['timestamp'] - base:5.1f}s {r['track_id']:<10} {r['behavior_type']:<20} "
            f"{r['state']:<9} strength={r['evidence_strength']:.2f} obs={r['observability']:.2f} "
            f"| {r['description']} | finish={ev.get('finish_reason')} missing={ev.get('signals_missing')}"
        )

    health = service.health_check()
    print("HEALTH:", {k: health[k] for k in ("modules_loaded", "subjects_analyzed", "open_episodes", "avg_processing_latency_ms")})
    print("METRICS:", health["metrics"])
    service.stop()
