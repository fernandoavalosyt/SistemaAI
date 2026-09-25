"""
decisionService.py

Servicio de Decision Engine (Servicio 7 del pipeline).

Responsabilidad UNICA: recibir BehaviorResult (evidencia temporal) desde
"Behavior Analysis" (Servicio 6), combinarla con contexto externo (tipo de
camara, horario, tipo de zona) y evaluarla contra reglas de negocio
DECLARATIVAS (JSON/YAML) para emitir un veredicto estructurado
(DecisionResult) hacia "Event / Evidence" (Servicio 8).

"La evidencia cumple la regla X de negocio -> se emite una decision
estructurada y explicable". Nada mas.

Este servicio NO procesa imagenes ni video (no importa OpenCV/PyTorch/YOLO),
NO envia notificaciones, correos ni alertas, y NO persiste en bases de datos
de largo plazo. Todo su estado es memoria acotada en proceso.

Convencion temporal: cooldowns, ventanas de secuencia y confirmaciones usan
el TIEMPO DE LOS DATOS (timestamp del BehaviorResult), no el reloj de pared,
para que las decisiones sean deterministas y reproducibles.
"""

from __future__ import annotations

import abc
import copy
import hashlib
import json
import logging
import os
import threading
import time
import uuid
import zlib
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Deque, Dict, Generator, List, Optional, Sequence, Tuple


# =============================================================================
# Logging
# =============================================================================

logger = logging.getLogger("decision_service")
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
    """BehaviorResult de entrada invalido."""


class RuleValidationError(Exception):
    """Definicion de regla invalida (se detecta al cargar, nunca en runtime)."""


class ContextConfigError(Exception):
    """Configuracion de contexto externo invalida."""


class Decision(str, Enum):
    EVENT_GENERATED = "EVENT_GENERATED"
    NO_EVENT = "NO_EVENT"


class Priority(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    @property
    def rank(self) -> int:
        return _PRIORITY_RANK[self]


_PRIORITY_RANK = {Priority.LOW: 0, Priority.MEDIUM: 1, Priority.HIGH: 2, Priority.CRITICAL: 3}


class EvaluationState(str, Enum):
    """Estado de evaluacion de una regla para un sujeto concreto."""

    NORMAL = "NORMAL"
    OBSERVATION = "OBSERVATION"          # la evidencia es relevante para la regla, pero no la cumple
    POSSIBLE_EVENT = "POSSIBLE_EVENT"    # la cumple, esperando confirmaciones
    EVENT_GENERATED = "EVENT_GENERATED"  # evento emitido; en cooldown


_EVENT_ID_NAMESPACE = uuid.UUID("6f1c2d0e-8a51-4b8e-9d3c-7a4e2f9b1c55")
_MISSING = object()


# =============================================================================
# DSL de condiciones (compilada y validada al cargar; sin eval())
# =============================================================================

def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _resolve_path(facts: Dict[str, Any], path: Tuple[str, ...]) -> Any:
    current: Any = facts
    for part in path:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return _MISSING
    return current


def _op_compare(fn):
    def op(actual: Any, expected: Any) -> bool:
        return _is_number(actual) and fn(actual, expected)
    return op


def _op_contains(actual: Any, expected: Any) -> bool:
    return isinstance(actual, (list, tuple, set, str)) and expected in actual


def _op_contains_any(actual: Any, expected: Any) -> bool:
    return isinstance(actual, (list, tuple, set)) and any(v in actual for v in expected)


_OPERATORS = {
    "eq": lambda a, e: a is not _MISSING and a == e,
    "ne": lambda a, e: a is not _MISSING and a != e,
    "gt": _op_compare(lambda a, e: a > e),
    "gte": _op_compare(lambda a, e: a >= e),
    "lt": _op_compare(lambda a, e: a < e),
    "lte": _op_compare(lambda a, e: a <= e),
    "between": _op_compare(lambda a, e: e[0] <= a <= e[1]),
    "in": lambda a, e: a is not _MISSING and a in e,
    "not_in": lambda a, e: a is not _MISSING and a not in e,
    "contains": _op_contains,
    "contains_any": _op_contains_any,
    "exists": lambda a, e: (a is not _MISSING and a is not None) == e,
}


def _validate_operand(op: str, value: Any, where: str) -> None:
    if op not in _OPERATORS:
        raise RuleValidationError(f"{where}: operador desconocido '{op}'")
    if op in ("gt", "gte", "lt", "lte") and not _is_number(value):
        raise RuleValidationError(f"{where}: '{op}' requiere un valor numerico")
    if op == "between" and not (
        isinstance(value, list) and len(value) == 2 and all(_is_number(v) for v in value) and value[0] <= value[1]
    ):
        raise RuleValidationError(f"{where}: 'between' requiere [min, max] numericos")
    if op in ("in", "not_in", "contains_any") and not isinstance(value, list):
        raise RuleValidationError(f"{where}: '{op}' requiere una lista")
    if op == "exists" and not isinstance(value, bool):
        raise RuleValidationError(f"{where}: 'exists' requiere true/false")


class Condition(abc.ABC):
    """Nodo del arbol de condiciones. evaluate() devuelve (resultado,
    traza), donde la traza lista las condiciones hoja que sustentan el
    resultado con su valor REAL: es la base del audit_trail."""

    @abc.abstractmethod
    def evaluate(self, facts: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]]]:
        raise NotImplementedError

    @staticmethod
    def compile(node: Any, where: str = "condition") -> "Condition":
        if not isinstance(node, dict) or len(node) == 0:
            raise RuleValidationError(f"{where}: se esperaba un objeto de condicion")
        if "all" in node or "any" in node:
            key = "all" if "all" in node else "any"
            children = node[key]
            if not isinstance(children, list) or not children:
                raise RuleValidationError(f"{where}.{key}: debe ser una lista no vacia")
            compiled = [Condition.compile(c, f"{where}.{key}[{i}]") for i, c in enumerate(children)]
            return AllCondition(compiled) if key == "all" else AnyCondition(compiled)
        if "not" in node:
            return NotCondition(Condition.compile(node["not"], f"{where}.not"))
        if "field" in node and "op" in node:
            if not isinstance(node["field"], str) or not node["field"]:
                raise RuleValidationError(f"{where}: 'field' invalido")
            _validate_operand(node["op"], node.get("value"), where)
            return LeafCondition(node["field"], node["op"], node.get("value"))
        raise RuleValidationError(f"{where}: nodo desconocido {sorted(node.keys())}")


class LeafCondition(Condition):
    def __init__(self, field_path: str, op: str, value: Any) -> None:
        self.field_path = field_path
        self._path = tuple(field_path.split("."))
        self.op = op
        self.value = value
        self._fn = _OPERATORS[op]

    def evaluate(self, facts: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]]]:
        actual = _resolve_path(facts, self._path)
        result = bool(self._fn(actual, self.value))
        return result, [{
            "field": self.field_path,
            "op": self.op,
            "expected": self.value,
            "actual": None if actual is _MISSING else actual,
            "result": result,
        }]


class AllCondition(Condition):
    def __init__(self, children: List[Condition]) -> None:
        self.children = children

    def evaluate(self, facts: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]]]:
        trace: List[Dict[str, Any]] = []
        for child in self.children:
            ok, child_trace = child.evaluate(facts)
            trace.extend(child_trace)
            if not ok:
                return False, trace
        return True, trace


class AnyCondition(Condition):
    def __init__(self, children: List[Condition]) -> None:
        self.children = children

    def evaluate(self, facts: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]]]:
        failed: List[Dict[str, Any]] = []
        for child in self.children:
            ok, child_trace = child.evaluate(facts)
            if ok:
                return True, child_trace  # solo la rama que realmente disparo
            failed.extend(child_trace)
        return False, failed


class NotCondition(Condition):
    def __init__(self, child: Condition) -> None:
        self.child = child

    def evaluate(self, facts: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]]]:
        ok, child_trace = self.child.evaluate(facts)
        return (not ok), [{"negated": True, "conditions": child_trace, "result": not ok}]


# =============================================================================
# Reglas
# =============================================================================

@dataclass(frozen=True)
class PriorityOverride:
    when: Condition
    priority: Priority


@dataclass(frozen=True)
class RuleAction:
    event_type: str
    priority: Priority
    priority_overrides: Tuple[PriorityOverride, ...]
    cooldown_seconds: float
    confirm_count: int
    pre_roll_seconds: float
    post_roll_seconds: float


@dataclass(frozen=True)
class SequenceStep:
    name: str
    condition: Condition
    require_zone_change: bool


@dataclass(frozen=True)
class Rule:
    rule_id: str
    name: str
    description: str
    rule_type: str                           # "simple" | "sequence"
    applies_to_behaviors: Optional[frozenset]
    scope: Optional[Condition]
    condition: Optional[Condition]
    steps: Tuple[SequenceStep, ...]
    window_seconds: float
    action: RuleAction
    order: int
    stop_processing: bool
    raw: Dict[str, Any]

    def is_relevant(self, behavior_type: str) -> bool:
        return self.applies_to_behaviors is None or behavior_type in self.applies_to_behaviors


def compile_rule(raw: Dict[str, Any], index: int) -> Rule:
    """Valida y compila una regla declarativa. Cualquier error se reporta
    con el rule_id y la ruta exacta del campo invalido."""
    if not isinstance(raw, dict):
        raise RuleValidationError(f"rules[{index}]: se esperaba un objeto")
    rule_id = raw.get("rule_id")
    if not isinstance(rule_id, str) or not rule_id:
        raise RuleValidationError(f"rules[{index}]: 'rule_id' obligatorio")
    where = f"rule '{rule_id}'"

    rule_type = raw.get("type", "simple")
    if rule_type not in ("simple", "sequence"):
        raise RuleValidationError(f"{where}: type debe ser 'simple' o 'sequence'")

    applies = raw.get("applies_to_behaviors")
    if applies is not None and (not isinstance(applies, list) or not all(isinstance(b, str) for b in applies)):
        raise RuleValidationError(f"{where}: 'applies_to_behaviors' debe ser lista de strings")

    scope = Condition.compile(raw["scope"], f"{where}.scope") if "scope" in raw else None

    condition: Optional[Condition] = None
    steps: List[SequenceStep] = []
    window_seconds = 0.0
    if rule_type == "simple":
        if "condition" not in raw:
            raise RuleValidationError(f"{where}: una regla simple requiere 'condition'")
        condition = Condition.compile(raw["condition"], f"{where}.condition")
    else:
        sequence = raw.get("sequence")
        if not isinstance(sequence, dict):
            raise RuleValidationError(f"{where}: una regla de secuencia requiere 'sequence'")
        window_seconds = float(sequence.get("window_seconds", 0))
        if window_seconds <= 0:
            raise RuleValidationError(f"{where}.sequence: 'window_seconds' debe ser > 0")
        raw_steps = sequence.get("steps")
        if not isinstance(raw_steps, list) or len(raw_steps) < 2:
            raise RuleValidationError(f"{where}.sequence: se requieren al menos 2 'steps'")
        for i, step in enumerate(raw_steps):
            if not isinstance(step, dict) or "condition" not in step:
                raise RuleValidationError(f"{where}.sequence.steps[{i}]: requiere 'condition'")
            steps.append(SequenceStep(
                name=str(step.get("name", f"step_{i}")),
                condition=Condition.compile(step["condition"], f"{where}.sequence.steps[{i}]"),
                require_zone_change=bool(step.get("require_zone_change", False)),
            ))
        if steps[0].require_zone_change:
            raise RuleValidationError(f"{where}: el primer paso no puede exigir cambio de zona")

    action_raw = raw.get("action")
    if not isinstance(action_raw, dict) or not action_raw.get("event_type"):
        raise RuleValidationError(f"{where}: 'action.event_type' obligatorio")
    try:
        priority = Priority(action_raw.get("priority", "MEDIUM"))
        overrides = tuple(
            PriorityOverride(
                when=Condition.compile(o["when"], f"{where}.action.priority_overrides[{i}]"),
                priority=Priority(o["priority"]),
            )
            for i, o in enumerate(action_raw.get("priority_overrides", []))
        )
    except (KeyError, ValueError) as exc:
        raise RuleValidationError(f"{where}.action: prioridad invalida ({exc})") from exc

    cooldown = float(action_raw.get("cooldown_seconds", 300))
    confirm = int(action_raw.get("confirm_count", 1))
    pre_roll = float(action_raw.get("pre_roll_seconds", 10))
    post_roll = float(action_raw.get("post_roll_seconds", 5))
    if cooldown < 0 or confirm < 1 or pre_roll < 0 or post_roll < 0:
        raise RuleValidationError(f"{where}.action: cooldown/pre/post >= 0 y confirm_count >= 1")

    return Rule(
        rule_id=rule_id,
        name=str(raw.get("name", rule_id)),
        description=str(raw.get("description", "")),
        rule_type=rule_type,
        applies_to_behaviors=frozenset(applies) if applies is not None else None,
        scope=scope,
        condition=condition,
        steps=tuple(steps),
        window_seconds=window_seconds,
        action=RuleAction(
            event_type=str(action_raw["event_type"]),
            priority=priority,
            priority_overrides=overrides,
            cooldown_seconds=cooldown,
            confirm_count=confirm,
            pre_roll_seconds=pre_roll,
            post_roll_seconds=post_roll,
        ),
        order=int(raw.get("order", 100)),
        stop_processing=bool(raw.get("stop_processing", False)),
        raw=copy.deepcopy(raw),
    )


# Reglas por defecto (mismo formato que un archivo JSON/YAML externo).
DEFAULT_RULES: List[Dict[str, Any]] = [
    {
        "rule_id": "SLEEPING_GUARD_DESK",
        "name": "Guardia con somnolencia sostenida en su puesto",
        "type": "simple",
        "order": 10,
        "applies_to_behaviors": ["somnolence"],
        "scope": {"all": [
            {"field": "behavior_type", "op": "eq", "value": "somnolence"},
            {"field": "context.zone_type", "op": "eq", "value": "security_desk"},
        ]},
        "condition": {"all": [
            {"field": "behavior_type", "op": "eq", "value": "somnolence"},
            {"field": "state", "op": "eq", "value": "SUSTAINED"},
            {"field": "duration", "op": "gte", "value": 15},
            {"field": "context.zone_type", "op": "eq", "value": "security_desk"},
            {"field": "observability", "op": "gte", "value": 0.4},
        ]},
        "action": {
            "event_type": "SLEEPING_GUARD",
            "priority": "HIGH",
            "priority_overrides": [
                {"when": {"field": "context.schedules", "op": "contains", "value": "night_shift"}, "priority": "CRITICAL"},
            ],
            "cooldown_seconds": 600,
            "confirm_count": 2,
            "pre_roll_seconds": 20,
            "post_roll_seconds": 5,
        },
    },
    {
        "rule_id": "LOITERING_SENSITIVE_ZONE",
        "name": "Permanencia prolongada en zona sensible",
        "type": "simple",
        "order": 20,
        "applies_to_behaviors": ["loitering"],
        "condition": {"all": [
            {"field": "behavior_type", "op": "eq", "value": "loitering"},
            {"field": "state", "op": "eq", "value": "SUSTAINED"},
            {"any": [
                {"field": "context.zone_type", "op": "in", "value": ["restricted", "exit", "storage"]},
                {"field": "context.schedules", "op": "contains", "value": "closed"},
            ]},
        ]},
        "action": {
            "event_type": "SUSPICIOUS_BEHAVIOR",
            "priority": "MEDIUM",
            "priority_overrides": [
                {"when": {"field": "context.schedules", "op": "contains", "value": "closed"}, "priority": "HIGH"},
            ],
            "cooldown_seconds": 300,
            "pre_roll_seconds": 30,
            "post_roll_seconds": 5,
        },
    },
    {
        "rule_id": "POSSIBLE_THEFT_SEQUENCE",
        "name": "Interaccion con producto seguida de desplazamiento a zona de salida/restringida",
        "type": "sequence",
        "order": 5,
        "sequence": {
            "window_seconds": 300,
            "steps": [
                {"name": "product_interaction", "condition": {"all": [
                    {"field": "behavior_type", "op": "eq", "value": "product_interaction"},
                    {"field": "state", "op": "in", "value": ["SUSTAINED", "FINISHED"]},
                ]}},
                {"name": "moved_to_sensitive_zone", "require_zone_change": True, "condition": {
                    "field": "context.zone_type", "op": "in", "value": ["exit", "restricted"],
                }},
            ],
        },
        "action": {
            "event_type": "POSSIBLE_THEFT",
            "priority": "HIGH",
            "cooldown_seconds": 900,
            "pre_roll_seconds": 15,
            "post_roll_seconds": 10,
        },
    },
]


def load_rules_file(path: str) -> List[Dict[str, Any]]:
    """Carga reglas desde JSON o YAML (si PyYAML esta disponible). Acepta
    una lista de reglas o un objeto {"rules": [...]}."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Archivo de reglas no encontrado: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        if path.lower().endswith((".yaml", ".yml")):
            try:
                import yaml
            except ImportError as exc:
                raise RuleValidationError("Se requiere PyYAML para cargar reglas YAML") from exc
            data = yaml.safe_load(handle)
        else:
            data = json.load(handle)
    rules = data.get("rules") if isinstance(data, dict) else data
    if not isinstance(rules, list):
        raise RuleValidationError("El archivo de reglas debe contener una lista de reglas")
    return rules


class RuleEngine:
    """Contenedor inmutable-por-version del conjunto de reglas. La recarga
    compila y valida TODO el conjunto antes de publicarlo: un archivo con
    errores nunca reemplaza a un conjunto valido (swap atomico)."""

    def __init__(self) -> None:
        self._rules: Tuple[Rule, ...] = ()
        self._lock = threading.Lock()
        self.source: str = "none"
        self.version_hash: str = ""
        self.loaded_at: Optional[float] = None
        self.last_error: Optional[str] = None

    @property
    def rules(self) -> Tuple[Rule, ...]:
        return self._rules  # lectura atomica de la referencia actual

    def load(self, raw_rules: List[Dict[str, Any]], source: str) -> None:
        with self._lock:
            try:
                compiled = [compile_rule(r, i) for i, r in enumerate(raw_rules)]
                ids = [r.rule_id for r in compiled]
                duplicates = {i for i in ids if ids.count(i) > 1}
                if duplicates:
                    raise RuleValidationError(f"rule_id duplicados: {sorted(duplicates)}")
            except (RuleValidationError, TypeError, ValueError) as exc:
                self.last_error = str(exc)
                logger.error("Carga de reglas rechazada (%s): %s. Se conserva el conjunto anterior.", source, exc)
                raise
            enabled = [r for r, raw in zip(compiled, raw_rules) if raw.get("enabled", True)]
            enabled.sort(key=lambda r: r.order)
            self._rules = tuple(enabled)
            self.source = source
            self.version_hash = hashlib.sha256(
                json.dumps(raw_rules, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()[:12]
            self.loaded_at = time.time()
            self.last_error = None
            logger.info("Reglas cargadas | fuente=%s version=%s activas=%s",
                        source, self.version_hash, [r.rule_id for r in self._rules])

    def status(self) -> Dict[str, Any]:
        return {
            "loaded": bool(self._rules),
            "source": self.source,
            "version": self.version_hash,
            "active_rules": [r.rule_id for r in self._rules],
            "loaded_at": self.loaded_at,
            "last_error": self.last_error,
        }


# =============================================================================
# Contexto externo: tipo de camara, zonas y horarios
# =============================================================================

def _parse_hhmm(value: str) -> int:
    hours, minutes = value.split(":")
    h, m = int(hours), int(minutes)
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise ValueError(value)
    return h * 60 + m


@dataclass(frozen=True)
class CameraContext:
    camera_id: str
    camera_type: str
    site: Optional[str]
    utc_offset_hours: float
    zones: Dict[str, Dict[str, Any]]
    schedules: Dict[str, Tuple[Tuple[int, int], ...]]   # nombre -> ((inicio_min, fin_min), ...)

    def active_schedules(self, timestamp: float) -> Tuple[List[str], datetime]:
        local = datetime.fromtimestamp(timestamp, tz=timezone(timedelta(hours=self.utc_offset_hours)))
        minute = local.hour * 60 + local.minute
        active = []
        for name, ranges in self.schedules.items():
            for start, end in ranges:
                inside = start <= minute < end if start <= end else (minute >= start or minute < end)
                if inside:
                    active.append(name)
                    break
        return sorted(active), local


class ContextProvider:
    """Contexto de negocio por camara. Formato (JSON):
    {"CAM-001": {"camera_type": "retail", "site": "tienda_centro",
                 "utc_offset_hours": -6,
                 "zones": {"salida": {"zone_type": "exit"}},
                 "schedules": {"night_shift": [["22:00", "06:00"]]}}}
    """

    def __init__(self) -> None:
        self._cameras: Dict[str, CameraContext] = {}
        self._lock = threading.Lock()
        self._warned_unknown: set = set()

    def configure(self, raw: Dict[str, Dict[str, Any]]) -> None:
        parsed: Dict[str, CameraContext] = {}
        for camera_id, cfg in raw.items():
            try:
                schedules = {
                    str(name): tuple((_parse_hhmm(r[0]), _parse_hhmm(r[1])) for r in ranges)
                    for name, ranges in cfg.get("schedules", {}).items()
                }
                parsed[camera_id] = CameraContext(
                    camera_id=camera_id,
                    camera_type=str(cfg.get("camera_type", "unknown")),
                    site=cfg.get("site"),
                    utc_offset_hours=float(cfg.get("utc_offset_hours", 0)),
                    zones={str(z): dict(v) for z, v in cfg.get("zones", {}).items()},
                    schedules=schedules,
                )
            except (TypeError, ValueError, IndexError, AttributeError) as exc:
                raise ContextConfigError(f"Contexto invalido para {camera_id}: {exc}") from exc
        with self._lock:
            self._cameras.update(parsed)
        logger.info("Contexto de camaras configurado | camaras=%s", sorted(parsed))

    def load_json(self, path: str) -> None:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Archivo de contexto no encontrado: {path}")
        with open(path, "r", encoding="utf-8") as handle:
            self.configure(json.load(handle))

    def facts_for(self, camera_id: str, zone_id: Optional[str], timestamp: float) -> Dict[str, Any]:
        with self._lock:
            ctx = self._cameras.get(camera_id)
        if ctx is None:
            if camera_id not in self._warned_unknown:
                self._warned_unknown.add(camera_id)
                logger.warning("Camara sin contexto configurado: %s (reglas dependientes de contexto no aplicaran)", camera_id)
            return {"camera_type": "unknown", "site": None, "zone_type": None, "zone_name": None,
                    "schedules": [], "local_hour": None, "weekday": None}
        schedules, local = ctx.active_schedules(timestamp)
        zone_cfg = ctx.zones.get(zone_id or "", {})
        return {
            "camera_type": ctx.camera_type,
            "site": ctx.site,
            "zone_type": zone_cfg.get("zone_type"),
            "zone_name": zone_cfg.get("name", zone_id),
            "schedules": schedules,
            "local_hour": local.hour,
            "weekday": local.strftime("%A").lower(),
        }


# =============================================================================
# Validacion de entrada
# =============================================================================

_VALID_BEHAVIOR_STATES = ("POSSIBLE", "SUSTAINED", "FINISHED")


def validate_behavior_result(payload: Optional[Dict[str, Any]]) -> None:
    if not payload or not isinstance(payload, dict):
        raise InputValidationError("BehaviorResult nulo o vacio")
    for key in ("camera_id", "session_id", "track_id", "behavior_type", "state", "duration", "timestamp"):
        if key not in payload or payload[key] is None:
            raise InputValidationError(f"Falta campo obligatorio: {key}")
    if payload["state"] not in _VALID_BEHAVIOR_STATES:
        raise InputValidationError(f"state invalido: {payload['state']!r}")
    if not _is_number(payload["timestamp"]) or payload["timestamp"] <= 0:
        raise InputValidationError("timestamp invalido")
    if not _is_number(payload["duration"]) or payload["duration"] < 0:
        raise InputValidationError("duration invalida")
    if "evidence" in payload and not isinstance(payload["evidence"], dict):
        raise InputValidationError("evidence debe ser un objeto")


def result_key(payload: Dict[str, Any]) -> str:
    """Clave de idempotencia. Se usa result_id (unico por emision de
    Behavior); si falta, una huella determinista del contenido."""
    if payload.get("result_id"):
        return f"rid:{payload['result_id']}"
    fingerprint = json.dumps(
        {k: payload.get(k) for k in ("camera_id", "track_id", "subject_id", "episode_id",
                                     "behavior_type", "state", "timestamp", "duration")},
        sort_keys=True, default=str,
    )
    return "fp:" + hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()


# =============================================================================
# Memoria temporal y estado por camara (particionado = thread-safety)
# =============================================================================

@dataclass
class HistoryEntry:
    result_key: str
    episode_id: Optional[str]
    behavior_type: str
    state: str
    timestamp: float
    start_time: float
    duration: float
    zone_id: Optional[str]
    facts: Dict[str, Any]


@dataclass
class RuleTracker:
    """Estado de UNA regla para UN sujeto."""

    state: EvaluationState = EvaluationState.NORMAL
    match_count: int = 0
    first_match_time: Optional[float] = None
    cooldown_until: float = 0.0
    last_event_id: Optional[str] = None
    last_update: float = 0.0
    path: Deque[str] = field(default_factory=lambda: deque(maxlen=10))

    def transition(self, new_state: EvaluationState, timestamp: float) -> None:
        if new_state != self.state:
            self.state = new_state
            self.path.append(f"{new_state.value}@{timestamp:.3f}")


class CameraState:
    """Toda la memoria de una camara bajo un unico lock. Distintas camaras
    se procesan en paralelo sin contencion entre si."""

    def __init__(self, camera_id: str, history_max_per_subject: int) -> None:
        self.camera_id = camera_id
        self.lock = threading.RLock()
        self.history: Dict[str, Deque[HistoryEntry]] = {}
        self.trackers: Dict[Tuple[str, str], RuleTracker] = {}
        self.latest_timestamp: float = 0.0
        self.last_wall_update: float = time.time()
        self._max_history = history_max_per_subject

    def record(self, subject: str, entry: HistoryEntry) -> None:
        """Compacta heartbeats: un mismo episodio en el mismo estado ocupa
        una sola entrada (se actualiza y se mueve al final)."""
        entries = self.history.setdefault(subject, deque(maxlen=self._max_history))
        if entry.episode_id:
            for existing in reversed(entries):
                if existing.episode_id == entry.episode_id and existing.state == entry.state:
                    entries.remove(existing)
                    break
        entries.append(entry)

    def prune(self, horizon_seconds: float) -> None:
        cutoff = self.latest_timestamp - horizon_seconds
        for subject in list(self.history):
            entries = self.history[subject]
            while entries and entries[0].timestamp < cutoff:
                entries.popleft()
            if not entries:
                del self.history[subject]
        for key in list(self.trackers):
            tracker = self.trackers[key]
            idle = self.latest_timestamp - tracker.last_update
            if tracker.cooldown_until <= self.latest_timestamp and idle > horizon_seconds:
                del self.trackers[key]

    def size(self) -> Tuple[int, int, int]:
        return len(self.history), sum(len(v) for v in self.history.values()), len(self.trackers)


# =============================================================================
# Contrato de salida
# =============================================================================

@dataclass
class DecisionResult:
    event_id: str
    camera_id: str
    session_id: str
    track_id: str
    subject_id: str
    decision: Decision
    event_type: Optional[str]
    priority: Optional[Priority]
    timestamp: float
    time_window_start: Optional[float]
    time_window_end: Optional[float]
    explanation: Dict[str, Any]
    rule_id: Optional[str] = None
    source_result_keys: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "camera_id": self.camera_id,
            "session_id": self.session_id,
            "track_id": self.track_id,
            "subject_id": self.subject_id,
            "decision": self.decision.value,
            "event_type": self.event_type,
            "priority": self.priority.value if self.priority else None,
            "timestamp": self.timestamp,
            "time_window_start": self.time_window_start,
            "time_window_end": self.time_window_end,
            "rule_id": self.rule_id,
            "source_result_keys": self.source_result_keys,
            "explanation": self.explanation,
        }


# =============================================================================
# Metricas
# =============================================================================

class DecisionMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.decisiones_procesadas = 0
        self.eventos_generados = 0
        self.eventos_descartados_por_cooldown = 0
        self.duplicados_ignorados = 0
        self.entradas_invalidas = 0
        self.descartes_por_backpressure = 0
        self.reglas_activadas: Dict[str, int] = {}
        self.errores_evaluacion = 0
        self._total_latency = 0.0

    def add(self, name: str, amount: int = 1) -> None:
        with self._lock:
            setattr(self, name, getattr(self, name) + amount)

    def rule_triggered(self, rule_id: str) -> None:
        with self._lock:
            self.reglas_activadas[rule_id] = self.reglas_activadas.get(rule_id, 0) + 1

    def latency(self, seconds: float) -> None:
        with self._lock:
            self._total_latency += seconds

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            processed = self.decisiones_procesadas
            return {
                "decisiones_procesadas": processed,
                "eventos_generados": self.eventos_generados,
                "eventos_descartados_por_cooldown": self.eventos_descartados_por_cooldown,
                "duplicados_ignorados": self.duplicados_ignorados,
                "entradas_invalidas": self.entradas_invalidas,
                "descartes_por_backpressure": self.descartes_por_backpressure,
                "errores_evaluacion": self.errores_evaluacion,
                "reglas_activadas": dict(self.reglas_activadas),
                "latencia_promedio_ms": round(self._total_latency / processed * 1000.0, 3) if processed else 0.0,
            }


# =============================================================================
# Cola acotada
# =============================================================================

class BoundedQueue:
    def __init__(self, max_size: int, on_drop=None) -> None:
        if max_size <= 0:
            raise ValueError("max_size debe ser mayor a 0")
        self._max_size = max_size
        self._buffer: Deque[Any] = deque()
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._on_drop = on_drop

    def put(self, item: Any) -> None:
        with self._not_empty:
            if len(self._buffer) >= self._max_size:
                self._buffer.popleft()
                if self._on_drop is not None:
                    self._on_drop()
            self._buffer.append(item)
            self._not_empty.notify()

    def get(self, timeout: Optional[float] = None) -> Optional[Any]:
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
# Configuracion
# =============================================================================

@dataclass(frozen=True)
class DecisionConfig:
    history_horizon_seconds: float = 900.0     # memoria temporal por sujeto (>= mayor ventana de secuencia)
    history_max_per_subject: int = 200
    idempotency_cache_size: int = 20000
    workers: int = 4
    input_queue_size: int = 2000
    output_queue_size: int = 2000
    publish_no_event: bool = False             # NO_EVENT solo se devuelve al llamador sincrono
    stale_camera_seconds: float = 3600.0
    max_evidence_bytes: int = 8192             # tope del evidence copiado a la explicacion

    @staticmethod
    def from_env() -> "DecisionConfig":
        return DecisionConfig(
            history_horizon_seconds=float(os.getenv("DECISION_HISTORY_SECONDS", "900")),
            workers=int(os.getenv("DECISION_WORKERS", "4")),
            input_queue_size=int(os.getenv("DECISION_INPUT_QUEUE_SIZE", "2000")),
            output_queue_size=int(os.getenv("DECISION_OUTPUT_QUEUE_SIZE", "2000")),
            publish_no_event=os.getenv("DECISION_PUBLISH_NO_EVENT", "false").lower() == "true",
        )


# =============================================================================
# Servicio principal
# =============================================================================

@dataclass
class _Match:
    rule: Rule
    trace: List[Dict[str, Any]]
    entries: List[HistoryEntry]          # evidencias que sustentan la decision (en orden temporal)
    step_names: List[str]


class DecisionService:
    """Punto de entrada del Servicio 7.

    Concurrencia: las entradas se reparten en N workers por hash estable de
    camera_id (crc32). Cada camara siempre cae en el mismo worker, lo que
    preserva el orden temporal por camara y permite paralelismo entre
    camaras. process_behavior_result() tambien es seguro para llamadas
    directas desde multiples hilos (lock por camara + cache de idempotencia
    con lock propio).
    """

    def __init__(
        self,
        config: Optional[DecisionConfig] = None,
        rules: Optional[List[Dict[str, Any]]] = None,
        context: Optional[ContextProvider] = None,
    ) -> None:
        self.config = config or DecisionConfig.from_env()
        self.metrics = DecisionMetrics()
        self.engine = RuleEngine()
        self.context = context or ContextProvider()

        rules_file = os.getenv("DECISION_RULES_FILE")
        if rules is not None:
            self.engine.load(rules, "inline")
        elif rules_file:
            self.engine.load(load_rules_file(rules_file), rules_file)
        else:
            self.engine.load(DEFAULT_RULES, "DEFAULT_RULES")

        context_file = os.getenv("DECISION_CONTEXT_FILE")
        if context_file:
            self.context.load_json(context_file)

        self._cameras: Dict[str, CameraState] = {}
        self._cameras_lock = threading.Lock()

        self._processed: "OrderedDict[str, List[DecisionResult]]" = OrderedDict()
        self._processed_lock = threading.Lock()

        self._queues = [
            BoundedQueue(self.config.input_queue_size, on_drop=self._on_input_drop)
            for _ in range(max(1, self.config.workers))
        ]
        self._output = BoundedQueue(self.config.output_queue_size, on_drop=self._on_output_drop)
        self._threads: List[threading.Thread] = []
        self._stop_event = threading.Event()
        self._start_wall = time.time()

    # -------------------------------------------------------------------
    # Ciclo de vida
    # -------------------------------------------------------------------

    def start(self) -> None:
        if any(t.is_alive() for t in self._threads):
            return
        self._stop_event.clear()
        self._threads = [
            threading.Thread(target=self._worker, args=(i,), name=f"DecisionWorker-{i}", daemon=True)
            for i in range(len(self._queues))
        ]
        for thread in self._threads:
            thread.start()
        logger.info("DecisionService iniciado | workers=%d reglas=%d", len(self._threads), len(self.engine.rules))

    def stop(self, timeout: float = 5.0) -> None:
        logger.info("Deteniendo DecisionService...")
        self._stop_event.set()
        for thread in self._threads:
            thread.join(timeout=timeout)
        for queue in self._queues:
            queue.clear()
        logger.info("DecisionService detenido")

    def reload_rules(self, raw_rules: Optional[List[Dict[str, Any]]] = None, path: Optional[str] = None) -> None:
        """Recarga en caliente. Si la nueva definicion es invalida se lanza
        RuleValidationError y el conjunto vigente sigue activo."""
        if path:
            self.engine.load(load_rules_file(path), path)
        elif raw_rules is not None:
            self.engine.load(raw_rules, "inline")

    # -------------------------------------------------------------------
    # Entrada / salida
    # -------------------------------------------------------------------

    def submit(self, payload: Dict[str, Any]) -> None:
        """Entrada asincrona (desde Behavior). Orden garantizado por camara."""
        camera_id = str(payload.get("camera_id", "")) if isinstance(payload, dict) else ""
        index = zlib.crc32(camera_id.encode("utf-8")) % len(self._queues)
        self._queues[index].put(payload)

    def get_next_decision(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        return self._output.get(timeout=timeout)

    def decision_stream(self) -> Generator[Dict[str, Any], None, None]:
        while not self._stop_event.is_set():
            decision = self.get_next_decision(timeout=1.0)
            if decision is not None:
                yield decision

    def process_behavior_result(self, payload: Dict[str, Any]) -> List[DecisionResult]:
        """Evaluacion sincrona e idempotente de un BehaviorResult. Siempre
        devuelve al menos una decision (EVENT_GENERATED o NO_EVENT)."""
        start = time.time()
        try:
            validate_behavior_result(payload)
        except InputValidationError as exc:
            self.metrics.add("entradas_invalidas")
            _rate_limited_logger.log("invalid_input", logging.WARNING, f"BehaviorResult invalidos (%d): {exc}")
            return []

        key = result_key(payload)
        with self._processed_lock:
            cached = self._processed.get(key)
            if cached is not None:
                self._processed.move_to_end(key)
        if cached is not None:
            self.metrics.add("duplicados_ignorados")
            logger.debug("BehaviorResult duplicado (%s): se devuelve la decision original", key)
            return cached

        camera = self._camera_state(str(payload["camera_id"]))
        with camera.lock:
            # Doble chequeo dentro del lock de camara: dos hilos con el mismo
            # payload no pueden evaluar ambos.
            with self._processed_lock:
                if key in self._processed:
                    self.metrics.add("duplicados_ignorados")
                    return self._processed[key]
            try:
                decisions = self._evaluate(camera, payload, key)
            except Exception:
                self.metrics.add("errores_evaluacion")
                logger.exception("Error evaluando reglas para %s (se continua)", key)
                decisions = [self._no_event(payload, key, {"error": "evaluation_error"})]
            with self._processed_lock:
                self._processed[key] = decisions
                while len(self._processed) > self.config.idempotency_cache_size:
                    self._processed.popitem(last=False)

        self.metrics.add("decisiones_procesadas")
        self.metrics.latency(time.time() - start)
        return decisions

    # -------------------------------------------------------------------
    # Health check
    # -------------------------------------------------------------------

    def health_check(self) -> Dict[str, Any]:
        with self._cameras_lock:
            cameras = list(self._cameras.values())
        subjects = entries = trackers = 0
        open_states: Dict[str, int] = {}
        for camera in cameras:
            with camera.lock:
                s, e, t = camera.size()
                subjects, entries, trackers = subjects + s, entries + e, trackers + t
                for tracker in camera.trackers.values():
                    if tracker.state != EvaluationState.NORMAL:
                        open_states[tracker.state.value] = open_states.get(tracker.state.value, 0) + 1
        with self._processed_lock:
            idempotency_size = len(self._processed)
        return {
            "service_active": any(t.is_alive() for t in self._threads),
            "rule_engine": self.engine.status(),
            "cameras": len(cameras),
            "temporal_cache": {
                "subjects": subjects,
                "history_entries": entries,
                "rule_trackers": trackers,
                "tracker_states": open_states,
                "idempotency_entries": idempotency_size,
            },
            "input_queue_usage": [f"{q.qsize()}/{q.max_size}" for q in self._queues],
            "output_queue_usage": f"{self._output.qsize()}/{self._output.max_size}",
            "uptime_seconds": round(time.time() - self._start_wall, 3),
            "metrics": self.metrics.to_dict(),
        }

    # -------------------------------------------------------------------
    # Nucleo de evaluacion
    # -------------------------------------------------------------------

    def _evaluate(self, camera: CameraState, payload: Dict[str, Any], key: str) -> List[DecisionResult]:
        now = float(payload["timestamp"])
        subject = str(payload.get("subject_id") or f"{payload['camera_id']}:{payload['track_id']}")
        facts = self._build_facts(payload)

        entry = HistoryEntry(
            result_key=key,
            episode_id=payload.get("episode_id"),
            behavior_type=str(payload["behavior_type"]),
            state=str(payload["state"]),
            timestamp=now,
            start_time=float(payload.get("start_time") or now - float(payload["duration"])),
            duration=float(payload["duration"]),
            zone_id=payload.get("zone_id"),
            facts=facts,
        )
        camera.latest_timestamp = max(camera.latest_timestamp, now)
        camera.last_wall_update = time.time()
        camera.record(subject, entry)
        history = list(camera.history.get(subject, ()))

        decisions: List[DecisionResult] = []
        rule_states: Dict[str, str] = {}
        suppressed: List[str] = []

        for rule in self.engine.rules:
            tracker = camera.trackers.setdefault((rule.rule_id, subject), RuleTracker())

            if tracker.state == EvaluationState.EVENT_GENERATED and now >= tracker.cooldown_until:
                tracker.transition(EvaluationState.NORMAL, now)
                tracker.match_count, tracker.first_match_time = 0, None

            if rule.rule_type == "simple" and not rule.is_relevant(entry.behavior_type):
                rule_states[rule.rule_id] = tracker.state.value
                continue

            match, partial = self._match_rule(rule, entry, facts, history)
            tracker.last_update = now

            if tracker.state == EvaluationState.EVENT_GENERATED:
                if match is not None:
                    self.metrics.add("eventos_descartados_por_cooldown")
                    suppressed.append(rule.rule_id)
                    _rate_limited_logger.log(
                        f"cooldown_{rule.rule_id}", logging.DEBUG,
                        f"Regla {rule.rule_id} en cooldown: evidencias repetidas descartadas (%d)",
                    )
                rule_states[rule.rule_id] = tracker.state.value
                continue

            if match is None:
                tracker.match_count, tracker.first_match_time = 0, None
                tracker.transition(EvaluationState.OBSERVATION if partial else EvaluationState.NORMAL, now)
                rule_states[rule.rule_id] = tracker.state.value
                continue

            tracker.match_count += 1
            tracker.first_match_time = tracker.first_match_time or now
            if tracker.match_count < rule.action.confirm_count:
                tracker.transition(EvaluationState.POSSIBLE_EVENT, now)
                rule_states[rule.rule_id] = tracker.state.value
                continue

            tracker.transition(EvaluationState.POSSIBLE_EVENT, now)
            tracker.transition(EvaluationState.EVENT_GENERATED, now)
            tracker.cooldown_until = now + rule.action.cooldown_seconds
            decision = self._build_event(payload, key, subject, facts, match, tracker)
            tracker.last_event_id = decision.event_id
            decisions.append(decision)
            rule_states[rule.rule_id] = tracker.state.value

            self.metrics.add("eventos_generados")
            self.metrics.rule_triggered(rule.rule_id)
            logger.info(
                "Regla %s hizo match -> %s prioridad=%s | camera=%s track=%s event_id=%s",
                rule.rule_id, rule.action.event_type, decision.priority.value if decision.priority else None,
                payload["camera_id"], payload["track_id"], decision.event_id,
            )
            if rule.stop_processing:
                break

        camera.prune(self.config.history_horizon_seconds)

        for decision in decisions:
            self._output.put(decision.to_dict())
        if decisions:
            return decisions

        no_event = self._no_event(payload, key, {"rule_states": rule_states, "suppressed_by_cooldown": suppressed})
        if self.config.publish_no_event:
            self._output.put(no_event.to_dict())
        return [no_event]

    def _match_rule(
        self,
        rule: Rule,
        entry: HistoryEntry,
        facts: Dict[str, Any],
        history: List[HistoryEntry],
    ) -> Tuple[Optional[_Match], bool]:
        """Devuelve (match completo | None, hubo match parcial)."""
        if rule.rule_type == "simple":
            ok, trace = rule.condition.evaluate(facts)  # type: ignore[union-attr]
            if ok:
                return _Match(rule, trace, [entry], [entry.behavior_type]), True
            partial = rule.scope.evaluate(facts)[0] if rule.scope is not None else False
            return None, partial
        return self._match_sequence(rule, entry, history)

    def _match_sequence(
        self,
        rule: Rule,
        entry: HistoryEntry,
        history: List[HistoryEntry],
    ) -> Tuple[Optional[_Match], bool]:
        """Emparejamiento hacia atras: la evidencia actual debe cumplir el
        ULTIMO paso; los pasos anteriores se buscan en la memoria del sujeto,
        en orden temporal estricto y dentro de window_seconds. El match
        parcial (algun paso inicial ya ocurrio dentro de la ventana) deja la
        regla en OBSERVATION."""
        steps = rule.steps
        window_start = entry.timestamp - rule.window_seconds
        in_window = [h for h in history if h.timestamp >= window_start]

        last_ok, last_trace = steps[-1].condition.evaluate(entry.facts)
        if last_ok:
            matched: List[Tuple[HistoryEntry, List[Dict[str, Any]]]] = [(entry, last_trace)]
            position = len(in_window) - 1
            while position >= 0 and in_window[position] is not entry:
                position -= 1
            complete = True
            for step_index in range(len(steps) - 2, -1, -1):
                later_entry = matched[0][0]
                later_step = steps[step_index + 1]
                found = None
                for candidate_index in range(position - 1, -1, -1):
                    candidate = in_window[candidate_index]
                    if candidate.timestamp > later_entry.timestamp:
                        continue
                    if later_step.require_zone_change and candidate.zone_id == later_entry.zone_id:
                        continue
                    ok, trace = steps[step_index].condition.evaluate(candidate.facts)
                    if ok:
                        found = (candidate_index, candidate, trace)
                        break
                if found is None:
                    complete = False
                    break
                position = found[0]
                matched.insert(0, (found[1], found[2]))
            if complete:
                trace = []
                for (matched_entry, step_trace), step in zip(matched, steps):
                    trace.append({"step": step.name, "result_key": matched_entry.result_key,
                                  "timestamp": matched_entry.timestamp, "zone_id": matched_entry.zone_id,
                                  "conditions": step_trace})
                return _Match(rule, trace, [m[0] for m in matched], [s.name for s in steps]), True

        partial = any(steps[0].condition.evaluate(h.facts)[0] for h in in_window)
        return None, partial

    # -------------------------------------------------------------------
    # Construccion de decisiones
    # -------------------------------------------------------------------

    def _build_facts(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        evidence = payload.get("evidence") or {}
        return {
            "behavior_type": payload["behavior_type"],
            "state": payload["state"],
            "duration": float(payload["duration"]),
            "timestamp": float(payload["timestamp"]),
            "start_time": payload.get("start_time"),
            "evidence_strength": payload.get("evidence_strength"),
            "observability": payload.get("observability"),
            "zone_id": payload.get("zone_id"),
            "camera_id": payload["camera_id"],
            "track_id": payload["track_id"],
            "subject_id": payload.get("subject_id"),
            "evidence": evidence if isinstance(evidence, dict) else {},
            "context": self.context.facts_for(str(payload["camera_id"]), payload.get("zone_id"), float(payload["timestamp"])),
        }

    def _resolve_priority(self, rule: Rule, facts: Dict[str, Any]) -> Tuple[Priority, List[Dict[str, Any]]]:
        priority = rule.action.priority
        applied: List[Dict[str, Any]] = []
        for override in rule.action.priority_overrides:
            ok, trace = override.when.evaluate(facts)
            if ok and override.priority.rank > priority.rank:
                applied.append({"escalated_to": override.priority.value, "conditions": trace})
                priority = override.priority
        return priority, applied

    def _compact_evidence(self, evidence: Any) -> Any:
        encoded = json.dumps(evidence, default=str)
        if len(encoded) <= self.config.max_evidence_bytes:
            return evidence
        return {"truncated": True, "size_bytes": len(encoded), "keys": sorted(evidence.keys()) if isinstance(evidence, dict) else None}

    def _build_event(
        self,
        payload: Dict[str, Any],
        key: str,
        subject: str,
        facts: Dict[str, Any],
        match: _Match,
        tracker: RuleTracker,
    ) -> DecisionResult:
        rule = match.rule
        priority, escalations = self._resolve_priority(rule, facts)
        window_start = min(e.start_time for e in match.entries) - rule.action.pre_roll_seconds
        window_end = float(payload["timestamp"]) + rule.action.post_roll_seconds

        # event_id determinista: el mismo BehaviorResult reprocesado (incluso
        # tras expulsarse del cache de idempotencia) produce el mismo id.
        event_id = str(uuid.uuid5(_EVENT_ID_NAMESPACE, f"{rule.rule_id}|{key}"))

        supporting = [
            {
                "result_key": e.result_key,
                "episode_id": e.episode_id,
                "behavior_type": e.behavior_type,
                "state": e.state,
                "timestamp": e.timestamp,
                "start_time": e.start_time,
                "duration": e.duration,
                "zone_id": e.zone_id,
                "zone_type": e.facts["context"].get("zone_type"),
                "evidence_strength": e.facts.get("evidence_strength"),
                "observability": e.facts.get("observability"),
                "evidence": self._compact_evidence(e.facts.get("evidence")),
            }
            for e in match.entries
        ]
        summary = (
            f"Regla '{rule.name}' ({rule.rule_id}) cumplida: "
            + " -> ".join(f"{e.behavior_type}/{e.state} {e.duration:.1f}s en zona {e.zone_id}" for e in match.entries)
            + f". Se emite {rule.action.event_type} con prioridad {priority.value}."
        )
        explanation = {
            "summary": summary,
            "rule": {
                "rule_id": rule.rule_id,
                "name": rule.name,
                "type": rule.rule_type,
                "rules_version": self.engine.version_hash,
                "definition": rule.raw,
            },
            "matched_conditions": match.trace,
            "priority": {"base": rule.action.priority.value, "final": priority.value, "escalations": escalations},
            "supporting_evidence": supporting,
            "context": facts["context"],
            "decision_path": list(tracker.path),
            "confirmations": tracker.match_count,
            "cooldown_until": tracker.cooldown_until,
        }
        return DecisionResult(
            event_id=event_id,
            camera_id=str(payload["camera_id"]),
            session_id=str(payload["session_id"]),
            track_id=str(payload["track_id"]),
            subject_id=subject,
            decision=Decision.EVENT_GENERATED,
            event_type=rule.action.event_type,
            priority=priority,
            timestamp=float(payload["timestamp"]),
            time_window_start=window_start,
            time_window_end=window_end,
            explanation=explanation,
            rule_id=rule.rule_id,
            source_result_keys=[e.result_key for e in match.entries],
        )

    def _no_event(self, payload: Dict[str, Any], key: str, detail: Dict[str, Any]) -> DecisionResult:
        return DecisionResult(
            event_id=str(uuid.uuid5(_EVENT_ID_NAMESPACE, f"NO_EVENT|{key}")),
            camera_id=str(payload.get("camera_id")),
            session_id=str(payload.get("session_id")),
            track_id=str(payload.get("track_id")),
            subject_id=str(payload.get("subject_id") or f"{payload.get('camera_id')}:{payload.get('track_id')}"),
            decision=Decision.NO_EVENT,
            event_type=None,
            priority=None,
            timestamp=float(payload.get("timestamp") or 0.0),
            time_window_start=None,
            time_window_end=None,
            explanation={"summary": "Ninguna regla de negocio se cumplio", **detail},
            source_result_keys=[key],
        )

    # -------------------------------------------------------------------
    # Workers, memoria y backpressure
    # -------------------------------------------------------------------

    def _camera_state(self, camera_id: str) -> CameraState:
        with self._cameras_lock:
            camera = self._cameras.get(camera_id)
            if camera is None:
                camera = CameraState(camera_id, self.config.history_max_per_subject)
                self._cameras[camera_id] = camera
            return camera

    def _worker(self, index: int) -> None:
        queue = self._queues[index]
        last_sweep = time.time()
        while not self._stop_event.is_set():
            payload = queue.get(timeout=0.5)
            if payload is not None:
                try:
                    self.process_behavior_result(payload)
                except Exception:
                    logger.exception("Error inesperado en DecisionWorker-%d (se continua)", index)
            if index == 0 and time.time() - last_sweep >= 30.0:
                self._sweep_idle_cameras()
                last_sweep = time.time()

    def _sweep_idle_cameras(self) -> None:
        """Libera camaras sin actividad cuyo cooldown mas largo ya vencio."""
        max_cooldown = max((r.action.cooldown_seconds for r in self.engine.rules), default=0.0)
        threshold = max(self.config.stale_camera_seconds, max_cooldown)
        now_wall = time.time()
        with self._cameras_lock:
            for camera_id in [c for c, s in self._cameras.items() if now_wall - s.last_wall_update > threshold]:
                del self._cameras[camera_id]
                logger.info("Memoria de camara inactiva liberada | camera_id=%s", camera_id)

    def _on_input_drop(self) -> None:
        self.metrics.add("descartes_por_backpressure")
        _rate_limited_logger.log("input_backpressure", logging.WARNING,
                                 "Cola de entrada de Decision llena: descartando lo mas antiguo (%d)")

    def _on_output_drop(self) -> None:
        self.metrics.add("descartes_por_backpressure")
        _rate_limited_logger.log("output_backpressure", logging.WARNING,
                                 "Cola de salida de Decision llena (consumidor lento): descartando lo mas antiguo (%d)")


# =============================================================================
# Smoke test manual
# =============================================================================

if __name__ == "__main__":
    context = ContextProvider()
    context.configure({
        "CAM-SEC-01": {
            "camera_type": "security", "site": "tienda_centro", "utc_offset_hours": -6,
            "zones": {"escritorio_guardia": {"zone_type": "security_desk"}},
            "schedules": {"night_shift": [["22:00", "06:00"]], "business_hours": [["09:00", "21:00"]]},
        },
        "CAM-001": {
            "camera_type": "retail", "site": "tienda_centro", "utc_offset_hours": -6,
            "zones": {"pasillo": {"zone_type": "aisle"}, "salida": {"zone_type": "exit"}},
            "schedules": {"business_hours": [["09:00", "21:00"]]},
        },
    })
    service = DecisionService(config=DecisionConfig(workers=2), context=context)
    service.start()

    # 23:30 hora local (UTC-6) -> turno nocturno.
    base = datetime(2026, 9, 24, 23, 30, tzinfo=timezone(timedelta(hours=-6))).timestamp()

    def behavior(camera: str, track: str, btype: str, state: str, t: float, start: float, zone: str,
                 episode: str, **evidence: Any) -> Dict[str, Any]:
        return {
            "result_id": f"{camera}-{track}-{btype}-{state}-{t}", "episode_id": episode,
            "camera_id": camera, "session_id": "s-demo", "track_id": track,
            "subject_id": f"{camera}:{track}", "timestamp": base + t, "behavior_type": btype,
            "state": state, "start_time": base + start, "duration": t - start, "last_update": base + t,
            "zone_id": zone, "evidence_strength": 0.85, "observability": 0.7, "evidence": evidence,
            "description": "",
        }

    print("\n--- Escenario 1: guardia con somnolencia (confirm_count=2, cooldown 600s) ---")
    dup = None
    for t in range(0, 40):
        state = "POSSIBLE" if t < 15 else "SUSTAINED"
        payload = behavior("CAM-SEC-01", "CAM-SEC-01-7", "somnolence", state, float(t), 0.0,
                           "escritorio_guardia", "ep-sleep", head_angle_deg=64.0, immobility_duration=float(t) + 4)
        for d in service.process_behavior_result(payload):
            if d.decision == Decision.EVENT_GENERATED:
                dup = payload
                print(f"t={t:>2}s {d.event_type} prioridad={d.priority.value} event_id={d.event_id}")
                print("   explicacion:", d.explanation["summary"])
                print("   escalamiento:", [e["escalated_to"] for e in d.explanation["priority"]["escalations"]])
                print("   ventana video:", round(d.time_window_start - base, 1), "->", round(d.time_window_end - base, 1))
                print("   ruta:", d.explanation["decision_path"])

    again = service.process_behavior_result(dup)  # type: ignore[arg-type]
    print("Idempotencia: reproceso devuelve el mismo event_id ->", again[0].event_id)

    print("\n--- Escenario 2: secuencia de posible robo ---")
    service.process_behavior_result(behavior("CAM-001", "CAM-001-3", "product_interaction", "SUSTAINED",
                                             100.0, 98.0, "pasillo", "ep-pi", roi_id="anaquel_1", hand="right"))
    service.process_behavior_result(behavior("CAM-001", "CAM-001-3", "product_interaction", "FINISHED",
                                             104.0, 98.0, "pasillo", "ep-pi", roi_id="anaquel_1"))
    for d in service.process_behavior_result(behavior("CAM-001", "CAM-001-3", "loitering", "POSSIBLE",
                                                      160.0, 160.0, "salida", "ep-loit", dwell_time=8.0)):
        print(d.decision.value, d.event_type, d.priority.value if d.priority else None)
        if d.decision == Decision.EVENT_GENERATED:
            print("   explicacion:", d.explanation["summary"])
            print("   pasos:", [(s["step"], s["zone_id"]) for s in d.explanation["matched_conditions"]])
            print("   ventana video:", round(d.time_window_start - base, 1), "->", round(d.time_window_end - base, 1))

    print("\n--- Escenario 3: somnolencia en pasillo (sin regla aplicable) ---")
    d = service.process_behavior_result(behavior("CAM-001", "CAM-001-9", "somnolence", "SUSTAINED",
                                                 200.0, 170.0, "pasillo", "ep-x"))[0]
    print(d.decision.value, d.explanation["rule_states"])

    print("\n--- Escenario 4: regla invalida rechazada, conjunto vigente intacto ---")
    try:
        service.reload_rules([{"rule_id": "BAD", "condition": {"field": "duration", "op": "gt", "value": "x"},
                               "action": {"event_type": "X"}}])
    except RuleValidationError as exc:
        print("Rechazada:", exc)

    print("\n--- Escenario 5: entrada asincrona multi-camara ---")
    service.submit(behavior("CAM-SEC-01", "CAM-SEC-01-8", "somnolence", "SUSTAINED", 300.0, 280.0,
                            "escritorio_guardia", "ep-2"))
    service.submit(behavior("CAM-SEC-01", "CAM-SEC-01-8", "somnolence", "SUSTAINED", 301.0, 280.0,
                            "escritorio_guardia", "ep-2"))
    time.sleep(0.5)
    while True:
        out = service.get_next_decision(timeout=0.2)
        if out is None:
            break
        print("salida ->", out["event_type"], out["priority"], out["track_id"], out["event_id"])

    health = service.health_check()
    print("\nHEALTH:", json.dumps({k: health[k] for k in ("rule_engine", "temporal_cache", "metrics")}, indent=1, default=str))
    service.stop()
