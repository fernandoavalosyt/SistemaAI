"""
visualizerService.py

Servicio 10: Visualizador "Byrack".

Compone sobre el frame ORIGINAL (Servicio 1) la informacion que produce el
resto del pipeline y la publica como stream MJPEG por HTTP:
  - Cajas, IDs y estado de cada track        (Servicio 4 - Tracking)
  - Esqueleto de cada persona                (Servicio 5 - Pose)
  - Comportamientos en curso y su duracion   (Servicio 6 - Behavior)
  - Banner de decisiones / eventos           (Servicio 7 - Decision)
  - Estado de la evidencia de video          (Servicio 8 - Evidence)
  - Zonas configuradas                       (config de Behavior/Decision)

Sincronizacion: los resultados llegan con la latencia del pipeline, asi que
las cajas y el esqueleto se dibujan sobre el MISMO frame que los origino
(se guarda un historial corto de frames por frame_id). Si el pipeline se
atrasa demasiado, se muestra el video en vivo sin overlays y un aviso.

Endpoints:
  GET /                 Pagina de visualizacion
  GET /stream           MJPEG (?camera=CAM-01). Uso: <img src=".../stream">
  GET /snapshot.jpg     Un frame compuesto
  GET /api/state        Estado resumido (tracks, comportamientos, eventos)
  GET /health           Salud agregada del pipeline (status, version, ...)

Solo observa: no modifica ni retroalimenta al pipeline.
"""

from __future__ import annotations

import html
import json
import logging
import os
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np
from flask import Flask, Response, request
from werkzeug.serving import make_server


logger = logging.getLogger("visualizer_service")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
logging.getLogger("werkzeug").setLevel(logging.WARNING)


# =============================================================================
# Catalogos de presentacion (ASCII: las fuentes Hershey de OpenCV no tienen acentos)
# =============================================================================

BEHAVIOR_LABELS = {"somnolence": "Somnolencia", "loitering": "Merodeo", "product_interaction": "Interaccion producto"}
PHASE_LABELS = {"POSSIBLE": "posible", "SUSTAINED": "SOSTENIDO", "FINISHED": "finalizado"}
EVENT_LABELS = {
    "POSSIBLE_THEFT": "POSIBLE ROBO",
    "SLEEPING_GUARD": "GUARDIA DORMIDO",
    "SUSPICIOUS_BEHAVIOR": "COMPORTAMIENTO SOSPECHOSO",
}
PRIORITY_LABELS = {"CRITICAL": "CRITICA", "HIGH": "ALTA", "MEDIUM": "MEDIA", "LOW": "BAJA"}

# Colores BGR
PRIORITY_BGR = {"CRITICAL": (68, 68, 239), "HIGH": (22, 115, 249), "MEDIUM": (8, 179, 234), "LOW": (248, 189, 56)}
TRACK_STATE_BGR = {"ACTIVE": (94, 197, 34), "NEW": (248, 189, 56), "RECOVERING": (8, 179, 234), "LOST": (130, 130, 130)}
PHASE_BGR = {"POSSIBLE": (8, 179, 234), "SUSTAINED": (68, 68, 239), "FINISHED": (150, 150, 150)}
ZONE_BGR = {"exit": (68, 68, 239), "restricted": (68, 68, 239), "checkout": (94, 197, 34),
            "shelf": (248, 189, 56), "product": (248, 189, 56), "security_desk": (200, 120, 240)}
ALERT_BGR = (60, 60, 235)
SKELETON_BGR = (255, 230, 120)

COCO_EDGES: Tuple[Tuple[str, str], ...] = (
    ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"), ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"), ("left_hip", "right_hip"), ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"), ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("nose", "left_eye"), ("nose", "right_eye"), ("left_eye", "left_ear"), ("right_eye", "right_ear"),
)

FONT = cv2.FONT_HERSHEY_SIMPLEX


# =============================================================================
# Configuracion
# =============================================================================

@dataclass(frozen=True)
class VisualizerConfig:
    host: str = "127.0.0.1"
    port: int = 8001
    render_fps: float = 10.0
    jpeg_quality: int = 80
    max_width: int = 1280
    frame_history: int = 90            # frames originales guardados para sincronizar overlays
    result_history: int = 40           # resultados de tracking/pose guardados por frame_id
    pose_max_lag_frames: int = 5       # cuanto esperar la pose antes de dibujar sin esqueleto
    pipeline_stale_seconds: float = 3.0
    banner_seconds: float = 12.0
    behavior_linger_seconds: float = 3.0
    cors_origin: str = "*"
    title: str = "BYRACK"
    alerts_url: Optional[str] = None

    @staticmethod
    def from_env() -> "VisualizerConfig":
        return VisualizerConfig(
            host=os.getenv("VISUALIZER_HOST", "127.0.0.1"),
            port=int(os.getenv("VISUALIZER_PORT", "8001")),
            render_fps=float(os.getenv("VISUALIZER_FPS", "10")),
            max_width=int(os.getenv("VISUALIZER_MAX_WIDTH", "1280")),
            cors_origin=os.getenv("VISUALIZER_CORS_ORIGIN", "*"),
            alerts_url=os.getenv("VISUALIZER_ALERTS_URL") or None,
        )


# =============================================================================
# Estado por camara
# =============================================================================

class _CameraView:
    def __init__(self, camera_id: str) -> None:
        self.camera_id = camera_id
        self.frames: "OrderedDict[int, Tuple[float, np.ndarray]]" = OrderedDict()
        self.tracking: "OrderedDict[int, List[Dict[str, Any]]]" = OrderedDict()
        self.poses: "OrderedDict[int, Dict[str, Dict[str, Any]]]" = OrderedDict()
        self.behaviors: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.banners: Deque[Dict[str, Any]] = deque(maxlen=5)
        self.evidence: Deque[Dict[str, Any]] = deque(maxlen=4)
        self.zones: List[Dict[str, Any]] = []
        self.last_frame_wall = 0.0
        self.last_pose_wall = 0.0
        self.input_fps = 0.0
        self._fps_count = 0
        self._fps_window = time.time()
        self.latest_jpeg: Optional[bytes] = None
        self.version = 0

    def register_frame_rate(self) -> None:
        self._fps_count += 1
        elapsed = time.time() - self._fps_window
        if elapsed >= 1.0:
            self.input_fps = self._fps_count / elapsed
            self._fps_count = 0
            self._fps_window = time.time()


def _trim(history: "OrderedDict[int, Any]", limit: int) -> None:
    while len(history) > limit:
        history.popitem(last=False)


# =============================================================================
# Dibujo
# =============================================================================

def _label(img: np.ndarray, text: str, org: Tuple[int, int], color: Tuple[int, int, int],
           scale: float, thickness: int, text_color: Tuple[int, int, int] = (255, 255, 255)) -> int:
    """Texto con fondo solido. Devuelve la altura ocupada."""
    (tw, th), baseline = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = org
    pad = max(2, int(4 * scale))
    x = max(0, min(x, img.shape[1] - tw - 2 * pad))
    y = max(th + 2 * pad, y)
    cv2.rectangle(img, (x, y - th - 2 * pad), (x + tw + 2 * pad, y + baseline), color, -1)
    cv2.putText(img, text, (x + pad, y - pad), FONT, scale, text_color, thickness, cv2.LINE_AA)
    return th + 2 * pad + baseline


def _blend_rect(img: np.ndarray, p1: Tuple[int, int], p2: Tuple[int, int], color: Tuple[int, int, int], alpha: float) -> None:
    x1, y1 = max(0, p1[0]), max(0, p1[1])
    x2, y2 = min(img.shape[1], p2[0]), min(img.shape[0], p2[1])
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    overlay = np.full_like(roi, color)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)


def _no_signal_frame(camera_id: str, seconds_since: Optional[float], title: str) -> np.ndarray:
    img = np.full((720, 1280, 3), (23, 17, 11), dtype=np.uint8)
    for y in range(0, 720, 4):
        img[y:y + 1, :] = (30, 24, 18)
    cv2.putText(img, "SIN SENAL", (470, 330), FONT, 2.0, (140, 150, 170), 4, cv2.LINE_AA)
    detail = "esperando el primer frame" if seconds_since is None else f"ultimo frame hace {seconds_since:.0f} s"
    cv2.putText(img, f"{camera_id}  |  {detail}", (430, 390), FONT, 0.8, (120, 130, 150), 2, cv2.LINE_AA)
    cv2.putText(img, title, (24, 44), FONT, 1.0, (248, 189, 56), 2, cv2.LINE_AA)
    return img


# =============================================================================
# Servicio
# =============================================================================

class VisualizerService:
    """Recibe resultados del pipeline (metodos on_*), compone los overlays en
    un hilo de render y los sirve por HTTP. Los metodos on_* son baratos y
    thread-safe: solo guardan referencias; el dibujo ocurre en el render."""

    def __init__(self, config: Optional[VisualizerConfig] = None,
                 health_provider: Optional[Callable[[], Dict[str, Any]]] = None) -> None:
        self.config = config or VisualizerConfig.from_env()
        self._health_provider = health_provider
        self._views: "OrderedDict[str, _CameraView]" = OrderedDict()
        self._lock = threading.RLock()
        self._frame_cond = threading.Condition()
        self._clients = 0
        self._stop = threading.Event()
        self._server = None
        self._server_thread: Optional[threading.Thread] = None
        self._render_thread: Optional[threading.Thread] = None
        self._render_ms = 0.0

    # -------------------------------------------------------------------
    # Entradas desde el pipeline
    # -------------------------------------------------------------------

    def _view(self, camera_id: str) -> _CameraView:
        with self._lock:
            view = self._views.get(camera_id)
            if view is None:
                view = _CameraView(camera_id)
                self._views[camera_id] = view
            return view

    def set_zones(self, camera_id: str, zones: List[Dict[str, Any]]) -> None:
        with self._lock:
            self._view(camera_id).zones = list(zones)

    def on_frame(self, frame: Dict[str, Any]) -> None:
        view = self._view(str(frame["camera_id"]))
        with self._lock:
            view.frames[int(frame["frame_id"])] = (float(frame["timestamp"]), frame["image"])
            _trim(view.frames, self.config.frame_history)
            view.last_frame_wall = time.time()
            view.register_frame_rate()

    def on_tracking(self, result: Dict[str, Any]) -> None:
        view = self._view(str(result["camera_id"]))
        with self._lock:
            view.tracking[int(result["frame_id"])] = list(result.get("tracks") or [])
            _trim(view.tracking, self.config.result_history)

    def on_pose(self, result: Dict[str, Any]) -> None:
        view = self._view(str(result["camera_id"]))
        poses = {str(p["track_id"]): p for p in result.get("poses") or [] if p.get("status") == "OK"}
        with self._lock:
            view.poses[int(result["frame_id"])] = poses
            _trim(view.poses, self.config.result_history)
            view.last_pose_wall = time.time()

    def on_behavior(self, result: Dict[str, Any]) -> None:
        view = self._view(str(result["camera_id"]))
        with self._lock:
            view.behaviors.setdefault(str(result["track_id"]), {})[str(result["behavior_type"])] = {
                "state": result.get("state"),
                "duration": float(result.get("duration") or 0.0),
                "strength": result.get("evidence_strength"),
                "wall": time.time(),
            }

    def on_decision(self, decision: Dict[str, Any]) -> None:
        if decision.get("decision") != "EVENT_GENERATED":
            return
        view = self._view(str(decision["camera_id"]))
        with self._lock:
            view.banners.append({
                "event_id": decision.get("event_id"),
                "event_type": decision.get("event_type"),
                "priority": decision.get("priority"),
                "track_id": str(decision.get("track_id")),
                "expires": time.time() + self.config.banner_seconds,
            })

    def on_evidence(self, evidence: Dict[str, Any]) -> None:
        view = self._view(str(evidence["camera_id"]))
        with self._lock:
            view.evidence.appendleft({
                "event_type": evidence.get("event_type"),
                "status": evidence.get("status"),
                "format": evidence.get("format"),
                "wall": time.time(),
            })

    # -------------------------------------------------------------------
    # Ciclo de vida
    # -------------------------------------------------------------------

    def start(self) -> None:
        app = self._build_app()
        self._server = make_server(self.config.host, self.config.port, app, threaded=True)
        self._server_thread = threading.Thread(target=self._server.serve_forever, name="VisualizerHTTP", daemon=True)
        self._server_thread.start()
        self._render_thread = threading.Thread(target=self._render_loop, name="VisualizerRender", daemon=True)
        self._render_thread.start()
        logger.info("Visualizador escuchando en http://%s:%d (stream en /stream)", self.config.host, self.config.port)

    def stop(self) -> None:
        self._stop.set()
        with self._frame_cond:
            self._frame_cond.notify_all()
        if self._server is not None:
            self._server.shutdown()
        logger.info("Visualizador detenido")

    # -------------------------------------------------------------------
    # Render
    # -------------------------------------------------------------------

    def _render_loop(self) -> None:
        interval = 1.0 / max(1.0, self.config.render_fps)
        while not self._stop.is_set():
            started = time.time()
            with self._lock:
                active = self._clients > 0
                cameras = list(self._views.keys())
            if active:
                for camera_id in cameras:
                    try:
                        jpeg = self.render_jpeg(camera_id)
                    except Exception:
                        logger.exception("Error renderizando la camara %s", camera_id)
                        continue
                    view = self._view(camera_id)
                    with self._frame_cond:
                        view.latest_jpeg = jpeg
                        view.version += 1
                        self._frame_cond.notify_all()
                self._render_ms = (time.time() - started) * 1000.0
            self._stop.wait(max(0.0, interval - (time.time() - started)))

    def _snapshot(self, view: _CameraView) -> Dict[str, Any]:
        """Selecciona el frame a mostrar y copia (bajo lock) todo lo que se
        dibujara, para no retener el lock durante el dibujo."""
        now = time.time()
        cfg = self.config
        with self._lock:
            if not view.frames:
                return {"image": None, "since": None if view.last_frame_wall == 0 else now - view.last_frame_wall}

            latest_fid = next(reversed(view.frames))
            latest_ts = view.frames[latest_fid][0]
            pose_active = bool(view.poses) and now - view.last_pose_wall < 2.0
            track_fids = [f for f in reversed(view.tracking.keys()) if f in view.frames]
            chosen: Optional[int] = None
            if track_fids:
                newest = track_fids[0]
                for fid in track_fids:
                    if not pose_active or fid in view.poses or newest - fid >= cfg.pose_max_lag_frames:
                        chosen = fid
                        break

            lagging = chosen is not None and latest_ts - view.frames[chosen][0] > cfg.pipeline_stale_seconds
            if chosen is None or lagging:
                fid, tracks, poses = latest_fid, None, {}
            else:
                fid, tracks, poses = chosen, view.tracking[chosen], view.poses.get(chosen, {})

            for track_id in list(view.behaviors):
                entries = view.behaviors[track_id]
                for btype in list(entries):
                    entry = entries[btype]
                    if entry["state"] == "FINISHED" and now - entry["wall"] > cfg.behavior_linger_seconds:
                        del entries[btype]
                    elif now - entry["wall"] > 30.0:  # sin heartbeat: el episodio ya no se reporta
                        del entries[btype]
                if not entries:
                    del view.behaviors[track_id]

            while view.banners and view.banners[0]["expires"] < now:
                view.banners.popleft()

            return {
                "image": view.frames[fid][1],
                "frame_id": fid,
                "frame_ts": view.frames[fid][0],
                "tracks": tracks,
                "poses": dict(poses),
                "behaviors": {t: {b: dict(e) for b, e in bs.items()} for t, bs in view.behaviors.items()},
                "banners": list(view.banners),
                "evidence": list(view.evidence),
                "zones": list(view.zones),
                "input_fps": view.input_fps,
                "lagging": lagging,
                "since": now - view.last_frame_wall,
            }

    def render_jpeg(self, camera_id: str) -> bytes:
        view = self._view(camera_id)
        state = self._snapshot(view)
        stale = state["since"] is not None and state["since"] > self.config.pipeline_stale_seconds
        if state["image"] is None or stale:
            img = _no_signal_frame(camera_id, state["since"], self.config.title)
        else:
            img = self._draw(camera_id, state)
        height, width = img.shape[:2]
        if width > self.config.max_width:
            img = cv2.resize(img, (self.config.max_width, int(height * self.config.max_width / width)),
                             interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), self.config.jpeg_quality])
        if not ok:
            raise RuntimeError("No se pudo codificar el frame del visualizador")
        return encoded.tobytes()

    def _draw(self, camera_id: str, state: Dict[str, Any]) -> np.ndarray:
        img = state["image"].copy()  # el frame original lo comparten otros servicios: nunca dibujar sobre el
        height, width = img.shape[:2]
        k = max(0.6, width / 1280.0)
        thick = max(1, int(round(2 * k)))
        small = 0.5 * k
        now = time.time()
        alerted = {b["track_id"] for b in state["banners"]}

        # Zonas
        for zone in state["zones"]:
            polygon = np.array(zone["polygon"], dtype=np.int32)
            color = ZONE_BGR.get(str(zone.get("zone_type") or zone.get("kind")), (200, 200, 200))
            overlay = img.copy()
            cv2.fillPoly(overlay, [polygon], color)
            cv2.addWeighted(overlay, 0.12, img, 0.88, 0, dst=img)
            cv2.polylines(img, [polygon], True, color, max(1, thick - 1), cv2.LINE_AA)
            # Etiqueta en el centro: las esquinas chocan con el HUD y los banners.
            cx, cy = polygon.mean(axis=0)
            _label(img, str(zone.get("name") or zone["zone_id"]).upper(), (int(cx) - int(40 * k), int(cy)),
                   color, small * 0.9, max(1, thick - 1))

        # Tracks, esqueletos y comportamientos
        for track in state["tracks"] or []:
            track_state = str(track.get("track_state"))
            if track_state == "TERMINATED":
                continue
            bbox = track.get("bbox") or {}
            try:
                x1, y1, x2, y2 = (int(round(float(bbox[c]))) for c in ("x1", "y1", "x2", "y2"))
            except (KeyError, TypeError, ValueError):
                continue
            track_id = str(track.get("track_id"))
            is_alert = track_id in alerted
            color = ALERT_BGR if is_alert else TRACK_STATE_BGR.get(track_state, (200, 200, 200))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thick * (2 if is_alert else 1) if track_state != "LOST" else 1)

            short_id = track_id.rsplit("-", 1)[-1]
            conf = track.get("detection_confidence")
            text = f"ID {short_id} {track.get('class_name', '')}" + (f" {float(conf):.2f}" if conf is not None else "")
            if track_state != "ACTIVE":
                text += f" [{track_state}]"
            _label(img, text, (x1, y1 - 2), color, small, max(1, thick - 1))

            pose = state["poses"].get(track_id)
            if pose:
                self._draw_skeleton(img, pose.get("keypoints") or {}, thick, k)

            y_cursor = y2 + int(4 * k)
            for btype, entry in (state["behaviors"].get(track_id) or {}).items():
                duration = entry["duration"] + (0.0 if entry["state"] == "FINISHED" else now - entry["wall"])
                text = f"{BEHAVIOR_LABELS.get(btype, btype)} {PHASE_LABELS.get(entry['state'], entry['state'])} {duration:.1f}s"
                y_cursor += _label(img, text, (x1, y_cursor + int(18 * k)), PHASE_BGR.get(entry["state"], (150, 150, 150)),
                                   small, max(1, thick - 1))

        self._draw_hud(img, camera_id, state, k, thick)
        self._draw_banners(img, state["banners"], k, thick)
        self._draw_evidence(img, state["evidence"], k, thick)
        return img

    @staticmethod
    def _draw_skeleton(img: np.ndarray, keypoints: Dict[str, Any], thick: int, k: float) -> None:
        points: Dict[str, Tuple[int, int]] = {}
        for name, kp in keypoints.items():
            if isinstance(kp, dict) and kp.get("visible") and kp.get("x") is not None and kp.get("y") is not None:
                points[name] = (int(round(kp["x"])), int(round(kp["y"])))
        for a, b in COCO_EDGES:
            if a in points and b in points:
                cv2.line(img, points[a], points[b], SKELETON_BGR, max(1, thick - 1), cv2.LINE_AA)
        for point in points.values():
            cv2.circle(img, point, max(2, int(3 * k)), (255, 255, 255), -1, cv2.LINE_AA)

    def _draw_hud(self, img: np.ndarray, camera_id: str, state: Dict[str, Any], k: float, thick: int) -> None:
        tracks = [t for t in state["tracks"] or [] if t.get("track_state") not in ("TERMINATED", "LOST")]
        latency_ms = (time.time() - state["frame_ts"]) * 1000.0
        lines = [
            f"{self.config.title}  |  {camera_id}  |  {time.strftime('%H:%M:%S')}",
            f"video {state['input_fps']:.1f} fps  |  tracks {len(tracks)}  |  retraso {latency_ms:.0f} ms",
        ]
        if state["lagging"]:
            lines.append("PIPELINE RETRASADO: video en vivo sin overlays")
        scale = 0.55 * k
        line_h = int(26 * k)
        panel_w = int(max(cv2.getTextSize(l, FONT, scale, 1)[0][0] for l in lines) + 28 * k)
        _blend_rect(img, (0, 0), (panel_w, line_h * len(lines) + int(16 * k)), (15, 12, 10), 0.62)
        for i, line in enumerate(lines):
            color = (248, 189, 56) if i == 0 else ((80, 80, 240) if "RETRASADO" in line else (225, 230, 235))
            cv2.putText(img, line, (int(14 * k), int(26 * k) + i * line_h), FONT, scale, color,
                        max(1, thick - 1), cv2.LINE_AA)

    @staticmethod
    def _draw_banners(img: np.ndarray, banners: List[Dict[str, Any]], k: float, thick: int) -> None:
        width = img.shape[1]
        bar_h = int(46 * k)
        for i, banner in enumerate(reversed(banners[-2:])):
            priority = str(banner.get("priority"))
            color = PRIORITY_BGR.get(priority, ALERT_BGR)
            top = img.shape[0] - bar_h * (i + 1) - int(8 * k) * i
            _blend_rect(img, (0, top), (width, top + bar_h), color, 0.85)
            event = EVENT_LABELS.get(str(banner.get("event_type")), str(banner.get("event_type")))
            short_id = str(banner.get("track_id")).rsplit("-", 1)[-1]
            text = f"ALERTA: {event}  |  ID {short_id}  |  PRIORIDAD {PRIORITY_LABELS.get(priority, priority)}"
            cv2.putText(img, text, (int(18 * k), top + int(31 * k)), FONT, 0.8 * k, (255, 255, 255),
                        max(2, thick), cv2.LINE_AA)

    @staticmethod
    def _draw_evidence(img: np.ndarray, evidence: List[Dict[str, Any]], k: float, thick: int) -> None:
        if not evidence:
            return
        scale = 0.48 * k
        line_h = int(22 * k)
        lines = []
        for item in evidence[:3]:
            event = EVENT_LABELS.get(str(item.get("event_type")), str(item.get("event_type")))
            detail = f"{item.get('status')}" + (f" ({item.get('format')})" if item.get("format") else "")
            lines.append(f"EVIDENCIA {event}: {detail}")
        panel_w = int(max(cv2.getTextSize(l, FONT, scale, 1)[0][0] for l in lines) + 24 * k)
        right = img.shape[1]
        _blend_rect(img, (right - panel_w, 0), (right, line_h * len(lines) + int(12 * k)), (15, 12, 10), 0.62)
        for i, line in enumerate(lines):
            color = (94, 197, 34) if "READY" in line else (80, 80, 240)
            cv2.putText(img, line, (right - panel_w + int(12 * k), int(20 * k) + i * line_h), FONT, scale, color,
                        max(1, thick - 1), cv2.LINE_AA)

    # -------------------------------------------------------------------
    # HTTP
    # -------------------------------------------------------------------

    def _stream(self, camera_id: str) -> Iterator[bytes]:
        view = self._view(camera_id)
        with self._lock:
            self._clients += 1
        last_version = -1
        try:
            while not self._stop.is_set():
                with self._frame_cond:
                    self._frame_cond.wait_for(lambda: view.version != last_version or self._stop.is_set(), timeout=1.0)
                    jpeg, version = view.latest_jpeg, view.version
                if jpeg is None or version == last_version:
                    continue
                last_version = version
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(jpeg)).encode("ascii") + b"\r\n\r\n" + jpeg + b"\r\n")
        finally:
            with self._lock:
                self._clients -= 1

    def _state_summary(self) -> Dict[str, Any]:
        now = time.time()
        with self._lock:
            cameras = {}
            for camera_id, view in self._views.items():
                latest_tracks = view.tracking[next(reversed(view.tracking))] if view.tracking else []
                cameras[camera_id] = {
                    "input_fps": round(view.input_fps, 2),
                    "seconds_since_frame": round(now - view.last_frame_wall, 2) if view.last_frame_wall else None,
                    "tracks": [
                        {"track_id": t.get("track_id"), "class_name": t.get("class_name"), "state": t.get("track_state")}
                        for t in latest_tracks if t.get("track_state") != "TERMINATED"
                    ],
                    "behaviors": {t: {b: {"state": e["state"], "duration": round(e["duration"], 2)} for b, e in bs.items()}
                                  for t, bs in view.behaviors.items()},
                    "active_alerts": [{k: v for k, v in b.items() if k != "expires"} for b in view.banners if b["expires"] > now],
                    "recent_evidence": [{k: v for k, v in e.items() if k != "wall"} for e in view.evidence],
                }
            return {"cameras": cameras, "stream_clients": self._clients, "render_ms": round(self._render_ms, 1)}

    def _json(self, payload: Any, status: int = 200) -> Response:
        response = Response(json.dumps(payload, default=str), status=status, mimetype="application/json")
        response.headers["Access-Control-Allow-Origin"] = self.config.cors_origin
        response.headers["Cache-Control"] = "no-store"
        return response

    def _resolve_camera(self) -> Optional[str]:
        requested = request.args.get("camera")
        with self._lock:
            if requested:
                return requested if requested in self._views else None
            return next(iter(self._views), None)

    def _build_app(self) -> Flask:
        app = Flask("byrack_visualizer")

        @app.after_request
        def _security_headers(response: Response) -> Response:
            response.headers.setdefault("X-Content-Type-Options", "nosniff")
            return response

        @app.get("/")
        def index() -> Response:
            with self._lock:
                cameras = list(self._views.keys())
            return Response(self._index_html(cameras), mimetype="text/html")

        @app.get("/stream")
        def stream() -> Response:
            camera_id = self._resolve_camera()
            if camera_id is None:
                return self._json({"error": "camera_not_found"}, 404)
            response = Response(self._stream(camera_id), mimetype="multipart/x-mixed-replace; boundary=frame")
            response.headers["Cache-Control"] = "no-store"
            return response

        @app.get("/snapshot.jpg")
        def snapshot() -> Response:
            camera_id = self._resolve_camera()
            if camera_id is None:
                return self._json({"error": "camera_not_found"}, 404)
            response = Response(self.render_jpeg(camera_id), mimetype="image/jpeg")
            response.headers["Cache-Control"] = "no-store"
            return response

        @app.get("/api/state")
        def api_state() -> Response:
            return self._json(self._state_summary())

        @app.get("/health")
        def health() -> Response:
            payload: Dict[str, Any] = {"status": "healthy", "version": "1.0.0"}
            if self._health_provider is not None:
                try:
                    payload = self._health_provider()
                except Exception as exc:
                    payload = {"status": "degraded", "version": "1.0.0", "error": str(exc)}
            payload["visualizer"] = {"stream_clients": self._clients, "render_ms": round(self._render_ms, 1)}
            return self._json(payload)

        return app

    def _index_html(self, cameras: List[str]) -> str:
        options = "".join(f'<option value="{html.escape(c)}">{html.escape(c)}</option>' for c in cameras)
        first = html.escape(cameras[0]) if cameras else ""
        alerts = (f'<a href="{html.escape(self.config.alerts_url)}" target="_blank" rel="noopener">Alertas</a>'
                  if self.config.alerts_url else "")
        return f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(self.config.title)} · Visualizador</title>
<style>
:root{{--bg:#0b0f17;--panel:#111827;--line:#1f2a3a;--text:#e5e9f0;--muted:#8b96a8;--accent:#38bdf8}}
*{{box-sizing:border-box;margin:0}}html,body{{height:100%}}
body{{background:var(--bg);color:var(--text);font-family:"Segoe UI",system-ui,sans-serif;display:grid;grid-template-rows:52px minmax(0,1fr);overflow:hidden}}
header{{display:flex;align-items:center;gap:16px;padding:0 20px;border-bottom:1px solid var(--line)}}
header b{{letter-spacing:.5px}}header span{{color:var(--muted);font-size:13px}}
header nav{{margin-left:auto;display:flex;gap:14px;align-items:center;font-size:13px}}
a{{color:var(--accent);text-decoration:none}}select{{background:var(--panel);color:var(--text);border:1px solid var(--line);border-radius:6px;padding:4px 8px}}
main{{display:grid;place-items:center;padding:14px;min-height:0}}
img{{max-width:100%;max-height:100%;border-radius:10px;border:1px solid var(--line);background:#000}}
</style></head><body>
<header><b>{html.escape(self.config.title)}</b><span>Visualizador del pipeline de IA</span>
<nav><select id="cam">{options}</select><a href="/api/state" target="_blank">Estado</a><a href="/health" target="_blank">Health</a>{alerts}</nav></header>
<main><img id="view" src="/stream?camera={first}" alt="Stream del visualizador"></main>
<script>document.getElementById('cam').addEventListener('change',function(e){{document.getElementById('view').src='/stream?camera='+encodeURIComponent(e.target.value);}});</script>
</body></html>"""
