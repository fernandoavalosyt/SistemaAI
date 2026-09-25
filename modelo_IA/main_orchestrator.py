"""
main_orchestrator.py

Orquestador central de modelo_IA. Instancia los 9 servicios del pipeline,
los conecta con hilos "pump" (salida de un servicio -> entrada de los
siguientes) y levanta el Visualizador Byrack (Servicio 10).

Flujo y fan-out:
    S1 Video ──┬─> S2 Preprocessing ─> S3 YOLO ─> S4 Tracking ──┬─> S5 Pose ─┐
               ├─> S5 (cache de frames originales)              └─> S6 <─────┘
               ├─> S8 (ring buffer de video)                        │
               └─> S10 Visualizador                                 v
                          S9 Alertas <─ S8 Evidencia <─ S7 Decision

Cada pump es un hilo independiente: una etapa lenta nunca bloquea la
lectura de las demas (cada servicio ya descarta lo mas antiguo en sus colas).

Uso local (desde cualquier carpeta):   python modelo_IA/main_orchestrator.py
Docker:                                docker compose up --build modelo-ia

Configuracion 100% por variables de entorno (ver modelo_IA/.env.example).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import sys
import threading
import time
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Tuple

from IA.alert_service.alertService import AlertConfig, AlertService
from IA.behavior_analysis.behaviorService.behaviorService import BehaviorConfig, BehaviorService
from IA.decision_engine.decisionService.decisionService import DecisionConfig, DecisionService
from IA.event_evidence.evidenceService.evidenceService import EvidenceConfig, EvidenceService
from IA.pose_specialized.poseService.poseService import PoseConfig, PoseService
from IA.preprocessing.preprocessingService.preprocessingService import CameraProcessingConfig, PreprocessingService
from IA.tracking.trackingService.trackingService import TrackConfig, TrackingService
from IA.video_ingestion.videoService import CameraWorker, build_camera_worker_from_env
from IA.visualizer.visualizerService import VisualizerConfig, VisualizerService
from IA.yolo_detection.yoloDetectionService.yoloDetectionService import YoloDetectionService

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_VERSION = "1.0.0"

logger = logging.getLogger("orchestrator")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "si")


# =============================================================================
# Pesos de los modelos
# =============================================================================

def ensure_model_file(env_var: str, default_path: str) -> str:
    """Garantiza que exista el archivo de pesos. Si falta y
    AUTO_DOWNLOAD_MODELS=true (default), descarga el asset oficial de
    Ultralytics con ese nombre (ej. yolov8n.pt, yolov8n-pose.pt)."""
    path = os.getenv(env_var) or default_path
    os.environ[env_var] = path
    if os.path.isfile(path):
        return path
    if not _env_bool("AUTO_DOWNLOAD_MODELS", True):
        raise SystemExit(f"Falta el modelo '{path}' ({env_var}) y AUTO_DOWNLOAD_MODELS=false")

    logger.info("Pesos no encontrados, descargando %s ...", path)
    from ultralytics.utils.downloads import attempt_download_asset

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    downloaded = attempt_download_asset(path)
    # Ultralytics puede devolver una copia ya existente en su carpeta global.
    if downloaded and os.path.abspath(downloaded) != os.path.abspath(path) and os.path.isfile(downloaded):
        shutil.copyfile(downloaded, path)
    if not os.path.isfile(path):
        raise SystemExit(f"No se pudo obtener el modelo '{path}'. Descargalo manualmente a esa ruta.")
    return path


# =============================================================================
# Pumps: conectores entre servicios
# =============================================================================

Sink = Tuple[str, Callable[[Dict[str, Any]], Any]]


class Pump(threading.Thread):
    """Lee de la salida de un servicio y entrega cada item a uno o varios
    destinos. El fallo de un destino no afecta a los demas."""

    def __init__(self, name: str, source: Callable[[float], Optional[Dict[str, Any]]], sinks: List[Sink],
                 stop_event: threading.Event) -> None:
        super().__init__(name=f"pump-{name}", daemon=True)
        self.label = name
        self._source = source
        self._sinks = sinks
        self._stop_event = stop_event
        self.moved = 0
        self.errors: Dict[str, int] = {sink_name: 0 for sink_name, _ in sinks}
        self.last_item_wall: Optional[float] = None

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._source(0.5)
            except Exception:
                logger.exception("La fuente del pump %s fallo", self.label)
                self._stop_event.wait(0.5)
                continue
            if item is None:
                continue
            self.moved += 1
            self.last_item_wall = time.time()
            for sink_name, sink in self._sinks:
                try:
                    sink(item)
                except Exception:
                    self.errors[sink_name] += 1
                    count = self.errors[sink_name]
                    if count <= 3 or count % 100 == 0:
                        logger.exception("Destino %s del pump %s fallo (%d errores)", sink_name, self.label, count)

    def stats(self) -> Dict[str, Any]:
        return {
            "alive": self.is_alive(),
            "moved": self.moved,
            "errors": {k: v for k, v in self.errors.items() if v},
            "seconds_since_last": None if self.last_item_wall is None else round(time.time() - self.last_item_wall, 2),
        }


# =============================================================================
# Zonas para el visualizador (mismos archivos que usan Behavior y Decision)
# =============================================================================

def load_visual_zones(camera_id: str) -> List[Dict[str, Any]]:
    zones_file = os.getenv("BEHAVIOR_ZONES_FILE")
    if not zones_file or not os.path.isfile(zones_file):
        return []
    with open(zones_file, "r", encoding="utf-8") as handle:
        raw_zones = json.load(handle).get(camera_id, [])

    zone_types: Dict[str, Any] = {}
    context_file = os.getenv("DECISION_CONTEXT_FILE")
    if context_file and os.path.isfile(context_file):
        with open(context_file, "r", encoding="utf-8") as handle:
            zone_types = json.load(handle).get(camera_id, {}).get("zones", {})

    return [
        {
            "zone_id": zone["zone_id"],
            "name": zone.get("name", zone["zone_id"]),
            "polygon": zone["polygon"],
            "kind": zone.get("kind", "area"),
            "zone_type": zone_types.get(zone["zone_id"], {}).get("zone_type"),
        }
        for zone in raw_zones
    ]


# =============================================================================
# Pipeline
# =============================================================================

class Pipeline:
    def __init__(self) -> None:
        self.camera_id = os.getenv("VIDEO_CAMERA_ID", "CAM-01")
        self._stop_event = threading.Event()
        self._pumps: List[Pump] = []
        self._start_wall = time.time()

        ensure_model_file("MODEL_PATH", "models/yolov8n.pt")
        ensure_model_file("POSE_MODEL_PATH", "models/yolov8n-pose.pt")

        logger.info("Inicializando servicios | camara=%s", self.camera_id)
        self.s1: CameraWorker = build_camera_worker_from_env()
        self.s2 = PreprocessingService()
        self.s3 = YoloDetectionService()
        self.s4 = TrackingService(config=TrackConfig())
        self.s5 = PoseService(config=PoseConfig.from_env())
        self.s6 = BehaviorService(config=BehaviorConfig.from_env())
        self.s7 = DecisionService(config=DecisionConfig.from_env())
        self.s8 = EvidenceService(config=EvidenceConfig.from_env())
        self.s9 = AlertService(AlertConfig(
            host=os.getenv("ALERT_HOST", "127.0.0.1"),
            port=int(os.getenv("ALERT_PORT", "8080")),
            evidence_root=os.getenv("EVIDENCE_STORAGE_ROOT", "evidence_store"),
            open_browser=_env_bool("ALERT_OPEN_BROWSER", True),
        ))
        viz_config = VisualizerConfig.from_env()
        if viz_config.alerts_url is None:
            viz_config = replace(viz_config, alerts_url=f"http://localhost:{os.getenv('ALERT_PORT', '8080')}/")
        self.s10 = VisualizerService(viz_config, health_provider=self.health)
        self.s10.set_zones(self.camera_id, load_visual_zones(self.camera_id))

    # -------------------------------------------------------------------
    # Conexiones
    # -------------------------------------------------------------------

    def _wire(self) -> None:
        cam = self.camera_id
        s1, s2, s3, s4, s5, s6, s7, s8, s9, s10 = (self.s1, self.s2, self.s3, self.s4, self.s5,
                                                  self.s6, self.s7, self.s8, self.s9, self.s10)
        stop = self._stop_event
        self._pumps = [
            Pump("video", lambda t: s1.get_next_frame(timeout=t), [
                ("preprocessing", lambda f: s2.submit_frame(cam, f)),
                ("pose_frame_cache", lambda f: s5.cache_frame(cam, int(f["frame_id"]), f["image"])),
                ("evidence_buffer", lambda f: s8.buffer.append_frame(cam, float(f["timestamp"]), f["image"])),
                ("visualizer", s10.on_frame),
            ], stop),
            Pump("preprocessing", lambda t: s2.get_next_processed(cam, timeout=t), [
                ("detection", s3.submit_frame),
            ], stop),
            Pump("detection", lambda t: s3.get_next_result(cam, timeout=t), [
                ("tracking", s4.submit_detection_result),
            ], stop),
            Pump("tracking", lambda t: s4.get_next_result(cam, timeout=t), [
                ("pose", s5.submit_tracking_result),
                ("behavior", s6.submit_tracking_result),
                ("visualizer", s10.on_tracking),
            ], stop),
            Pump("pose", lambda t: s5.get_next_result(cam, timeout=t), [
                ("behavior", s6.submit_pose_result),
                ("visualizer", s10.on_pose),
            ], stop),
            Pump("behavior", lambda t: s6.get_next_result(timeout=t), [
                ("decision", s7.submit),
                ("visualizer", s10.on_behavior),
            ], stop),
            Pump("decision", lambda t: s7.get_next_decision(timeout=t), [
                ("evidence", s8.submit_decision),
                ("visualizer", s10.on_decision),
            ], stop),
            Pump("evidence", lambda t: s8.get_next_result(timeout=t), [
                ("alerts", s9.receive_evidence),
                ("visualizer", s10.on_evidence),
            ], stop),
        ]

    # -------------------------------------------------------------------
    # Ciclo de vida
    # -------------------------------------------------------------------

    def start(self) -> None:
        # De atras hacia adelante: cada consumidor listo antes que su productor.
        self.s9.start()
        self.s8.start()
        self.s7.start()
        self.s6.start()
        self.s5.start()
        self.s3.start()
        target = int(os.getenv("PREPROCESS_TARGET_SIZE", "640"))
        self.s2.register_camera(CameraProcessingConfig(
            camera_id=self.camera_id, target_width=target, target_height=target,
            target_fps=float(os.getenv("VIDEO_TARGET_FPS", "10")),
        ))
        self.s10.start()
        self._wire()
        for pump in self._pumps:
            pump.start()
        self.s1.start()
        logger.info("Pipeline en marcha | visualizador=http://localhost:%s/  alertas=http://localhost:%s/",
                    self.s10.config.port, self.s9.config.port)

    def stop(self) -> None:
        logger.info("Deteniendo pipeline...")
        self.s1.stop()                      # 1) cortar la entrada de video
        self._stop_event.set()              # 2) detener los pumps
        for pump in self._pumps:
            pump.join(timeout=2.0)
        for name, stopper in (("preprocessing", self.s2.stop_all), ("detection", self.s3.stop),
                              ("tracking", self.s4.stop_all), ("pose", self.s5.stop), ("behavior", self.s6.stop),
                              ("decision", self.s7.stop), ("evidence", lambda: self.s8.stop(drain_timeout=8.0)),
                              ("alerts", self.s9.stop), ("visualizer", self.s10.stop)):
            try:
                stopper()
            except Exception:
                logger.exception("Error deteniendo %s", name)
        logger.info("Pipeline detenido")

    # -------------------------------------------------------------------
    # Salud agregada (expuesta por el visualizador en /health)
    # -------------------------------------------------------------------

    def health(self) -> Dict[str, Any]:
        services: Dict[str, Any] = {}
        for name, provider in (("video_ingestion", self.s1.health_check), ("preprocessing", self.s2.health_check),
                               ("detection", self.s3.health_check), ("tracking", self.s4.health_check),
                               ("pose", self.s5.health_check), ("behavior", self.s6.health_check),
                               ("decision", self.s7.health_check), ("evidence", self.s8.health_check),
                               ("alerts", self.s9.health_check)):
            try:
                services[name] = provider()
            except Exception as exc:
                services[name] = {"error": str(exc)}

        camera_state = services.get("video_ingestion", {}).get("connection_state")
        pumps = {pump.label: pump.stats() for pump in self._pumps}
        pumps_ok = bool(pumps) and all(p["alive"] for p in pumps.values())
        return {
            "status": "healthy" if camera_state == "CONNECTED" and pumps_ok else "degraded",
            "version": PIPELINE_VERSION,
            "service": "modelo-ia",
            "camera_id": self.camera_id,
            "camera_state": camera_state,
            "uptime_seconds": round(time.time() - self._start_wall, 1),
            "pumps": pumps,
            "services": services,
        }


# =============================================================================
# Entrada
# =============================================================================

def main() -> None:
    os.chdir(BASE_DIR)  # rutas relativas (models/, config/, evidence_store/) siempre desde modelo_IA/
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(BASE_DIR, ".env"), override=False)  # las variables del entorno/compose mandan
    except ImportError:
        pass

    stop_requested = threading.Event()

    def _request_stop(signum: int, _frame: Any) -> None:
        logger.info("Senal %s recibida, cerrando...", signum)
        stop_requested.set()

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    pipeline = Pipeline()
    pipeline.start()
    try:
        while not stop_requested.wait(0.5):
            pass
    finally:
        pipeline.stop()


if __name__ == "__main__":
    sys.exit(main())
