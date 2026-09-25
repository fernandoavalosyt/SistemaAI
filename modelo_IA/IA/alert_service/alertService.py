"""
alertService.py

Servicio de Alertas (Servicio 9 del pipeline) - MODO PRUEBAS LOCALES.

Recibe EvidenceResult desde "Event / Evidence" (Servicio 8), los mantiene en
memoria, levanta un servidor web ligero (http.server de la libreria estandar)
en 127.0.0.1 y abre automaticamente el navegador en la alerta recibida.

Entradas soportadas:
  - Llamada directa en proceso:   service.receive_evidence(evidence_result_dict)
  - HTTP desde otro proceso:      POST http://127.0.0.1:8080/api/evidence  (JSON)

Uso:
  python alertService.py                 # servidor esperando alertas
  python alertService.py --demo          # genera alertas simuladas
  python alertService.py --port 5000 --no-browser --evidence-root ./evidence_store

Este servicio NO envia SMS, WhatsApp ni correos, NO usa frameworks frontend y
NO usa bases de datos: las alertas viven en memoria mientras viva el proceso.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
import webbrowser
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Iterator, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse


# =============================================================================
# Logging
# =============================================================================

logger = logging.getLogger("alert_service")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


# =============================================================================
# Configuracion y catalogos de presentacion
# =============================================================================

@dataclass(frozen=True)
class AlertConfig:
    host: str = "127.0.0.1"               # solo loopback: nunca expuesto a la red
    port: int = 8080
    evidence_root: str = "evidence_store"  # mismo root que usa el Servicio 8
    open_browser: bool = True
    ui_active_window_seconds: float = 6.0  # si hay una pestaña abierta, no abrir otra
    max_alerts: int = 100
    max_post_bytes: int = 1024 * 1024
    max_checksum_bytes: int = 500 * 1024 * 1024
    max_stream_seconds: float = 600.0
    max_stream_frames: int = 3000


PRIORITY_STYLE: Dict[str, Dict[str, str]] = {
    "CRITICAL": {"label": "Crítica", "color": "#ef4444", "soft": "rgba(239,68,68,.14)", "glow": "rgba(239,68,68,.45)"},
    "HIGH": {"label": "Alta", "color": "#f97316", "soft": "rgba(249,115,22,.14)", "glow": "rgba(249,115,22,.40)"},
    "MEDIUM": {"label": "Media", "color": "#eab308", "soft": "rgba(234,179,8,.14)", "glow": "rgba(234,179,8,.35)"},
    "LOW": {"label": "Baja", "color": "#38bdf8", "soft": "rgba(56,189,248,.14)", "glow": "rgba(56,189,248,.30)"},
}

EVENT_LABELS: Dict[str, str] = {
    "POSSIBLE_THEFT": "Posible robo",
    "SLEEPING_GUARD": "Guardia con somnolencia",
    "SUSPICIOUS_BEHAVIOR": "Comportamiento sospechoso",
}

_BROWSER_SAFE_FOURCC = {"avc1", "h264", "x264", "vp80", "vp90", "av01"}


def event_label(event_type: str) -> str:
    return EVENT_LABELS.get(event_type, event_type.replace("_", " ").capitalize())


def fmt_time(ts: Optional[float]) -> str:
    if not isinstance(ts, (int, float)) or ts <= 0:
        return "—"
    return datetime.fromtimestamp(ts).strftime("%d/%m/%Y %H:%M:%S")


def fmt_size(size: Any) -> str:
    if not isinstance(size, (int, float)) or size <= 0:
        return "—"
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def esc(value: Any) -> str:
    """Todo dato que viene del pipeline se escapa antes de ir al HTML."""
    return html.escape("—" if value is None or value == "" else str(value), quote=True)


# =============================================================================
# Modelo en memoria
# =============================================================================

@dataclass
class Alert:
    alert_id: str
    received_at: float
    evidence: Dict[str, Any]
    video_path: Optional[str] = None
    video_mode: str = "none"            # "video" | "stream" | "none"
    video_note: str = ""
    checksum_status: str = "n/a"        # verified | mismatch | pending | unverified | n/a

    @property
    def priority(self) -> str:
        value = str(self.evidence.get("priority", "LOW")).upper()
        return value if value in PRIORITY_STYLE else "LOW"

    def summary(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "event_id": self.evidence.get("event_id"),
            "event_type": self.evidence.get("event_type"),
            "label": event_label(str(self.evidence.get("event_type", ""))),
            "priority": self.priority,
            "camera_id": self.evidence.get("camera_id"),
            "status": self.evidence.get("status"),
            "received_at": self.received_at,
            "url": f"/alerts/{self.alert_id}",
        }


class AlertStore:
    """Lista acotada en memoria, thread-safe e idempotente por event_id."""

    def __init__(self, max_alerts: int) -> None:
        self._alerts: "OrderedDict[str, Alert]" = OrderedDict()
        self._by_event: Dict[str, str] = {}
        self._max = max_alerts
        self._lock = threading.Lock()
        self.version = 0

    def upsert(self, alert: Alert) -> Tuple[Alert, bool]:
        with self._lock:
            event_id = str(alert.evidence.get("event_id"))
            existing_id = self._by_event.get(event_id)
            if existing_id and existing_id in self._alerts:
                current = self._alerts[existing_id]
                current.evidence = alert.evidence
                current.video_path, current.video_mode = alert.video_path, alert.video_mode
                current.video_note, current.checksum_status = alert.video_note, alert.checksum_status
                self.version += 1
                return current, False
            self._alerts[alert.alert_id] = alert
            self._by_event[event_id] = alert.alert_id
            while len(self._alerts) > self._max:
                _, dropped = self._alerts.popitem(last=False)
                self._by_event.pop(str(dropped.evidence.get("event_id")), None)
            self.version += 1
            return alert, True

    def get(self, alert_id: str) -> Optional[Alert]:
        with self._lock:
            return self._alerts.get(alert_id)

    def latest(self) -> Optional[Alert]:
        with self._lock:
            return next(reversed(self._alerts.values()), None)

    def recent(self, limit: int = 50) -> List[Alert]:
        with self._lock:
            return list(reversed(self._alerts.values()))[:limit]

    def touch(self) -> None:
        with self._lock:
            self.version += 1


# =============================================================================
# Acceso seguro a la evidencia de video
# =============================================================================

def resolve_video_path(reference: Optional[str], evidence_root: str) -> Optional[str]:
    """Traduce video_reference a una ruta local SOLO si queda dentro de
    evidence_root. La URL publica nunca contiene rutas: se sirve por
    alert_id, evitando lecturas arbitrarias de archivos."""
    if not reference:
        return None
    root = os.path.abspath(evidence_root)
    if reference.startswith("local://"):
        candidate = os.path.join(root, unquote(reference[len("local://"):]))
    elif reference.startswith("file://"):
        candidate = unquote(urlparse(reference).path)
        if re.match(r"^/[A-Za-z]:", candidate):
            candidate = candidate[1:]
    else:
        candidate = reference
    candidate = os.path.abspath(candidate)
    try:
        if os.path.commonpath([candidate, root]) != root:
            return None
    except ValueError:  # distinta unidad en Windows
        return None
    return candidate if os.path.isfile(candidate) else None


def detect_video_mode(path: Optional[str]) -> Tuple[str, str]:
    """Decide como mostrar el clip. OpenCV suele escribir MP4 con codec
    'mp4v' (MPEG-4 Part 2), que los navegadores NO reproducen; en ese caso
    se sirve un stream MJPEG decodificado en el servidor."""
    if path is None:
        return "none", ""
    lower = path.lower()
    if lower.endswith(".mjpeg"):
        return "stream", "Contenedor MJPEG reproducido como stream"
    if lower.endswith((".mp4", ".webm", ".mov", ".m4v")):
        try:
            import cv2
            capture = cv2.VideoCapture(path)
            fourcc_code = int(capture.get(cv2.CAP_PROP_FOURCC))
            capture.release()
            fourcc = "".join(chr((fourcc_code >> 8 * i) & 0xFF) for i in range(4)).strip().lower()
        except Exception:
            return "video", "Reproducción nativa del navegador"
        if fourcc in _BROWSER_SAFE_FOURCC or lower.endswith(".webm"):
            return "video", f"Reproducción nativa ({fourcc})"
        return "stream", f"Códec '{fourcc or 'desconocido'}' no compatible con navegador: stream MJPEG"
    return "none", "Formato de evidencia no soportado para vista previa"


def iter_clip_jpegs(path: str, max_frames: int) -> Tuple[List[bytes], float]:
    """Frames JPEG del clip + fps. Soporta MP4 (via OpenCV) y el contenedor
    'vigilia-mjpeg-v1' del Servicio 8 (sin dependencias)."""
    if path.lower().endswith(".mjpeg"):
        frames: List[bytes] = []
        with open(path, "rb") as handle:
            header_len = int.from_bytes(handle.read(4), "big")
            header = json.loads(handle.read(header_len).decode("utf-8"))
            while len(frames) < max_frames:
                if len(handle.read(32)) < 32:
                    break
                size = int.from_bytes(handle.read(8), "big")
                data = handle.read(size)
                if len(data) < size:
                    break
                frames.append(data)
        return frames, float(header.get("fps") or 10.0)

    import cv2
    capture = cv2.VideoCapture(path)
    fps = capture.get(cv2.CAP_PROP_FPS) or 10.0
    frames = []
    try:
        while len(frames) < max_frames:
            ok, image = capture.read()
            if not ok:
                break
            encoded_ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if encoded_ok:
                frames.append(encoded.tobytes())
    finally:
        capture.release()
    return frames, float(max(1.0, min(fps, 60.0)))


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# =============================================================================
# Plantillas HTML/CSS/JS (embebidas; sin frameworks ni recursos externos)
# =============================================================================

BASE_CSS = """
:root{
  --bg:#0b0f17; --panel:#111827; --panel-2:#0f1623; --line:#1f2a3a; --text:#e5e9f0;
  --muted:#8b96a8; --faint:#5b667a; --ok:#22c55e; --radius:14px;
  --p:#38bdf8; --p-soft:rgba(56,189,248,.14); --p-glow:rgba(56,189,248,.3);
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%}
body{
  font-family:"Segoe UI",Inter,system-ui,-apple-system,Roboto,Helvetica,Arial,sans-serif;
  background:radial-gradient(1200px 600px at 20% -10%, var(--p-soft), transparent 60%), var(--bg);
  color:var(--text); overflow:hidden; display:grid; grid-template-rows:56px auto minmax(0,1fr);
  -webkit-font-smoothing:antialiased;
}
a{color:inherit;text-decoration:none}
.mono{font-family:"Cascadia Code","JetBrains Mono",Consolas,ui-monospace,monospace}

/* Barra superior */
.topbar{display:flex;align-items:center;justify-content:space-between;padding:0 22px;border-bottom:1px solid var(--line);background:rgba(11,15,23,.85);backdrop-filter:blur(8px)}
.brand{display:flex;align-items:center;gap:10px;font-weight:650;letter-spacing:.3px}
.brand-mark{width:26px;height:26px;border-radius:8px;display:grid;place-items:center;background:linear-gradient(135deg,#1e293b,#0f172a);border:1px solid var(--line)}
.brand small{color:var(--muted);font-weight:500;margin-left:6px}
.top-right{display:flex;align-items:center;gap:18px;color:var(--muted);font-size:13px}
.live{display:flex;align-items:center;gap:8px}
.live i{width:8px;height:8px;border-radius:50%;background:var(--ok);box-shadow:0 0 0 0 rgba(34,197,94,.6);animation:ping 2s infinite}
@keyframes ping{0%{box-shadow:0 0 0 0 rgba(34,197,94,.55)}70%{box-shadow:0 0 0 9px rgba(34,197,94,0)}100%{box-shadow:0 0 0 0 rgba(34,197,94,0)}}

/* Banda de alerta */
.hero{margin:16px 22px 0;padding:16px 20px;border-radius:var(--radius);border:1px solid var(--line);
  background:linear-gradient(90deg,var(--p-soft),rgba(17,24,39,.6) 55%);display:flex;align-items:center;gap:18px;position:relative;overflow:hidden}
.hero:before{content:"";position:absolute;left:0;top:0;bottom:0;width:4px;background:var(--p)}
.hero-icon{width:46px;height:46px;border-radius:12px;display:grid;place-items:center;background:var(--p-soft);color:var(--p);flex:none;border:1px solid var(--p-glow)}
.pulse .hero-icon{animation:alarm 1.6s ease-in-out infinite}
@keyframes alarm{0%,100%{box-shadow:0 0 0 0 var(--p-glow)}50%{box-shadow:0 0 0 10px transparent}}
.hero h1{font-size:21px;font-weight:680;letter-spacing:.2px}
.hero p{color:var(--muted);font-size:13.5px;margin-top:3px}
.hero-badges{margin-left:auto;display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end}
.badge{display:inline-flex;align-items:center;gap:6px;padding:5px 11px;border-radius:999px;font-size:12px;font-weight:600;border:1px solid var(--line);background:var(--panel-2);color:var(--muted);white-space:nowrap}
.badge.prio{background:var(--p);color:#0b0f17;border-color:transparent}
.badge.ok{color:var(--ok);border-color:rgba(34,197,94,.35);background:rgba(34,197,94,.08)}
.badge.fail{color:#f87171;border-color:rgba(248,113,113,.35);background:rgba(248,113,113,.08)}

/* Grid principal */
.main{display:grid;grid-template-columns:minmax(0,1.75fr) minmax(300px,1fr) 270px;gap:16px;padding:16px 22px 18px;min-height:0}
.card{background:linear-gradient(180deg,var(--panel),var(--panel-2));border:1px solid var(--line);border-radius:var(--radius);min-height:0;display:flex;flex-direction:column}
.card-h{display:flex;align-items:center;justify-content:space-between;padding:12px 16px;border-bottom:1px solid var(--line);font-size:12px;letter-spacing:.8px;text-transform:uppercase;color:var(--muted);font-weight:600}
.card-b{padding:14px 16px;min-height:0}

/* Reproductor */
.player{position:relative;flex:1;min-height:0;margin:12px;border-radius:10px;overflow:hidden;background:#05070b;border:1px solid var(--line);display:grid;place-items:center}
.player video,.player img.stream{width:100%;height:100%;object-fit:contain;background:#05070b;display:block}
.overlay{position:absolute;left:12px;right:12px;top:10px;display:flex;justify-content:space-between;pointer-events:none}
.chip{background:rgba(5,7,11,.72);border:1px solid rgba(255,255,255,.08);padding:4px 9px;border-radius:7px;font-size:11.5px;display:flex;align-items:center;gap:6px;backdrop-filter:blur(4px)}
.chip .rec{width:7px;height:7px;border-radius:50%;background:#ef4444;animation:blink 1.2s steps(2) infinite}
@keyframes blink{50%{opacity:.2}}
.player-foot{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;padding:0 12px 12px}
.stat{background:var(--panel-2);border:1px solid var(--line);border-radius:10px;padding:9px 11px}
.stat span{display:block;color:var(--faint);font-size:10.5px;text-transform:uppercase;letter-spacing:.7px}
.stat b{display:block;margin-top:3px;font-size:13.5px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.placeholder{text-align:center;color:var(--muted);padding:26px;position:relative;width:100%;height:100%;display:grid;place-content:center;gap:10px;
  background:repeating-linear-gradient(0deg,rgba(255,255,255,.018) 0 2px,transparent 2px 4px),radial-gradient(500px 260px at 50% 45%,var(--p-soft),transparent 70%)}
.placeholder svg{margin:0 auto;color:var(--p);opacity:.9}
.placeholder h3{color:var(--text);font-size:16px;font-weight:620}
.placeholder p{font-size:13px;max-width:420px;margin:0 auto}
.note{font-size:11.5px;color:var(--faint);padding:0 14px 10px}

/* Detalles */
.kv{display:grid;grid-template-columns:auto 1fr;gap:9px 14px;font-size:13px;align-items:center}
.kv dt{color:var(--faint);font-size:11.5px;text-transform:uppercase;letter-spacing:.6px}
.kv dd{color:var(--text);min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;display:flex;align-items:center;gap:6px}
.copy{border:1px solid var(--line);background:var(--panel-2);color:var(--muted);border-radius:6px;font-size:10.5px;padding:2px 6px;cursor:pointer}
.copy:hover{color:var(--text)}
.section{margin-top:14px;padding-top:12px;border-top:1px dashed var(--line)}
.section h4{font-size:11px;letter-spacing:.8px;text-transform:uppercase;color:var(--muted);margin-bottom:8px;font-weight:600}
.rule{background:var(--panel-2);border:1px solid var(--line);border-left:3px solid var(--p);border-radius:8px;padding:10px 12px;font-size:12.8px;line-height:1.45;color:#cbd3df}
.rule b{color:var(--text);display:block;margin-bottom:3px;font-size:12px}
.bar{height:8px;border-radius:99px;background:#1b2433;overflow:hidden;margin-top:6px}
.bar i{display:block;height:100%;background:linear-gradient(90deg,var(--p),#e5e9f0);border-radius:99px}
.integrity{display:flex;align-items:center;gap:8px;font-size:12.5px}
.integrity .dot{width:9px;height:9px;border-radius:50%}
.details{overflow:auto}

/* Lista de alertas recientes */
.list{overflow:auto;padding:8px}
.item{display:grid;grid-template-columns:10px 1fr;gap:10px;padding:10px;border-radius:10px;border:1px solid transparent;margin-bottom:6px}
.item:hover{background:var(--panel-2);border-color:var(--line)}
.item.active{background:var(--panel-2);border-color:var(--line);box-shadow:inset 3px 0 0 var(--p)}
.item .d{width:10px;height:10px;border-radius:50%;margin-top:4px}
.item strong{font-size:13px;display:block;font-weight:600}
.item small{color:var(--muted);font-size:11.5px}
.empty-list{color:var(--faint);font-size:12.5px;text-align:center;padding:18px}

/* Toast */
.toast{position:fixed;right:22px;bottom:22px;background:var(--panel);border:1px solid var(--line);border-left:4px solid var(--p);border-radius:12px;padding:12px 14px;display:none;gap:12px;align-items:center;box-shadow:0 18px 40px rgba(0,0,0,.45);z-index:10;max-width:360px}
.toast.show{display:flex;animation:slide .25s ease-out}
@keyframes slide{from{transform:translateY(12px);opacity:0}to{transform:none;opacity:1}}
.toast a{background:var(--text);color:#0b0f17;padding:6px 10px;border-radius:8px;font-size:12px;font-weight:650}

/* Estado vacio */
.waiting{display:grid;place-items:center;text-align:center;padding:40px}
.radar{width:140px;height:140px;border-radius:50%;border:1px solid var(--line);margin:0 auto 22px;position:relative;
  background:radial-gradient(circle,rgba(56,189,248,.12) 0 30%,transparent 31%),repeating-radial-gradient(circle,transparent 0 22px,rgba(255,255,255,.04) 22px 23px)}
.radar:after{content:"";position:absolute;inset:0;border-radius:50%;background:conic-gradient(from 0deg,rgba(56,189,248,.35),transparent 25%);animation:spin 2.6s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.waiting h2{font-size:20px;font-weight:650}
.waiting p{color:var(--muted);margin-top:8px;font-size:14px}
.waiting code{display:inline-block;margin-top:14px;background:var(--panel);border:1px solid var(--line);padding:8px 12px;border-radius:8px;color:#cbd3df;font-size:12.5px}

@media (max-width:1180px){.main{grid-template-columns:minmax(0,1.5fr) minmax(280px,1fr)}.recent{display:none}}
@media (max-width:820px){body{overflow:auto;grid-template-rows:56px auto auto}.main{grid-template-columns:1fr}.player{min-height:240px}.player-foot{grid-template-columns:repeat(2,1fr)}.hero{flex-wrap:wrap}.hero-badges{margin-left:0}}
"""

BASE_JS = """
(function(){
  var clock=document.getElementById('clock');
  function tick(){var d=new Date();clock.textContent=d.toLocaleDateString()+'  '+d.toLocaleTimeString();}
  tick();setInterval(tick,1000);

  document.querySelectorAll('[data-copy]').forEach(function(btn){
    btn.addEventListener('click',function(){
      navigator.clipboard&&navigator.clipboard.writeText(btn.getAttribute('data-copy'));
      var t=btn.textContent;btn.textContent='Copiado';setTimeout(function(){btn.textContent=t;},1200);
    });
  });

  document.querySelectorAll('[data-ago]').forEach(function(el){
    function upd(){var s=Math.max(0,Math.round(Date.now()/1000-parseFloat(el.getAttribute('data-ago'))));
      el.textContent=s<60?('hace '+s+' s'):s<3600?('hace '+Math.floor(s/60)+' min'):('hace '+Math.floor(s/3600)+' h');}
    upd();setInterval(upd,5000);
  });

  var video=document.getElementById('evidence-video');
  if(video){
    var fallback=function(){var src=video.getAttribute('data-stream');if(!src)return;
      var img=document.createElement('img');img.className='stream';img.src=src;img.alt='Evidencia';
      video.replaceWith(img);var n=document.getElementById('player-note');if(n)n.textContent='El navegador no pudo decodificar el video: se muestra como stream MJPEG';};
    video.addEventListener('error',fallback,true);
    setTimeout(function(){if(video.isConnected&&video.readyState<1)fallback();},3000);
  }

  var state=window.__ALERT_STATE__||{version:0,current:null};
  var toast=document.getElementById('toast');
  function poll(){
    fetch('/api/alerts?since='+state.version,{cache:'no-store'}).then(function(r){return r.json();}).then(function(data){
      if(data.version!==state.version){
        var newest=data.alerts[0];
        state.version=data.version;
        if(newest&&newest.alert_id!==state.current&&(!state.known||state.known.indexOf(newest.alert_id)<0)){
          if(!state.current){location.href=newest.url;return;}
          document.getElementById('toast-title').textContent=newest.label+' · '+newest.camera_id;
          document.getElementById('toast-link').href=newest.url;
          toast.className='toast show';
          if(newest.priority==='CRITICAL'){setTimeout(function(){location.href=newest.url;},1800);}
        }
        state.known=data.alerts.map(function(a){return a.alert_id;});
      }
    }).catch(function(){}).finally(function(){setTimeout(poll,2000);});
  }
  state.known=state.known||[];
  poll();
})();
"""

ICON_ALERT = ('<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
              'stroke-linecap="round" stroke-linejoin="round"><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 '
              '1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>')
ICON_CAM_OFF = ('<svg width="56" height="56" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" '
                'stroke-linecap="round" stroke-linejoin="round"><path d="M16 16v1a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V7a2 2 0 0 1 '
                '2-2h2m5.66 0H14a2 2 0 0 1 2 2v3.34l1 1L23 7v10"/><line x1="1" y1="1" x2="23" y2="23"/></svg>')
ICON_BRAND = ('<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2.2" '
              'stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/>'
              '<circle cx="12" cy="12" r="3"/></svg>')

PAGE_TEMPLATE = """<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{{TITLE}}</title><style>{{CSS}}</style><style>:root{{{PRIORITY_VARS}}}</style></head>
<body>
<header class="topbar">
  <div class="brand"><div class="brand-mark">{{ICON_BRAND}}</div>VIGILIA<small>Centro de alertas · entorno local</small></div>
  <div class="top-right"><span class="live"><i></i>En vivo</span><span id="clock" class="mono"></span></div>
</header>
{{BODY}}
<div class="toast" id="toast"><div><div style="font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.7px">Nueva alerta</div>
<div id="toast-title" style="font-weight:600;font-size:13.5px;margin-top:2px"></div></div><a id="toast-link" href="#">Ver</a></div>
<script>window.__ALERT_STATE__={{STATE}};</script><script>{{JS}}</script>
</body></html>"""


def _json_for_script(value: Any) -> str:
    return json.dumps(value).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _priority_vars(priority: str) -> str:
    style = PRIORITY_STYLE.get(priority, PRIORITY_STYLE["LOW"])
    return f"--p:{style['color']};--p-soft:{style['soft']};--p-glow:{style['glow']};"


def render_page(title: str, body: str, priority: str, state: Dict[str, Any]) -> str:
    # BODY (que contiene texto del pipeline) se inserta al final para que
    # ningun dato externo pueda coincidir con otro marcador de la plantilla.
    return (PAGE_TEMPLATE
            .replace("{{CSS}}", BASE_CSS)
            .replace("{{{PRIORITY_VARS}}}", "{" + _priority_vars(priority) + "}")
            .replace("{{ICON_BRAND}}", ICON_BRAND)
            .replace("{{STATE}}", _json_for_script(state))
            .replace("{{JS}}", BASE_JS)
            .replace("{{TITLE}}", esc(title))
            .replace("{{BODY}}", body))


def render_recent(alerts: List[Alert], active_id: Optional[str]) -> str:
    if not alerts:
        return '<div class="empty-list">Sin alertas todavía</div>'
    rows = []
    for alert in alerts:
        style = PRIORITY_STYLE[alert.priority]
        ev = alert.evidence
        rows.append(
            f'<a class="item{" active" if alert.alert_id == active_id else ""}" href="/alerts/{esc(alert.alert_id)}">'
            f'<span class="d" style="background:{style["color"]}"></span><div>'
            f'<strong>{esc(event_label(str(ev.get("event_type", ""))))}</strong>'
            f'<small>{esc(ev.get("camera_id"))} · {esc(style["label"])} · <span data-ago="{alert.received_at}"></span></small>'
            f'</div></a>'
        )
    return "".join(rows)


def render_player(alert: Alert) -> str:
    ev = alert.evidence
    overlay = (f'<div class="overlay"><span class="chip mono"><span class="rec"></span>{esc(ev.get("camera_id"))}</span>'
               f'<span class="chip mono">{esc(fmt_time(ev.get("start_time")))}</span></div>')
    if alert.video_mode == "video":
        return (f'<video id="evidence-video" src="/media/{esc(alert.alert_id)}" controls autoplay muted loop playsinline '
                f'data-stream="/media/{esc(alert.alert_id)}/stream"></video>{overlay}')
    if alert.video_mode == "stream":
        return f'<img class="stream" src="/media/{esc(alert.alert_id)}/stream" alt="Evidencia en video">{overlay}'
    reason = (ev.get("metadata") or {}).get("failure_reason") or alert.video_note or "El clip no fue adjuntado a este evento."
    status = str(ev.get("status", ""))
    title = "Evidencia de video no disponible" if status == "FAILED" else "Evidencia pendiente de adjuntar"
    return (f'<div class="placeholder">{ICON_CAM_OFF}<h3>{esc(title)}</h3>'
            f'<p>{esc(reason)}</p><p class="mono" style="font-size:11.5px;color:var(--faint)">'
            f'Referencia: {esc(ev.get("video_reference"))}</p></div>{overlay}')


def render_alert_page(alert: Alert, recent: List[Alert], version: int) -> str:
    ev = alert.evidence
    meta = ev.get("metadata") or {}
    style = PRIORITY_STYLE[alert.priority]
    rules = meta.get("rules_activated") or []
    rule = rules[0] if rules and isinstance(rules[0], dict) else {}
    coverage = (meta.get("coverage") or {}).get("coverage_ratio")
    clip = meta.get("clip") or {}
    checksum = (meta.get("checksum") or {}).get("value")
    status = str(ev.get("status", "—"))
    status_badge = "ok" if status == "READY" else "fail"

    integrity_color, integrity_text = {
        "verified": ("#22c55e", "SHA-256 verificado contra el archivo"),
        "mismatch": ("#ef4444", "El checksum NO coincide con el archivo"),
        "pending": ("#eab308", "Verificando integridad…"),
        "unverified": ("#8b96a8", "Archivo no disponible para verificar"),
    }.get(alert.checksum_status, ("#5b667a", "Sin checksum declarado"))

    coverage_html = ""
    if isinstance(coverage, (int, float)):
        pct = max(0.0, min(100.0, coverage * 100))
        coverage_html = (f'<div class="section"><h4>Cobertura del clip</h4><div style="display:flex;justify-content:space-between;'
                         f'font-size:12.5px"><span>Video disponible en la ventana solicitada</span><b>{pct:.0f}%</b></div>'
                         f'<div class="bar"><i style="width:{pct:.1f}%"></i></div></div>')

    duration = ev.get("duration")
    body = f"""
<section class="hero{' pulse' if alert.priority in ('CRITICAL', 'HIGH') else ''}">
  <div class="hero-icon">{ICON_ALERT}</div>
  <div><h1>{esc(event_label(str(ev.get('event_type', ''))))}</h1>
  <p>Cámara <b class="mono">{esc(ev.get('camera_id'))}</b> · Track <span class="mono">{esc(ev.get('track_id'))}</span> · recibida <span data-ago="{alert.received_at}"></span></p></div>
  <div class="hero-badges">
    <span class="badge prio">Prioridad {esc(style['label'])}</span>
    <span class="badge {status_badge}">Evidencia {esc(status)}</span>
    <span class="badge mono">{esc(ev.get('event_type'))}</span>
  </div>
</section>
<main class="main">
  <section class="card">
    <div class="card-h"><span>Evidencia de video</span><span class="mono">{esc(ev.get('format'))}</span></div>
    <div class="player">{render_player(alert)}</div>
    <div class="note" id="player-note">{esc(alert.video_note) if alert.video_note else ''}</div>
    <div class="player-foot">
      <div class="stat"><span>Inicio</span><b class="mono">{esc(fmt_time(ev.get('start_time')))}</b></div>
      <div class="stat"><span>Fin</span><b class="mono">{esc(fmt_time(ev.get('end_time')))}</b></div>
      <div class="stat"><span>Duración</span><b>{esc(f"{duration:.1f} s" if isinstance(duration, (int, float)) else None)}</b></div>
      <div class="stat"><span>Tamaño · fps</span><b>{esc(fmt_size(ev.get('size')))} · {esc(clip.get('fps'))}</b></div>
    </div>
  </section>
  <section class="card">
    <div class="card-h"><span>Detalles del evento</span><span class="badge {status_badge}">{esc(status)}</span></div>
    <div class="card-b details">
      <dl class="kv">
        <dt>Evento</dt><dd class="mono">{esc(str(ev.get('event_id'))[:18])}… <button class="copy" data-copy="{esc(ev.get('event_id'))}">Copiar</button></dd>
        <dt>Evidencia</dt><dd class="mono">{esc(str(ev.get('evidence_id'))[:18])}… <button class="copy" data-copy="{esc(ev.get('evidence_id'))}">Copiar</button></dd>
        <dt>Cámara</dt><dd class="mono">{esc(ev.get('camera_id'))}</dd>
        <dt>Sesión</dt><dd class="mono">{esc(ev.get('session_id'))}</dd>
        <dt>Track</dt><dd class="mono">{esc(ev.get('track_id'))}</dd>
        <dt>Prioridad</dt><dd><span style="color:var(--p);font-weight:650">{esc(style['label'])}</span></dd>
        <dt>Recibida</dt><dd class="mono">{esc(fmt_time(alert.received_at))}</dd>
      </dl>
      <div class="section"><h4>Regla activada</h4>
        <div class="rule"><b class="mono">{esc(rule.get('rule_id'))}{(' · ' + esc(rule.get('rule_name'))) if rule.get('rule_name') else ''}</b>{esc(rule.get('summary') or 'Sin explicación adjunta')}</div>
      </div>
      {coverage_html}
      <div class="section"><h4>Integridad</h4>
        <div class="integrity"><span class="dot" style="background:{integrity_color}"></span>{esc(integrity_text)}</div>
        <div class="mono" style="font-size:11px;color:var(--faint);margin-top:6px;word-break:break-all">{esc(checksum)}</div>
      </div>
    </div>
  </section>
  <aside class="card recent">
    <div class="card-h"><span>Alertas recientes</span><span>{len(recent)}</span></div>
    <div class="list">{render_recent(recent, alert.alert_id)}</div>
  </aside>
</main>"""
    state = {"version": version, "current": alert.alert_id, "known": [a.alert_id for a in recent]}
    return render_page(f"{event_label(str(ev.get('event_type', '')))} · {ev.get('camera_id')}", body, alert.priority, state)


def render_waiting_page(port: int, version: int) -> str:
    body = f"""
<section></section>
<main class="waiting"><div>
  <div class="radar"></div>
  <h2>Monitoreando el pipeline</h2>
  <p>No hay alertas todavía. Esta pantalla se actualizará sola cuando llegue la primera evidencia.</p>
  <code class="mono">POST http://127.0.0.1:{port}/api/evidence</code>
</div></main>"""
    return render_page("VIGILIA · Esperando alertas", body, "LOW", {"version": version, "current": None, "known": []})


# =============================================================================
# Servidor HTTP
# =============================================================================

class _Handler(BaseHTTPRequestHandler):
    server_version = "VigiliaAlerts/1.0"
    service: "AlertService"

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.debug("%s - %s", self.address_string(), fmt % args)

    def _send(self, status: int, body: bytes, content_type: str, extra: Optional[Dict[str, str]] = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; "
                         "img-src 'self' data:; media-src 'self'; connect-src 'self'; frame-ancestors 'none'")
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status: int, payload: Any) -> None:
        self._send(status, json.dumps(payload, default=str).encode("utf-8"), "application/json; charset=utf-8")

    def _html(self, status: int, page: str) -> None:
        self._send(status, page.encode("utf-8"), "text/html; charset=utf-8")

    # --- GET -----------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        svc = self.service
        try:
            if path == "/":
                latest = svc.store.latest()
                if latest is None:
                    self._html(200, render_waiting_page(svc.config.port, svc.store.version))
                else:
                    self._send(302, b"", "text/plain", {"Location": f"/alerts/{latest.alert_id}"})
                return
            match = re.fullmatch(r"/alerts/([0-9a-f]{32})", path)
            if match:
                alert = svc.store.get(match.group(1))
                if alert is None:
                    self._html(404, render_waiting_page(svc.config.port, svc.store.version))
                    return
                self._html(200, render_alert_page(alert, svc.store.recent(), svc.store.version))
                return
            if path == "/api/alerts":
                svc.mark_ui_active()
                self._json(200, {"version": svc.store.version, "alerts": [a.summary() for a in svc.store.recent()]})
                return
            match = re.fullmatch(r"/api/alerts/([0-9a-f]{32})", path)
            if match:
                alert = svc.store.get(match.group(1))
                if alert is None:
                    self._json(404, {"error": "not_found"})
                else:
                    self._json(200, {**alert.summary(), "evidence": alert.evidence, "video_mode": alert.video_mode,
                                     "checksum_status": alert.checksum_status})
                return
            match = re.fullmatch(r"/media/([0-9a-f]{32})(/stream)?", path)
            if match:
                alert = svc.store.get(match.group(1))
                if alert is None or alert.video_path is None:
                    self._json(404, {"error": "video_not_available"})
                elif match.group(2):
                    self._stream(alert.video_path)
                else:
                    self._serve_file(alert.video_path)
                return
            if path == "/health":
                self._json(200, svc.health_check())
                return
            if path == "/favicon.ico":
                self._send(204, b"", "image/x-icon")
                return
            self._json(404, {"error": "not_found"})
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception:
            logger.exception("Error atendiendo %s", path)
            try:
                self._json(500, {"error": "internal_error"})
            except Exception:
                pass

    do_HEAD = do_GET

    # --- POST ----------------------------------------------------------------

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/evidence":
            self._json(404, {"error": "not_found"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > self.service.config.max_post_bytes:
            self._json(413 if length > 0 else 400, {"error": "invalid_body_size"})
            return
        if "application/json" not in (self.headers.get("Content-Type") or ""):
            self._json(415, {"error": "content_type_must_be_json"})
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            alert_id = self.service.receive_evidence(payload)
        except (ValueError, UnicodeDecodeError) as exc:
            self._json(400, {"error": str(exc)})
            return
        self._json(201, {"alert_id": alert_id, "url": self.service.alert_url(alert_id)})

    # --- Media ---------------------------------------------------------------

    def _serve_file(self, file_path: str) -> None:
        """Entrega el archivo con soporte de Range (necesario para <video>)."""
        size = os.path.getsize(file_path)
        content_type = {".mp4": "video/mp4", ".m4v": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime"}.get(
            os.path.splitext(file_path)[1].lower(), "application/octet-stream")
        start, end, status = 0, size - 1, 200
        range_header = self.headers.get("Range")
        if range_header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
            if not match or (not match.group(1) and not match.group(2)):
                self._send(416, b"", "text/plain", {"Content-Range": f"bytes */{size}"})
                return
            if match.group(1):
                start = int(match.group(1))
                end = int(match.group(2)) if match.group(2) else size - 1
            else:
                start = max(0, size - int(match.group(2)))
            end = min(end, size - 1)
            if start > end:
                self._send(416, b"", "text/plain", {"Content-Range": f"bytes */{size}"})
                return
            status = 206
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("X-Content-Type-Options", "nosniff")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if self.command == "HEAD":
            return
        with open(file_path, "rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining > 0:
                chunk = handle.read(min(256 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def _stream(self, file_path: str) -> None:
        """Reproduce el clip como multipart/x-mixed-replace (MJPEG) en bucle.
        Funciona en cualquier navegador sin importar el codec original."""
        cfg = self.service.config
        frames, fps = iter_clip_jpegs(file_path, cfg.max_stream_frames)
        if not frames:
            self._json(404, {"error": "no_frames"})
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if self.command == "HEAD":
            return
        delay = 1.0 / fps
        deadline = time.time() + cfg.max_stream_seconds
        while time.time() < deadline and not self.service.stopping:
            for data in frames:
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                 + str(len(data)).encode("ascii") + b"\r\n\r\n" + data + b"\r\n")
                self.wfile.flush()
                time.sleep(delay)
            time.sleep(0.8)  # pausa breve al final del clip antes de repetir


# =============================================================================
# Servicio principal
# =============================================================================

REQUIRED_FIELDS = ("event_id", "camera_id", "event_type", "priority", "status")


class AlertService:
    def __init__(self, config: Optional[AlertConfig] = None) -> None:
        self.config = config or AlertConfig()
        self.store = AlertStore(self.config.max_alerts)
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._last_ui_poll = 0.0
        self._browser_lock = threading.Lock()
        self._received = 0
        self.stopping = False

    # --- ciclo de vida -------------------------------------------------------

    def start(self) -> None:
        handler = type("BoundHandler", (_Handler,), {"service": self})
        self._server = ThreadingHTTPServer((self.config.host, self.config.port), handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, name="AlertHTTP", daemon=True)
        self._thread.start()
        logger.info("Alert Service escuchando en http://%s:%d", self.config.host, self.config.port)

    def stop(self) -> None:
        self.stopping = True
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        logger.info("Alert Service detenido")

    def alert_url(self, alert_id: str) -> str:
        return f"http://{self.config.host}:{self.config.port}/alerts/{alert_id}"

    def mark_ui_active(self) -> None:
        self._last_ui_poll = time.time()

    # --- entrada -------------------------------------------------------------

    def receive_evidence(self, evidence: Dict[str, Any]) -> str:
        """Registra un EvidenceResult y abre el navegador. Idempotente por
        event_id: un reintento actualiza la misma alerta."""
        if not isinstance(evidence, dict):
            raise ValueError("EvidenceResult debe ser un objeto JSON")
        missing = [f for f in REQUIRED_FIELDS if not evidence.get(f)]
        if missing:
            raise ValueError(f"EvidenceResult incompleto, faltan: {', '.join(missing)}")

        video_path = resolve_video_path(evidence.get("video_reference"), self.config.evidence_root)
        video_mode, video_note = detect_video_mode(video_path)
        if evidence.get("video_reference") and video_path is None:
            video_note = "La referencia de video no existe dentro de la carpeta de evidencias configurada"

        checksum = ((evidence.get("metadata") or {}).get("checksum") or {}).get("value")
        checksum_status = "n/a" if not checksum else ("pending" if video_path else "unverified")

        alert, created = self.store.upsert(Alert(
            alert_id=uuid.uuid4().hex, received_at=time.time(), evidence=evidence,
            video_path=video_path, video_mode=video_mode, video_note=video_note, checksum_status=checksum_status,
        ))
        self._received += 1
        logger.info("Alerta %s | %s prioridad=%s camara=%s evidencia=%s video=%s",
                    "recibida" if created else "actualizada (reintento)", evidence["event_type"],
                    evidence["priority"], evidence["camera_id"], evidence["status"], video_mode)

        if checksum_status == "pending":
            threading.Thread(target=self._verify_checksum, args=(alert, video_path, checksum), daemon=True).start()
        if created:
            self._maybe_open_browser(alert)
        return alert.alert_id

    def _verify_checksum(self, alert: Alert, path: str, expected: str) -> None:
        try:
            if os.path.getsize(path) > self.config.max_checksum_bytes:
                alert.checksum_status = "unverified"
            else:
                alert.checksum_status = "verified" if sha256_file(path) == expected else "mismatch"
        except OSError:
            alert.checksum_status = "unverified"
        if alert.checksum_status == "mismatch":
            logger.warning("Checksum NO coincide para la evidencia del evento %s", alert.evidence.get("event_id"))

    def _maybe_open_browser(self, alert: Alert) -> None:
        """Abre una pestaña solo si no hay ya una pagina de alertas abierta
        (la pagina abierta hace polling y muestra la nueva alerta)."""
        if not self.config.open_browser:
            return
        with self._browser_lock:
            if time.time() - self._last_ui_poll < self.config.ui_active_window_seconds:
                return
            self._last_ui_poll = time.time()  # evita rafagas de pestañas
        url = self.alert_url(alert.alert_id)
        threading.Thread(target=webbrowser.open_new_tab, args=(url,), daemon=True).start()
        logger.info("Abriendo navegador en %s", url)

    def health_check(self) -> Dict[str, Any]:
        return {
            "service_active": self._thread is not None and self._thread.is_alive(),
            "url": f"http://{self.config.host}:{self.config.port}/",
            "alerts_in_memory": len(self.store.recent(self.config.max_alerts)),
            "alerts_received": self._received,
            "ui_open": time.time() - self._last_ui_poll < self.config.ui_active_window_seconds,
            "evidence_root": os.path.abspath(self.config.evidence_root),
        }


# =============================================================================
# Demo / ejecucion directa
# =============================================================================

def _make_demo_clip(root: str, camera_id: str, evidence_id: str, seconds: float = 6.0, fps: int = 10) -> Optional[Tuple[str, int, str]]:
    """Genera un MP4 de prueba (mismo layout que LocalFileStorage del
    Servicio 8). Retorna (video_reference, size, sha256) o None sin OpenCV."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    relative = f"{camera_id}/{datetime.now(timezone.utc).strftime('%Y-%m-%d')}/{evidence_id}.mp4"
    path = os.path.join(root, relative)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 360))
    for i in range(int(seconds * fps)):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        frame[:] = (28, 22, 18)
        cv2.rectangle(frame, (0, 250), (640, 360), (40, 34, 30), -1)
        x = 80 + i * 6
        cv2.rectangle(frame, (x, 120), (x + 70, 290), (70, 160, 240), 2)
        cv2.circle(frame, (x + 35, 140), 16, (200, 200, 200), 2)
        cv2.putText(frame, f"{camera_id}  DEMO  #{i:03d}", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1)
        writer.write(frame)
    writer.release()
    return f"local://{relative}", os.path.getsize(path), sha256_file(path)


def _demo_evidence(root: str, event_type: str, priority: str, camera_id: str, with_video: bool) -> Dict[str, Any]:
    now = time.time()
    event_id = str(uuid.uuid4())
    evidence_id = str(uuid.uuid5(uuid.NAMESPACE_URL, event_id))
    clip = _make_demo_clip(root, camera_id, evidence_id) if with_video else None
    rule_ids = {"POSSIBLE_THEFT": "POSSIBLE_THEFT_SEQUENCE", "SLEEPING_GUARD": "SLEEPING_GUARD_DESK",
                "SUSPICIOUS_BEHAVIOR": "LOITERING_SENSITIVE_ZONE"}
    summaries = {
        "POSSIBLE_THEFT": "product_interaction/FINISHED 6.0s en zona pasillo -> loitering/POSSIBLE en zona salida.",
        "SLEEPING_GUARD": "somnolence/SUSTAINED 16.0s en zona escritorio_guardia (turno nocturno: escalada a CRITICAL).",
        "SUSPICIOUS_BEHAVIOR": "loitering/SUSTAINED 64.0s en zona almacen.",
    }
    metadata: Dict[str, Any] = {
        "rules_activated": [{"rule_id": rule_ids.get(event_type), "rule_name": event_label(event_type),
                             "summary": summaries.get(event_type)}],
    }
    if clip:
        metadata.update({"checksum": {"algorithm": "sha256", "value": clip[2]},
                         "coverage": {"coverage_ratio": 0.97}, "clip": {"fps": 10.0, "format": "mp4"}})
    else:
        metadata.update({"failure_reason": "insufficient_video: la cámara no tenía video en el buffer",
                         "coverage": {"coverage_ratio": 0.0}})
    return {
        "evidence_id": evidence_id, "event_id": event_id, "camera_id": camera_id, "session_id": "s-demo",
        "track_id": f"{camera_id}-7", "event_type": event_type, "priority": priority,
        "video_reference": clip[0] if clip else None, "start_time": now - 15, "end_time": now + 10 if clip else now,
        "duration": 25.0 if clip else 0.0, "size": clip[1] if clip else 0, "format": "mp4" if clip else None,
        "status": "READY" if clip else "FAILED", "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="VIGILIA · Alert Service (pruebas locales)")
    parser.add_argument("--port", type=int, default=int(os.getenv("ALERT_PORT", "8080")))
    parser.add_argument("--evidence-root", default=os.getenv("EVIDENCE_STORAGE_ROOT", "evidence_store"))
    parser.add_argument("--no-browser", action="store_true", help="No abrir el navegador automaticamente")
    parser.add_argument("--demo", action="store_true", help="Enviar alertas simuladas")
    args = parser.parse_args()

    demo_root = tempfile.mkdtemp(prefix="vigilia_alert_demo_") if args.demo else None
    config = AlertConfig(port=args.port, evidence_root=demo_root or args.evidence_root, open_browser=not args.no_browser)
    service = AlertService(config)
    service.start()
    print(f"\n  VIGILIA Alert Service -> http://127.0.0.1:{args.port}/\n  POST de evidencias -> http://127.0.0.1:{args.port}/api/evidence\n  Ctrl+C para salir\n")

    if args.demo:
        service.receive_evidence(_demo_evidence(demo_root, "SUSPICIOUS_BEHAVIOR", "MEDIUM", "CAM-003", with_video=False))
        time.sleep(1.0)
        service.receive_evidence(_demo_evidence(demo_root, "POSSIBLE_THEFT", "HIGH", "CAM-001", with_video=True))
        time.sleep(1.0)
        service.receive_evidence(_demo_evidence(demo_root, "SLEEPING_GUARD", "CRITICAL", "CAM-SEC-01", with_video=True))

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        service.stop()
        if demo_root:
            shutil.rmtree(demo_root, ignore_errors=True)


if __name__ == "__main__":
    main()
