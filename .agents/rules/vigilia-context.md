# Contexto del Proyecto: Vigilia

## Propósito Principal
**Vigilia** es un sistema avanzado de videovigilancia e inteligencia artificial diseñado para **detectar somnolencia o sueño en personal de seguridad (guardias) en tiempo real**. Su objetivo es identificar periodos de inactividad, posturas de sueño o cabeceos, y emitir alertas inmediatas para prevenir brechas de seguridad derivadas de guardias dormidos en horario laboral.

## Arquitectura de Integración (Video Sources)
- El sistema no depende de cámaras web estáticas locales, sino que está diseñado para **conectarse a sistemas de CCTV existentes**.
- Se integra con DVRs (cámaras análogas) y NVRs/Cámaras IP a través del protocolo **RTSP (Real Time Streaming Protocol)** u otros protocolos de transmisión de video en red.

## Stack de Inteligencia Artificial (modelo_IA)
- **Detección de Objetos:** Utiliza **YOLO** (You Only Look Once) para detectar personas (guardias) en el encuadre.
- **Seguimiento (Tracking):** Implementa algoritmos de tracking (como ByteTrack o BoT-SORT integrados en YOLO) para asignar y mantener un ID único por cada individuo detectado, incluso si se cruzan o se ocultan temporalmente.
- **Detección de Somnolencia (Core Business Logic):**
  - **Falta de Movimiento (Parones):** Análisis de inactividad prolongada del bounding box.
  - **Análisis de Postura:** Detección de patrones posturales (ej. cabeza caída sobre el escritorio, cuerpo recostado).
  - *Sugerencia técnica futura:* Implementar detección de keypoints (YOLO-Pose) o estimación de la orientación de la cabeza (Head Pose Estimation) para mayor precisión.

## Acciones y Alertas
- Cuando el "Risk Score" (Nivel de Riesgo por somnolencia) supera un umbral crítico, el sistema debe disparar acciones automatizadas (alertas sonoras locales, notificaciones push/email a supervisores, registro en base de datos).

## Directrices para Agentes de IA
1. **Enfoque en Somnolencia:** Cualquier desarrollo en `modelo_IA/app/core/behavior.py` y `risk_engine.py` debe priorizar la detección de sueño, inactividad y posturas sedentarias, por encima de otras métricas de seguridad genéricas (como intrusión o merodeo rápido).
2. **Optimización de RTSP:** El procesamiento de video debe estar preparado para manejar latencia, pérdida de frames y re-conexiones típicas de streams RTSP.
3. **Escalabilidad:** El backend y el worker de IA deben procesar múltiples streams concurrentes (Múltiples cámaras conectadas a un DVR).
4. **Persistencia de Eventos:** Los frames o clips exactos donde se detecta al guardia durmiendo deben recortarse y guardarse en el Object Storage (MinIO) como evidencia.
