#  Vigilia — Sistema de Videovigilancia y Análisis de Comportamiento

Plataforma de **videovigilancia inteligente y análisis de video en tiempo real**, construida sobre una arquitectura de **microservicios desacoplada**.

El sistema integra:

* ⚛️ **Frontend:** React + Vite
* 🟢 **Backend:** Node.js + Express
* 🐍 **Motor de IA:** Python + YOLOv8
* 🐳 **Orquestación:** Docker Compose
* 📡 **Streaming:** HTTP MJPEG
* 🗄️ **Base de datos:** MySQL + Prisma

El sistema procesa flujos de video para:

*  Detectar personas y objetos.
*  Rastrear trayectorias mediante `Track ID`.
*  Analizar postura corporal.
*  Evaluar tiempo de permanencia en zonas.
*  Detectar comportamientos potencialmente sospechosos.
*  Registrar métricas y eventos.
*  Emitir video procesado mediante streaming MJPEG.

---

##  Arquitectura General

```text
┌─────────────────────────────────────────────────────────────────────┐
│                         DOCKER COMPOSE                              │
│                                                                     │
│  ┌───────────────────┐   ┌───────────────────┐   ┌───────────────┐ │
│  │  Frontend (React) │   │  Backend (Node)    │   │  Motor IA     │ │
│  │      Port: 3000   │   │      Port: 4000    │   │  Port: 5000   │ │
│  └─────────┬─────────┘   └─────────┬─────────┘   └───────┬───────┘ │
│            │                       │                     │         │
└────────────┼───────────────────────┼─────────────────────┼─────────┘
             │                       │                     │
             ▼                       ▼                     ▼
      ┌─────────────┐        ┌─────────────┐       ┌──────────────┐
      │  Navegador  │◄──────►│ API REST /  │       │ Stream MJPEG │
      │    Web      │        │     DB      │       │              │
      └─────────────┘        └─────────────┘       └──────────────┘

      http://localhost:3000
      http://localhost:4000/api
      http://localhost:5000/stream
```

### 🔄 Flujo general

```text
Cámara / Video
      │
      ▼
VideoService
      │
      ▼
PreprocessService
      │
      ▼
YOLOv8 ───────────────► Detección
      │
      ▼
TrackingService ──────► Track IDs
      │
      ▼
PoseService ──────────► Postura corporal
      │
      ▼
BehaviorService ──────► Zonas + Permanencia
      │
      ▼
DecisionEngine ───────► Reglas de negocio
      │
      ▼
AnomalyClassifier ────► Clasificación
      │
      ▼
MetricsService ───────► Eventos + Métricas
      │
      ▼
Visualizer
      │
      ▼
HTTP MJPEG Stream
```

---

# 🧩 Flujo del Motor de Inteligencia Artificial

El contenedor `modelo_ia` ejecuta un **Orquestador Central** mediante:

```text
main_orchestrator.py
```

Este componente conecta secuencialmente los diferentes servicios encargados de procesar cada frame en tiempo real.

##  Pipeline S1 — S10

| Etapa   | Servicio            | Función                                                                                     |
| ------- | ------------------- | ------------------------------------------------------------------------------------------- |
| **S1**  | `VideoService`      | Captura frames desde webcam, RTSP o archivos `.mp4`. Mantiene el contador global de frames. |
| **S2**  | `PreprocessService` | Redimensiona, normaliza y ajusta brillo/contraste.                                          |
| **S3**  | `YoloService`       | Ejecuta inferencia con modelos YOLOv8 para detectar personas y objetos.                     |
| **S4**  | `TrackingService`   | Asigna identificadores únicos (`Track ID`) y mantiene trayectorias.                         |
| **S5**  | `PoseService`       | Extrae puntos clave del cuerpo para estimación de postura.                                  |
| **S6**  | `BehaviorService`   | Analiza zonas de interés y tiempo de permanencia.                                           |
| **S7**  | `DecisionEngine`    | Aplica reglas de negocio y evalúa condiciones de riesgo o anomalías.                        |
| **S8**  | `AnomalyClassifier` | Clasifica eventos como `POSSIBLE_THEFT`, `LOITERING` o `NORMAL`.                            |
| **S9**  | `MetricsService`    | Registra eventos, FPS, latencia y logs de auditoría.                                        |
| **S10** | `Visualizer`        | Superpone detecciones, esqueletos, IDs y alertas, y expone el stream MJPEG.                 |

---

# 📁 Estructura del Proyecto

```text
.
├── docker-compose.yml
├── README.md
│
├── backend/
│   ├── Dockerfile
│   ├── package.json
│   └── src/
│
├── frontend/
│   ├── Dockerfile
│   ├── package.json
│   └── src/
│
└── modelo_IA/
    ├── Dockerfile
    ├── requirements.txt
    ├── main_orchestrator.py
    │
    ├── models/
    │   ├── yolov8n.pt
    │   └── pose.pt
    │
    ├── config/
    │   └── behavior_rules.json
    │
    └── servicios/
        ├── __init__.py
        ├── video_service.py
        ├── yolo_service.py
        ├── tracking_service.py
        ├── pose_service.py
        ├── behavior_service.py
        ├── decision_engine.py
        ├── anomaly_classifier.py
        └── metrics_service.py
```

---

# ⚙️ Requisitos Previos

Antes de ejecutar el proyecto se requiere:

* 🐳 Docker Engine `20.10+`
* 🐳 Docker Compose `2.0+`
* 🌐 Git
* 📷 Cámara web integrada/USB **o** un archivo `.mp4`

Puedes comprobar las instalaciones con:

```bash
docker --version
docker compose version
git --version
```

---

# 🚀 Instalación y Puesta en Marcha

## 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/byrack-ai.git
cd byrack-ai
```

> **Nota:** Reemplaza `tu-usuario/byrack-ai` por la URL real del repositorio.

---

## 2. Configurar Variables de Entorno

Crea un archivo `.env` en la raíz del proyecto.

Ejemplo:

```env
# ==============================
# Backend
# ==============================

PORT=4000

DATABASE_URL="mysql://root:rootpassword@db:3306/byrack_db"


# ==============================
# Motor de Inteligencia Artificial
# ==============================

VIDEO_DEVICE_INDEX=0

LOITERING_THRESHOLD_SECONDS=3

VISUALIZER_PORT=5000
```

###  Fuente de video

Para utilizar una webcam:

```env
VIDEO_DEVICE_INDEX=0
```

Para utilizar un archivo de video:

```env
VIDEO_DEVICE_INDEX="/app/videos/demo_test.mp4"
```

---

## 3. Ejecutar el sistema

Una vez configurado el proyecto, ejecuta:

```bash
docker compose up --build
```

Docker Compose levantará los servicios definidos en:

```text
docker-compose.yml
```

###  Detener los servicios

```bash
docker compose down
```

###  Ejecutar en segundo plano

```bash
docker compose up --build -d
```

### 📋 Ver logs

```bash
docker compose logs -f
```

Para consultar únicamente los logs del motor de IA:

```bash
docker compose logs -f modelo_ia
```

---

# 🖥️ Puertos y Servicios

| Servicio         | Tecnología        | Puerto | URL                            |
| ---------------- | ----------------- | -----: | ------------------------------ |
|  Frontend     | React + Vite      | `3000` | `http://localhost:3000`        |
|  Backend API   | Node.js + Express | `4000` | `http://localhost:4000/api`    |
|  Visualizer AI | Python + Flask    | `5000` | `http://localhost:5000/stream` |

### Acceso rápido

**Panel de control**

```text
http://localhost:3000
```

**API REST**

```text
http://localhost:4000/api
```

**Stream de video**

```text
http://localhost:5000/stream
```

---

#  Integración del Visualizador en React

El stream MJPEG puede integrarse directamente en React utilizando un elemento `<img>`.

### `CameraViewer.jsx`

```jsx
export const CameraViewer = () => {
  return (
    <div className="relative rounded-lg overflow-hidden border border-slate-700 bg-black">

      <h2 className="text-white p-2 font-bold text-sm bg-slate-800">
        Monitoreo en Vivo — Vigilia AI Engine
      </h2>

      <img
        src="http://localhost:5000/stream"
        alt="Transmisión en vivo Vigilia"
        className="w-full h-auto object-contain"
      />

    </div>
  );
};
```

El navegador recibirá continuamente los frames procesados por el motor de IA.

---

# ⚙️ Ajuste de Reglas de Negocio

Las zonas de interés y los tiempos de alerta pueden configurarse desde:

```text
modelo_IA/config/behavior_rules.json
```

Ejemplo:

```json
{
  "zones": [
    {
      "zone_id": "EXIT_DOOR_01",
      "coordinates": [
        [100, 200],
        [300, 200],
        [300, 400],
        [100, 400]
      ],
      "max_allowed_loitering_seconds": 3.0,
      "alert_level": "CRITICAL"
    }
  ]
}
```

###  `max_allowed_loitering_seconds`

Define el tiempo máximo permitido para que un `Track ID` permanezca dentro de una zona determinada.

Por ejemplo:

```json
"max_allowed_loitering_seconds": 3.0
```

significa que, bajo las reglas configuradas, permanecer más de **3 segundos** puede generar una condición de alerta.

El sistema puede clasificar el evento como:

```text
NORMAL
LOITERING
POSSIBLE_THEFT
```

> La clasificación final depende de las reglas configuradas en el motor de decisión y del contexto analizado.

---

#  Clasificación de Eventos

El motor utiliza diferentes estados para representar el comportamiento detectado.

| Estado           | Descripción                                                          |
| ---------------- | -------------------------------------------------------------------- |
| `NORMAL`         | No se detecta una condición que active una alerta.                   |
| `LOITERING`      | Se detecta permanencia prolongada en una zona configurada.           |
| `POSSIBLE_THEFT` | Las reglas configuradas determinan una posible condición sospechosa. |

> `POSSIBLE_THEFT` representa una **clasificación automatizada del sistema**, no una determinación de que un robo haya ocurrido.

---

# 📊 Métricas y Monitoreo

`MetricsService` registra información relacionada con el funcionamiento del sistema.

Entre las métricas consideradas se encuentran:

*  FPS
*  Latencia
*  Detecciones
*  Track IDs
*  Eventos
*  Zonas activadas
*  Logs de auditoría

Ejemplo conceptual:

```text
Frame #12540
│
├── FPS: 28.4
├── Latencia: 42 ms
├── Personas detectadas: 3
├── Track IDs: 12, 18, 21
├── Zona activa: EXIT_DOOR_01
└── Evento: LOITERING
```

---

# 🐳 Arquitectura de Microservicios

El sistema está dividido en servicios independientes:

```text
                    ┌─────────────────────┐
                    │     Docker Compose  │
                    └──────────┬──────────┘
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                  │
            ▼                  ▼                  ▼
     ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
     │  Frontend   │    │   Backend   │    │  Modelo IA  │
     │    React    │    │    Node     │    │   Python    │
     │             │    │             │    │             │
     │    :3000    │    │    :4000    │    │    :5000    │
     └─────────────┘    └──────┬──────┘    └─────────────┘
                                │
                                ▼
                         ┌─────────────┐
                         │    MySQL    │
                         └─────────────┘
```

### Ventajas de esta arquitectura

* 🔹 Separación de responsabilidades.
* 🔹 Desarrollo independiente de cada servicio.
* 🔹 Escalabilidad.
* 🔹 Facilidad para actualizar componentes.
* 🔹 Aislamiento mediante contenedores.
* 🔹 Despliegue reproducible mediante Docker Compose.

---

#  Solución de Problemas Frecuentes

## 1. Error accediendo a `/dev/video0`

### Linux

Verifica que el dispositivo exista:

```bash
ls -l /dev/video0
```

Si es necesario, ajusta temporalmente los permisos:

```bash
sudo chmod 666 /dev/video0
```

También es necesario que el contenedor tenga acceso al dispositivo mediante la configuración correspondiente en `docker-compose.yml`.

---

## 2. Windows / macOS / WSL2

Docker Desktop no proporciona acceso directo a dispositivos USB/Webcam Linux como `/dev/video0` de la misma forma que un host Linux.

Para pruebas locales, una alternativa es utilizar un archivo `.mp4`:

```env
VIDEO_DEVICE_INDEX="/app/videos/demo_test.mp4"
```

Y montar el directorio correspondiente dentro del contenedor.

Ejemplo conceptual:

```yaml
volumes:
  - ./videos:/app/videos
```

---

## 3. Parpadeo o pérdida de seguimiento

Si los `Track ID` se reinician o duplican inesperadamente después de reconectar una transmisión, verifica que `VideoService` mantenga correctamente el estado necesario del procesamiento.

Especialmente:

```text
Frame Counter
      │
      ▼
VideoService
      │
      ▼
TrackingService
      │
      ▼
Track IDs
```

El contador global de frames debe manejar correctamente las reconexiones para evitar inconsistencias en el seguimiento.

---

## 4. Revisar el estado de los contenedores

Ejecuta:

```bash
docker compose ps
```

Para revisar los logs:

```bash
docker compose logs -f
```

Para revisar un servicio específico:

```bash
docker compose logs -f backend
```

```bash
docker compose logs -f frontend
```

```bash
docker compose logs -f modelo_ia
```

---

# 🧪 Pruebas con Archivo de Video

Para realizar pruebas sin una cámara física:

1. Coloca un archivo `.mp4` dentro de:

```text
videos/
└── demo_test.mp4
```

2. Configura:

```env
VIDEO_DEVICE_INDEX="/app/videos/demo_test.mp4"
```

3. Asegúrate de montar el directorio:

```yaml
volumes:
  - ./videos:/app/videos
```

4. Inicia el sistema:

```bash
docker compose up --build
```

5. Abre el visualizador:

```text
http://localhost:5000/stream
```

---

# 📌 Flujo Completo de Ejecución

```text
                 ┌───────────────┐
                 │ Cámara / MP4  │
                 └───────┬───────┘
                         │
                         ▼
                 ┌───────────────┐
                 │ VideoService  │
                 └───────┬───────┘
                         │
                         ▼
                 ┌───────────────┐
                 │ Preprocess    │
                 └───────┬───────┘
                         │
                         ▼
                 ┌───────────────┐
                 │    YOLOv8     │
                 └───────┬───────┘
                         │
                         ▼
                 ┌───────────────┐
                 │   Tracking    │
                 └───────┬───────┘
                         │
                         ▼
                 ┌───────────────┐
                 │     Pose      │
                 └───────┬───────┘
                         │
                         ▼
                 ┌───────────────┐
                 │   Behavior    │
                 └───────┬───────┘
                         │
                         ▼
                 ┌───────────────┐
                 │ DecisionEngine│
                 └───────┬───────┘
                         │
                         ▼
                 ┌───────────────┐
                 │  Classifier   │
                 └───────┬───────┘
                         │
                         ▼
                 ┌───────────────┐
                 │    Metrics    │
                 └───────┬───────┘
                         │
                         ▼
                 ┌───────────────┐
                 │  Visualizer   │
                 └───────┬───────┘
                         │
                         ▼
                  HTTP MJPEG
                         │
                         ▼
                 ┌───────────────┐
                 │ React Frontend│
                 └───────────────┘
```

---

#  Licencia

Este proyecto se encuentra bajo la **Licencia MIT**.

Consulta el archivo:

```text
LICENSE
```

para conocer los términos completos de la licencia.

---

#  Proyecto

**Vigilia — Sistema de Videovigilancia y Análisis de Comportamiento**

Arquitectura basada en:

```text
React
  +
Node.js / Express
  +
Python / YOLOv8
  +
MySQL
  +
Docker Compose
```

>  **Vigilia** integra procesamiento de video, visión por computadora, análisis de comportamiento y visualización en tiempo real dentro de una arquitectura modular y contenerizada.
