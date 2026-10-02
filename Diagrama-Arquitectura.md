CÁMARA / DVR
              │
             RTSP
              ↓
     ┌─────────────────┐
     │ VIDEO INGESTION │ (Servicio 1 - Completado)
     └────────┬────────┘
              ↓ [Comunicación: Frame + Metadata Original]
     ┌─────────────────┐
     │  PREPROCESSING  │ <--- [ESTAMOS AQUÍ]
     └────────┬────────┘
              ↓ [Comunicación: Frame Normalizado + Metadata Transformación]
     ┌─────────────────┐
     │ YOLO / DETECTION│ (Servicio 3)
     └────────┬────────┘
              ↓
     ┌─────────────────┐
     │    TRACKING     │
     └────────┬────────┘
              ↓
     ┌─────────────────┐
     │ POSE / MODELOS  │
     └────────┬────────┘
              ↓
     ┌─────────────────┐
     │ BEHAVIOR ENGINE │
     └────────┬────────┘
              ↓
     ┌─────────────────┐
     │ DECISION ENGINE │
     └────────┬────────┘
              ↓
     ┌─────────────────┐
     │ EVENT / EVIDENCE│
     └────────┬────────┘
              ↓
           ALERTAS