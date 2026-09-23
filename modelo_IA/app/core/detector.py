import os
from ultralytics import YOLO
from loguru import logger
from app.config import settings

class YoloDetector:
    """Wrapper para el modelo YOLOv8/v11 de detección de objetos."""
    
    def __init__(self):
        self.model_path = os.path.join(os.path.dirname(__file__), "..", "models", settings.YOLO_MODEL)
        self.confidence = settings.YOLO_CONFIDENCE
        self.iou = settings.YOLO_IOU_THRESHOLD
        
        logger.info(f"Cargando modelo YOLO: {settings.YOLO_MODEL}")
        
        # Si el modelo no existe localmente, YOLO lo descargará la primera vez
        if not os.path.exists(self.model_path):
            logger.info("El modelo no existe localmente. Se descargará automáticamente.")
            self.model = YOLO(settings.YOLO_MODEL)
            # Guardamos el modelo descargado en la carpeta de models
            self.model.save(self.model_path)
        else:
            self.model = YOLO(self.model_path)
            
        logger.info("Modelo YOLO cargado exitosamente.")

    def track(self, frame):
        """
        Ejecuta la detección y seguimiento sobre un frame.
        Retorna los resultados de YOLO con track_ids.
        """
        results = self.model.track(
            frame, 
            conf=self.confidence, 
            iou=self.iou,
            persist=True, # Mantiene el tracking entre frames
            tracker="bytetrack.yaml", # Usa ByteTrack integrado
            verbose=False,
            classes=[0] # Solo detectar personas
        )
        return results
