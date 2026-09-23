import cv2
from loguru import logger
from app.core.detector import YoloDetector
from app.core.behavior import BehaviorAnalyzer
from app.core.risk_engine import RiskScoreEngine

class VideoPipeline:
    """Orquestador que une la detección, análisis de comportamiento y riesgo."""
    
    def __init__(self):
        self.detector = YoloDetector()
        self.behavior = BehaviorAnalyzer()
        self.risk_engine = RiskScoreEngine()
        
    def process_frame(self, frame):
        """
        Procesa un solo frame:
        1. Tracking de YOLO
        2. Análisis de comportamiento
        3. Cálculo de riesgo
        4. Dibuja los resultados en el frame
        """
        # 1. Detección y Tracking
        results = self.detector.track(frame)
        
        # Extraer cajas para dibujarlas luego
        tracks = results[0] 
        
        # 2. Análisis de comportamiento
        behavior_data = self.behavior.update([tracks])
        
        # 3. Calcular riesgo
        risks = self.risk_engine.evaluate(behavior_data)
        
        # 4. Dibujar visualizaciones en el frame
        annotated_frame = frame.copy()
        
        if tracks.boxes is not None and tracks.boxes.id is not None:
            boxes = tracks.boxes.xyxy.cpu().numpy()
            track_ids = tracks.boxes.id.int().cpu().numpy()
            
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                risk = risks.get(track_id, 0)
                
                # Color basado en riesgo (Verde -> Amarillo -> Rojo)
                if risk < 30:
                    color = (0, 255, 0) # Verde
                elif risk < 70:
                    color = (0, 255, 255) # Amarillo
                else:
                    color = (0, 0, 255) # Rojo
                    
                # Dibujar caja
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Etiqueta visual
                b_data = behavior_data.get(track_id, {})
                inact_time = int(b_data.get('inactivity_time', 0))
                label = f"ID:{track_id} Risk:{risk:.0f}% ZzZ:{inact_time}s"
                
                cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            
        return annotated_frame, behavior_data, risks
