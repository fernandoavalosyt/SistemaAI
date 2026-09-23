from loguru import logger
from app.config import settings

class RiskScoreEngine:
    """Evalúa el nivel de riesgo de somnolencia basado en inactividad prolongada."""
    
    def __init__(self):
        # Umbrales (En un futuro vendrán de settings para ser configurables por cámara)
        self.sleep_threshold = 60.0  # Empieza a sumar riesgo tras 60 segundos sin moverse
        self.max_critical_time = 180.0 # 3 minutos sin moverse = Riesgo 100% (Alerta roja)
        
    def evaluate(self, behavior_data):
        """
        Calcula el riesgo de somnolencia (0-100) para cada persona detectada.
        """
        risks = {}
        
        for track_id, data in behavior_data.items():
            inactivity = data.get('inactivity_time', 0)
            score = 0.0
            
            # Si supera el tiempo de tolerancia, el riesgo sube progresivamente
            if inactivity > self.sleep_threshold:
                time_over_threshold = inactivity - self.sleep_threshold
                window = self.max_critical_time - self.sleep_threshold
                
                # Regla de 3 simple capada a 100
                score = min(100.0, (time_over_threshold / window) * 100.0)
                
            risks[track_id] = score
            
        return risks
