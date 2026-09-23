import time

class BehaviorAnalyzer:
    """Analiza el comportamiento de los objetos seguidos, buscando inactividad (somnolencia)."""
    
    def __init__(self):
        # track_id: {'first_seen': t, 'last_seen': t, 'last_move_time': t, 'last_pos': (x,y), 'inactivity_time': s}
        self.history = {}
        # Tolerancia de movimiento en píxeles (pequeños ajustes de postura no resetean la inactividad si no superan el umbral)
        self.movement_threshold = 15.0 
        
    def update(self, tracks):
        """
        Actualiza el historial y calcula el tiempo de inactividad de cada ID.
        """
        current_time = time.time()
        behavior_data = {}
        
        for track in tracks:
            if track.id is None:
                continue
                
            for box in track.boxes:
                if box.id is None:
                    continue
                    
                track_id = int(box.id[0].item())
                
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                
                if track_id not in self.history:
                    self.history[track_id] = {
                        'first_seen': current_time,
                        'last_seen': current_time,
                        'last_move_time': current_time,
                        'last_pos': (cx, cy),
                        'inactivity_time': 0
                    }
                else:
                    data = self.history[track_id]
                    data['last_seen'] = current_time
                    
                    # Calcular distancia desde el último punto de movimiento registrado
                    last_px, last_py = data['last_pos']
                    dist = ((cx - last_px)**2 + (cy - last_py)**2)**0.5
                    
                    if dist > self.movement_threshold:
                        # Se movió significativamente, reiniciar contador de sueño
                        data['last_move_time'] = current_time
                        data['last_pos'] = (cx, cy)
                    
                    # Calcular tiempo total inactivo
                    data['inactivity_time'] = current_time - data['last_move_time']
                
                behavior_data[track_id] = {
                    'inactivity_time': self.history[track_id]['inactivity_time'],
                    'cx': cx,
                    'cy': cy
                }
                
        # Limpieza de tracks antiguos (10 segundos sin verse)
        to_remove = [tid for tid, data in self.history.items() if current_time - data['last_seen'] > 10]
        for tid in to_remove:
            del self.history[tid]
            
        return behavior_data
