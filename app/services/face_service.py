import insightface
from insightface.app import FaceAnalysis
import numpy as np
import cv2
from typing import Optional, Tuple
from app.core.config import settings

class FaceRecognitionService:
    def __init__(self):
        self.app = FaceAnalysis(
            name='buffalo_l',
            providers=['CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
    
    def detect_face(self, image: np.ndarray) -> Optional[dict]:
        faces = self.app.get(image)
        
        if len(faces) == 0:
            return None
        
        face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        
        # CAMBIO: usar det_score como liveness (0.3 es más realista)
        liveness_score = float(face.det_score)
        
        # Ajustar umbral a 0.3 o menor
        if liveness_score < 0.3:  # CAMBIAR AQUÍ
            return {
                "face": None,
                "embedding": None,
                "liveness_score": liveness_score,
                "is_spoofing": True
            }
        
        embedding = face.normed_embedding
        
        return {
            "face": face,
            "embedding": embedding,
            "liveness_score": liveness_score,
            "is_spoofing": False,
            "bbox": face.bbox.tolist()
        }
    
    def calculate_quality(self, face) -> float:
        """
        Calcula calidad del rostro (0-1)
        """
        # Basado en tamaño y det_score
        bbox = face.bbox
        size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        quality = min(1.0, size / 50000 * face.det_score)
        return float(quality)
    
    def cosine_distance(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calcula distancia coseno entre embeddings
        """
        return 1 - np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

# Singleton
face_service = FaceRecognitionService()