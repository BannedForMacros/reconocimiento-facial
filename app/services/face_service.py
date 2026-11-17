import insightface
from insightface.app import FaceAnalysis
import numpy as np
import cv2
from typing import Optional
from loguru import logger
from app.services.gpu import providers, usa_gpu


class FaceRecognitionService:
    def __init__(self):
        logger.info("Cargando modelos InsightFace (buffalo_l)...")
        # Solo deteccion + reconocimiento: el sistema no usa landmarks 2D/3D ni
        # genero/edad, y esos 3 modelos extra casi duplicaban el tiempo por frame.
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=providers(),
            allowed_modules=["detection", "recognition"],
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        en_gpu = all(usa_gpu(m.session) for m in self.app.models.values())
        logger.info(f"Modelos InsightFace cargados ({'GPU' if en_gpu else 'CPU'})")

    def detect_face(self, image: np.ndarray) -> Optional[dict]:
        """
        Detecta la cara mas grande en la imagen.
        Retorna embedding normalizado L2, bbox, det_score, dimensiones.
        """
        faces = self.app.get(image)

        if len(faces) == 0:
            return None

        # Seleccionar la cara mas grande
        face = max(
            faces,
            key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
        )

        bbox = face.bbox.tolist()
        face_w = bbox[2] - bbox[0]
        face_h = bbox[3] - bbox[1]

        return {
            "face": face,
            "embedding": face.normed_embedding,
            "det_score": float(face.det_score),
            "bbox": bbox,
            "face_width": face_w,
            "face_height": face_h,
        }

    def get_face_crop(
        self, image: np.ndarray, bbox: list, margin: float = 0.25
    ) -> np.ndarray:
        """Extrae recorte del rostro con margen para analisis de liveness."""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        fw, fh = x2 - x1, y2 - y1
        mx, my = int(fw * margin), int(fh * margin)
        x1 = max(0, x1 - mx)
        y1 = max(0, y1 - my)
        x2 = min(w, x2 + mx)
        y2 = min(h, y2 + my)
        return image[y1:y2, x1:x2].copy()

    def cosine_distance(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Distancia coseno entre dos embeddings."""
        dot = np.dot(emb1, emb2)
        norm = np.linalg.norm(emb1) * np.linalg.norm(emb2)
        return float(1 - dot / (norm + 1e-7))


# Singleton - se carga UNA VEZ al importar
face_service = FaceRecognitionService()
