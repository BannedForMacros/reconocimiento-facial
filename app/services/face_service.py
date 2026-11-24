from insightface.app import FaceAnalysis
import numpy as np
from typing import Optional
from loguru import logger
from app.core.config import settings


class FaceRecognitionService:
    def __init__(self):
        logger.info("Cargando modelos InsightFace (buffalo_l)...")
        # Solo los modulos que se usan: deteccion (SCRFD), reconocimiento (ArcFace)
        # y 106 puntos faciales (para la prueba de profundidad). Se omiten el
        # modelo 3D de 68 puntos y el de genero/edad (~145 MB y ~35% del tiempo).
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition", "landmark_2d_106"],
        )
        self.app.prepare(ctx_id=0, det_size=(settings.DET_SIZE, settings.DET_SIZE))
        logger.info(f"Modelos InsightFace cargados (det_size={settings.DET_SIZE})")

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
            "landmarks_106": getattr(face, "landmark_2d_106", None),
        }


# Singleton - se carga UNA VEZ al importar
face_service = FaceRecognitionService()
