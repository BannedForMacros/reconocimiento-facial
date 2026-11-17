"""
Anti-spoofing con modelo entrenado: MiniFASNetV2 + MiniFASNetV1SE
(Silent-Face-Anti-Spoofing de Minivision, Apache-2.0; ONNX de
github.com/yakhyo/face-anti-spoofing).

Cada modelo mira la cara a una escala distinta (2.7x y 4.0x el bbox: incluye bordes
de la foto/pantalla) y devuelve 3 clases; la 1 es "real". Se promedian ambos.
Complementa a la heuristica de liveness_service.py.
"""
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from loguru import logger

from app.core.config import settings
from app.services.gpu import providers, usa_gpu

DIR_MODELOS = Path(__file__).resolve().parents[2] / "models_ml" / "antispoof"
MODELOS = {"MiniFASNetV2.onnx": 2.7, "MiniFASNetV1SE.onnx": 4.0}
CLASE_REAL = 1


class AntiSpoofService:

    def __init__(self):
        self.sesiones = []
        for archivo, escala in MODELOS.items():
            ruta = DIR_MODELOS / archivo
            if not ruta.exists():
                logger.warning(f"Anti-spoofing: falta {ruta}; modelo desactivado")
                self.sesiones = []
                return
            s = ort.InferenceSession(str(ruta), providers=providers())
            self.sesiones.append((s, s.get_inputs()[0].name, escala))
        logger.info(
            f"Anti-spoofing MiniFASNet cargado "
            f"({'GPU' if usa_gpu(self.sesiones[0][0]) else 'CPU'})"
        )

    @property
    def disponible(self) -> bool:
        return bool(self.sesiones)

    @staticmethod
    def _recorte(image: np.ndarray, bbox: list, escala: float) -> np.ndarray:
        h_img, w_img = image.shape[:2]
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        escala = min((h_img - 1) / h, (w_img - 1) / w, escala)
        cx, cy = x1 + w / 2, y1 + h / 2
        nw, nh = w * escala, h * escala
        a1, b1 = max(0, int(cx - nw / 2)), max(0, int(cy - nh / 2))
        a2, b2 = min(w_img - 1, int(cx + nw / 2)), min(h_img - 1, int(cy + nh / 2))
        return cv2.resize(image[b1:b2 + 1, a1:a2 + 1], (80, 80))

    def analyze(self, image_bgr: np.ndarray, bbox: list) -> dict:
        """
        Returns:
            dict con prob_real (0-1), is_live y probabilidad por modelo.
        """
        if not self.disponible:
            return {"is_live": True, "prob_real": None, "modelos": {}}
        probs = []
        for sesion, entrada, escala in self.sesiones:
            # BGR, 0-255 sin normalizar, NCHW (como en el entrenamiento original)
            x = self._recorte(image_bgr, bbox, escala).astype(np.float32)
            x = x.transpose(2, 0, 1)[None]
            logits = sesion.run(None, {entrada: x})[0][0]
            p = np.exp(logits - logits.max())
            probs.append(p / p.sum())
        media = np.mean(probs, axis=0)
        prob_real = float(media[CLASE_REAL])
        return {
            "is_live": prob_real >= settings.ANTISPOOF_THRESHOLD,
            "prob_real": round(prob_real, 3),
            "modelos": {
                n: round(float(p[CLASE_REAL]), 3) for n, p in zip(MODELOS, probs)
            },
        }


antispoof_service = AntiSpoofService() if settings.ANTISPOOF_ENABLED else None
