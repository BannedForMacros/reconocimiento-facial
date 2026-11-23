"""
Servicio de Anti-Spoofing (Liveness Detection) basado en redes neuronales.

Modelos: MiniFASNetV2 (escala 2.7) + MiniFASNetV1SE (escala 4.0)
  - Origen: Silent-Face-Anti-Spoofing (Minivision), Apache-2.0
  - ONNX:   github.com/yakhyo/face-anti-spoofing (releases/weights), Apache-2.0
  - Entrada: recorte 80x80 BGR, valores crudos 0-255 (sin normalizar), NCHW
  - Salida:  3 clases; indice 1 = rostro real, 0 y 2 = ataques (foto/pantalla)

Cada modelo mira la cara con un contexto distinto alrededor (2.7x y 4x el
tamano del bbox), lo que permite ver bordes de celulares, marcos de fotos,
reflejos, etc. Las probabilidades de ambos se promedian (ensamble original
de Silent-Face).
"""

import hashlib
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from loguru import logger

from app.core.config import settings

MODELS_DIR = Path(__file__).resolve().parents[2] / "models" / "antispoof"

# (archivo, escala de recorte, sha256 esperado)
MODELS = [
    ("MiniFASNetV2.onnx", 2.7,
     "b32929adc2d9c34b9486f8c4c7bc97c1b69bc0ea9befefc380e4faae4e463907"),
    ("MiniFASNetV1SE.onnx", 4.0,
     "ebab7f90c7833fbccd46d3a555410e78d969db5438e169b6524be444862b3676"),
]

REAL_CLASS = 1


class LivenessService:

    def __init__(self):
        self.threshold = settings.LIVENESS_THRESHOLD
        self.min_brightness = settings.LIVENESS_MIN_BRIGHTNESS
        self.models = []

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2

        for filename, scale, expected_sha in MODELS:
            path = MODELS_DIR / filename
            if not path.exists():
                raise FileNotFoundError(
                    f"Falta el modelo anti-spoofing {path}. "
                    f"Descargalo de https://github.com/yakhyo/face-anti-spoofing/releases/tag/weights"
                )
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            if sha != expected_sha:
                raise RuntimeError(f"Hash invalido para {filename}: {sha}")

            session = ort.InferenceSession(
                str(path), sess_options=opts, providers=["CPUExecutionProvider"]
            )
            inp = session.get_inputs()[0]
            h, w = inp.shape[2], inp.shape[3]
            self.models.append({
                "name": filename.removesuffix(".onnx"),
                "session": session,
                "input": inp.name,
                "scale": scale,
                "size": (w, h),
            })

        logger.info(
            f"Liveness cargado: {', '.join(m['name'] for m in self.models)} "
            f"(umbral={self.threshold})"
        )

    # ------------------------------------------------------------------ #

    @staticmethod
    def _crop(image: np.ndarray, bbox: list, scale: float, size: tuple) -> np.ndarray:
        """
        Recorte con contexto, igual que CropImage de Silent-Face: la caja se
        agranda 'scale' veces desde el centro y, si se sale de la imagen, se
        desplaza (no se encoge) para conservar el tamano.
        """
        src_h, src_w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        box_w, box_h = x2 - x1, y2 - y1

        scale = min((src_h - 1) / box_h, (src_w - 1) / box_w, scale)
        new_w, new_h = box_w * scale, box_h * scale
        cx, cy = x1 + box_w / 2, y1 + box_h / 2

        left, top = cx - new_w / 2, cy - new_h / 2
        right, bottom = cx + new_w / 2, cy + new_h / 2

        if left < 0:
            right -= left
            left = 0
        if top < 0:
            bottom -= top
            top = 0
        if right > src_w - 1:
            left -= right - src_w + 1
            right = src_w - 1
        if bottom > src_h - 1:
            top -= bottom - src_h + 1
            bottom = src_h - 1

        left, top = max(0, int(left)), max(0, int(top))
        right, bottom = int(right), int(bottom)
        crop = image[top:bottom + 1, left:right + 1]
        return cv2.resize(crop, size)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()

    def analyze(self, image_bgr: np.ndarray, bbox: list) -> dict:
        """
        Analiza si el rostro es real o un ataque (foto, pantalla, video).

        Args:
            image_bgr: fotograma COMPLETO en BGR (el modelo necesita el contexto
                       alrededor de la cara, no solo el recorte).
            bbox: [x1, y1, x2, y2] del rostro detectado.

        Returns:
            dict con is_live, score (probabilidad de rostro real, 0-1),
            detail ("ok" | "low_light" | "invalid_input") y score por modelo.
        """
        if image_bgr is None or image_bgr.size == 0 or bbox is None:
            return {"is_live": False, "score": 0.0, "detail": "invalid_input"}

        x1, y1, x2, y2 = [float(v) for v in bbox]
        if x2 - x1 < 2 or y2 - y1 < 2:
            return {"is_live": False, "score": 0.0, "detail": "invalid_input"}

        # Con muy poca luz el modelo no es confiable: se informa como falta de
        # luz (no como suplantacion) para no culpar al usuario.
        fx1, fy1 = max(0, int(x1)), max(0, int(y1))
        face = image_bgr[fy1:int(y2), fx1:int(x2)]
        brightness = float(np.mean(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY))) if face.size else 0.0
        if brightness < self.min_brightness:
            return {
                "is_live": False,
                "score": 0.0,
                "detail": "low_light",
                "brightness": round(brightness, 1),
            }

        probs_sum = np.zeros(3, dtype=np.float64)
        per_model = {}
        for m in self.models:
            crop = self._crop(image_bgr, [x1, y1, x2, y2], m["scale"], m["size"])
            tensor = crop.astype(np.float32).transpose(2, 0, 1)[np.newaxis]
            logits = m["session"].run(None, {m["input"]: tensor})[0][0]
            probs = self._softmax(logits)
            probs_sum += probs
            per_model[m["name"]] = round(float(probs[REAL_CLASS]), 3)

        probs_avg = probs_sum / len(self.models)
        score = float(probs_avg[REAL_CLASS])

        return {
            "is_live": score >= self.threshold,
            "score": round(score, 3),
            "detail": "ok",
            "brightness": round(brightness, 1),
            "models": per_model,
        }


liveness_service = LivenessService()
