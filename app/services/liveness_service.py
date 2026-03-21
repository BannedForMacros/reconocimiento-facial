"""
Servicio de Anti-Spoofing (Liveness Detection)

Detecta intentos de suplantacion mediante analisis multi-cue:
1. Textura (LBP - Local Binary Patterns): patrones artificiales en fotos impresas
2. Frecuencia (FFT): patrones moire de pantallas de celular/monitor
3. Color (YCbCr): distorsion cromatica en reproducciones

Referencias academicas:
- Maatta et al. (2011) - Face spoofing detection using micro-texture analysis
- Boulkenafet et al. (2016) - Face spoofing detection using colour texture analysis
"""

import cv2
import numpy as np
from app.core.config import settings


class LivenessService:

    def __init__(self):
        self.threshold = settings.LIVENESS_THRESHOLD

    def analyze(self, face_crop_bgr: np.ndarray) -> dict:
        """
        Analiza si un rostro recortado es real o spoof.

        Args:
            face_crop_bgr: Imagen BGR del rostro recortado, minimo 80x80 px

        Returns:
            dict con score (0-1), is_live (bool), y scores individuales
        """
        if face_crop_bgr is None or face_crop_bgr.size == 0:
            return {"is_live": False, "score": 0.0, "detail": "invalid_input"}

        h, w = face_crop_bgr.shape[:2]
        if h < 40 or w < 40:
            return {"is_live": False, "score": 0.0, "detail": "face_too_small"}

        face = cv2.resize(face_crop_bgr, (128, 128))
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        texture_score = self._analyze_texture(gray)
        frequency_score = self._analyze_frequency(gray)
        color_score = self._analyze_color(face)

        final_score = (
            texture_score * 0.40 +
            frequency_score * 0.35 +
            color_score * 0.25
        )

        return {
            "is_live": final_score >= self.threshold,
            "score": round(float(final_score), 3),
            "texture": round(float(texture_score), 3),
            "frequency": round(float(frequency_score), 3),
            "color": round(float(color_score), 3),
        }

    # ------------------------------------------------------------------ #
    #  LBP (Local Binary Patterns) - Textura                              #
    # ------------------------------------------------------------------ #

    def _compute_lbp(self, gray: np.ndarray) -> np.ndarray:
        """LBP vectorizado con 8 vecinos, radio 1."""
        h, w = gray.shape
        padded = np.pad(gray.astype(np.int16), 1, mode="edge")
        center = padded[1 : h + 1, 1 : w + 1]

        offsets = [
            (-1, -1), (-1, 0), (-1, 1), (0, 1),
            (1, 1), (1, 0), (1, -1), (0, -1),
        ]

        lbp = np.zeros((h, w), dtype=np.uint8)
        for i, (dy, dx) in enumerate(offsets):
            neighbor = padded[1 + dy : h + 1 + dy, 1 + dx : w + 1 + dx]
            lbp |= ((neighbor >= center).astype(np.uint8) << i)

        return lbp

    def _analyze_texture(self, gray: np.ndarray) -> float:
        """
        Rostros reales tienen distribucion LBP con alta entropia (textura organica).
        Fotos impresas/pantallas tienen baja entropia (patrones regulares de trama/pixel).
        """
        lbp = self._compute_lbp(gray)

        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(np.float64)
        total = hist.sum()
        if total == 0:
            return 0.0
        hist /= total

        nonzero = hist[hist > 0]
        entropy = -np.sum(nonzero * np.log2(nonzero))

        # Max entropia para 256 bins = 8.0
        # Rostros reales: ~5.5-7.0 | Spoof: ~3.5-5.5
        entropy_score = np.clip((entropy - 4.0) / 3.0, 0.0, 1.0)

        lbp_var = np.var(lbp.astype(np.float64))
        var_score = np.clip(lbp_var / 5000.0, 0.0, 1.0)

        return float(0.6 * entropy_score + 0.4 * var_score)

    # ------------------------------------------------------------------ #
    #  FFT - Frecuencia (deteccion de moire)                              #
    # ------------------------------------------------------------------ #

    def _analyze_frequency(self, gray: np.ndarray) -> float:
        """
        Pantallas generan patrones moire detectables en el dominio de frecuencia.
        Fotos impresas muestran frecuencias de trama de impresion.
        Rostros reales tienen caida de frecuencia natural.
        """
        f = np.fft.fft2(gray.astype(np.float64))
        fshift = np.fft.fftshift(f)
        magnitude = np.log1p(np.abs(fshift))

        rows, cols = magnitude.shape
        crow, ccol = rows // 2, cols // 2

        r_low = min(rows, cols) // 8
        r_high = min(rows, cols) // 3

        Y, X = np.ogrid[:rows, :cols]
        dist_sq = (Y - crow) ** 2 + (X - ccol) ** 2

        low_mask = dist_sq <= r_low ** 2
        high_mask = dist_sq > r_high ** 2
        mid_mask = ~low_mask & ~high_mask

        low_energy = np.mean(magnitude[low_mask]) if np.any(low_mask) else 1.0
        high_energy = np.mean(magnitude[high_mask]) if np.any(high_mask) else 0.0

        if low_energy < 1e-7:
            return 0.5

        ratio = high_energy / low_energy
        # Rostros reales: ratio ~0.3-0.5 | Pantallas: ~0.5-0.8
        freq_score = np.clip((0.7 - ratio) / 0.4, 0.0, 1.0)

        # Deteccion de picos de moire en frecuencias medias-altas
        mid_values = magnitude[mid_mask]
        if len(mid_values) > 10:
            mean_mid = np.mean(mid_values)
            std_mid = np.std(mid_values)
            if std_mid > 0:
                peaks = np.sum(mid_values > mean_mid + 3 * std_mid)
                peak_ratio = peaks / len(mid_values)
                moire_score = np.clip(1.0 - peak_ratio * 50, 0.0, 1.0)
                freq_score = 0.6 * freq_score + 0.4 * moire_score

        return float(freq_score)

    # ------------------------------------------------------------------ #
    #  Color (YCbCr + HSV)                                                #
    # ------------------------------------------------------------------ #

    def _analyze_color(self, face_bgr: np.ndarray) -> float:
        """
        Piel real tiene distribucion cromatica especifica en YCbCr.
        Reproducciones (impresion/pantalla) distorsionan los canales de crominancia.
        """
        ycrcb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2YCrCb)

        cb = ycrcb[:, :, 2].astype(np.float64)
        cr = ycrcb[:, :, 1].astype(np.float64)

        cb_std = np.std(cb)
        cr_std = np.std(cr)

        # Rostros reales: std crominancia ~5-20 | Pantallas: ~2-8
        cb_score = np.clip((cb_std - 3) / 15, 0.0, 1.0)
        cr_score = np.clip((cr_std - 3) / 15, 0.0, 1.0)

        # Saturacion HSV
        hsv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1].astype(np.float64)
        sat_mean = np.mean(sat)
        sat_std = np.std(sat)

        sat_score = 1.0
        if sat_mean < 30 or sat_mean > 140:
            sat_score *= 0.5
        if sat_std < 10 or sat_std > 80:
            sat_score *= 0.7

        return float(np.clip(
            cb_score * 0.3 + cr_score * 0.3 + sat_score * 0.4,
            0.0, 1.0,
        ))


liveness_service = LivenessService()
