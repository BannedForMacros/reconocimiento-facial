"""
Liveness Detection — Análisis heurístico multi-cue
Detecta ataques de spoofing via textura LBP, frecuencia FFT,
análisis de color YCbCr, gradiente Sobel y nitidez Laplaciana.
"""

import cv2
import numpy as np
from loguru import logger
from app.core.config import settings


class LivenessService:

    def __init__(self):
        self.threshold = settings.LIVENESS_THRESHOLD

    def analyze(self, image_bgr: np.ndarray, bbox: list) -> dict:
        """
        Analiza si un rostro es real o spoof.

        Args:
            image_bgr: Frame completo BGR
            bbox:      [x1, y1, x2, y2] del detector de rostros

        Returns:
            dict con score, is_live y scores individuales para calibracion
        """
        # Crop interno con margen 20%
        h_img, w_img = image_bgr.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        fw, fh = x2 - x1, y2 - y1
        mx, my = int(fw * 0.20), int(fh * 0.20)
        x1c = max(0, x1 - mx); y1c = max(0, y1 - my)
        x2c = min(w_img, x2 + mx); y2c = min(h_img, y2 + my)
        face = image_bgr[y1c:y2c, x1c:x2c]

        if face is None or face.size == 0 or face.shape[0] < 40 or face.shape[1] < 40:
            return {"is_live": False, "score": 0.0, "scores": []}

        face = cv2.resize(face, (128, 128))
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        t = self._analyze_texture(gray)
        f = self._analyze_frequency(gray)
        c = self._analyze_color(face)
        s = self._analyze_sharpness(gray)

        # Pesos calibrados con datos reales de esta camara:
        # - sharpness es el mayor discriminador (cara=0.36, pantalla=0.27)
        # - color se INVIERTE: pantallas OLED tienen colores mas vividos que piel real
        # - texture ayuda un poco (cara > pantalla)
        # - frequency y gradient se descartan (sin poder discriminatorio aqui)
        final = t * 0.20 + (1.0 - c) * 0.20 + s * 0.60

        logger.debug(
            f"HEURISTIC texture={t:.3f} color={c:.3f}(inv={1-c:.3f}) "
            f"sharp={s:.3f} → FINAL={final:.3f} "
            f"(threshold={self.threshold})"
        )

        return {
            "is_live": final >= self.threshold,
            "score":   round(float(final), 3),
            "scores":  {
                "texture":   round(float(t), 3),
                "color_inv": round(float(1.0 - c), 3),
                "sharpness": round(float(s), 3),
            },
        }

    # ── LBP Textura ─────────────────────────────────────────────────────────

    def _compute_lbp(self, gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape
        padded = np.pad(gray.astype(np.int16), 1, mode="edge")
        center = padded[1:h+1, 1:w+1]
        offsets = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
        lbp = np.zeros((h, w), dtype=np.uint8)
        for i, (dy, dx) in enumerate(offsets):
            nb = padded[1+dy:h+1+dy, 1+dx:w+1+dx]
            lbp |= ((nb >= center).astype(np.uint8) << i)
        return lbp

    def _analyze_texture(self, gray: np.ndarray) -> float:
        lbp = self._compute_lbp(gray)
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(np.float64)
        total = hist.sum()
        if total == 0:
            return 0.0
        hist /= total
        nonzero = hist[hist > 0]
        entropy = -np.sum(nonzero * np.log2(nonzero))
        entropy_score = np.clip((entropy - 4.0) / 3.0, 0.0, 1.0)
        var_score = np.clip(np.var(lbp.astype(np.float64)) / 5000.0, 0.0, 1.0)
        return float(0.6 * entropy_score + 0.4 * var_score)

    # ── FFT Frecuencia ───────────────────────────────────────────────────────

    def _analyze_frequency(self, gray: np.ndarray) -> float:
        f = np.fft.fft2(gray.astype(np.float64))
        fshift = np.fft.fftshift(f)
        magnitude = np.log1p(np.abs(fshift))
        rows, cols = magnitude.shape
        crow, ccol = rows // 2, cols // 2
        r_low  = min(rows, cols) // 8
        r_high = min(rows, cols) // 3
        Y, X = np.ogrid[:rows, :cols]
        dist_sq = (Y - crow)**2 + (X - ccol)**2
        low_mask  = dist_sq <= r_low**2
        high_mask = dist_sq > r_high**2
        mid_mask  = ~low_mask & ~high_mask
        low_e  = np.mean(magnitude[low_mask])  if np.any(low_mask)  else 1.0
        high_e = np.mean(magnitude[high_mask]) if np.any(high_mask) else 0.0
        if low_e < 1e-7:
            return 0.5
        ratio = high_e / low_e
        freq_score = float(np.clip((0.7 - ratio) / 0.4, 0.0, 1.0))
        mid_vals = magnitude[mid_mask]
        if len(mid_vals) > 10:
            mean_m = np.mean(mid_vals)
            std_m  = np.std(mid_vals)
            if std_m > 0:
                peaks = np.sum(mid_vals > mean_m + 3 * std_m)
                peak_ratio = peaks / len(mid_vals)
                moire = float(np.clip(1.0 - peak_ratio * 50, 0.0, 1.0))
                freq_score = 0.6 * freq_score + 0.4 * moire
        return freq_score

    # ── Color YCbCr ──────────────────────────────────────────────────────────

    def _analyze_color(self, face_bgr: np.ndarray) -> float:
        ycrcb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2YCrCb)
        cb = ycrcb[:,:,2].astype(np.float64)
        cr = ycrcb[:,:,1].astype(np.float64)
        cb_score = float(np.clip((np.std(cb) - 3) / 15, 0.0, 1.0))
        cr_score = float(np.clip((np.std(cr) - 3) / 15, 0.0, 1.0))
        hsv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:,:,1].astype(np.float64)
        sat_m, sat_s = np.mean(sat), np.std(sat)
        sat_score = 1.0
        if sat_m < 30 or sat_m > 140: sat_score *= 0.5
        if sat_s < 10 or sat_s > 80:  sat_score *= 0.7
        return float(np.clip(cb_score * 0.3 + cr_score * 0.3 + sat_score * 0.4, 0.0, 1.0))

    # ── Gradiente Sobel ──────────────────────────────────────────────────────

    def _analyze_gradient(self, gray: np.ndarray) -> float:
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(sx**2 + sy**2)
        grad_score = float(np.clip(np.var(mag) / 2500.0, 0.0, 1.0))
        angle = np.arctan2(sy, sx)
        ahist, _ = np.histogram(angle, bins=36, range=(-np.pi, np.pi))
        ahist = ahist.astype(np.float64)
        total = ahist.sum()
        if total > 0:
            ahist /= total
            nz = ahist[ahist > 0]
            orient_entropy = -np.sum(nz * np.log2(nz + 1e-9))
            orient_score = float(np.clip(orient_entropy / 5.0, 0.0, 1.0))
        else:
            orient_score = 0.0
        return float(0.5 * grad_score + 0.5 * orient_score)

    # ── Nitidez Laplaciana ───────────────────────────────────────────────────

    def _analyze_sharpness(self, gray: np.ndarray) -> float:
        lap_var = np.var(cv2.Laplacian(gray, cv2.CV_64F))
        if lap_var < 50:
            sharpness_score = 0.1
        elif lap_var < 200:
            sharpness_score = float(np.clip((lap_var - 50) / 150, 0.1, 0.5))
        elif lap_var <= 3000:
            sharpness_score = float(np.clip(1.0 - abs(lap_var - 800) / 2200, 0.4, 1.0))
        else:
            sharpness_score = float(np.clip(1.0 - (lap_var - 3000) / 7000, 0.0, 0.5))

        h, w = gray.shape
        bs = 16
        block_means = [
            np.mean(gray[y:y+bs, x:x+bs])
            for y in range(0, h - bs, bs)
            for x in range(0, w - bs, bs)
        ]
        if len(block_means) > 4:
            bv = np.var(block_means)
            u = float(np.clip(bv / 400.0, 0.0, 1.0))
            if bv > 800:
                u = float(np.clip(1.0 - (bv - 800) / 1200, 0.0, 1.0))
        else:
            u = 0.5
        return float(np.clip(0.6 * sharpness_score + 0.4 * u, 0.0, 1.0))


liveness_service = LivenessService()
