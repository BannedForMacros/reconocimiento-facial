"""
Reproduce una sesion grabada (cuadros.csv de sesion_camara.py) por la logica REAL del
endpoint: modelo pasivo -> reto activo -> confianza del detector -> votacion.
Muestrea cada 0.4 s (intervalo de la pagina). La identidad se aproxima: "coincide"
si el detector es confiable y la cara esta casi de frente (no se guardaron embeddings).

Uso:  ..\\venv_gpu\\Scripts\\python.exe benchmark_gpu\\simular_reto.py [carpeta_sesion]
"""
import csv
import os
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
os.chdir(RAIZ)
os.environ.setdefault("USE_GPU", "false")

from app.core.config import settings  # noqa: E402
import app.services.reto_service as rs  # noqa: E402
import app.routes.reconocimiento as rec  # noqa: E402

INTERVALO = 0.4
DESFASE = float(os.environ.get("DESFASE", "0"))
carpeta = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(
    (RAIZ / "benchmark_gpu" / "resultados").glob("sesion_*"))[-1]
filas = list(csv.DictReader(open(carpeta / "cuadros.csv", encoding="utf-8")))


class Reloj:
    t = 1_000_000.0
    def time(self):
        return self.t


reloj = Reloj()
rs.time = reloj        # el modulo usa time.time()
rec.time = reloj


def kps_con_giro(g: float):
    """5 puntos sinteticos con el giro medido (solo importan ojos y nariz)."""
    import numpy as np
    return np.array([[0, 0], [1, 0], [0.5 + g, 0.5], [0.2, 1], [0.8, 1]], dtype=float)


def simular(fase: str, lado: str) -> str:
    fs = [r for r in filas if r["fase"] == fase]
    rec._vote_buffers.clear(); rec._cooldowns.clear(); rs.reto_service._estado.clear()
    sesion = "sim"
    rs.reto_service._nuevo(sesion)["lado"] = lado
    inicio = reloj.t
    proximo = DESFASE
    for r in fs:
        t = float(r["t"])
        if t < proximo:
            continue
        proximo = t + INTERVALO
        reloj.t = inicio + t
        if float(r["modelo"]) < settings.ANTISPOOF_THRESHOLD:
            continue  # LIVENESS_FAILED: el reto no avanza
        reto = rs.reto_service.actualizar(sesion, kps_con_giro(float(r["giro"])))
        if float(r["det_score"]) < settings.DETECTION_CONFIDENCE:
            continue
        coincide = abs(float(r["giro"])) < 0.6
        v = rec._add_vote(sesion, 1 if coincide else None, 0.3, permitir_confirmar=reto["cumplido"])
        if v["status"] == "MATCH_CONFIRMED":
            return f"CONFIRMADO en {t:.1f} s"
    return "no confirma"


print(f"Sesion {carpeta.name}: giro >= {settings.RETO_GIRO_MIN}, frente <= {settings.RETO_FRONTAL_MAX}, "
      f"modelo >= {settings.ANTISPOOF_THRESHOLD}, votos {settings.VOTING_THRESHOLD}/{settings.VOTING_WINDOW}\n")
for fase in dict.fromkeys(r["fase"] for r in filas):
    es_real = filas[[r["fase"] for r in filas].index(fase)]["real"] == "True"
    res = {lado: simular(fase, lado) for lado in ("izquierda", "derecha")}
    print(f"{fase:16s} ({'REAL ' if es_real else 'ATAQUE'})  reto izquierda: {res['izquierda']:20s} "
          f"reto derecha: {res['derecha']}")
