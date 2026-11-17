"""
Analiza una sesion de sesion_camara.py: heuristica vs modelo vs giro (prueba activa).

Uso:  ..\\venv_gpu\\Scripts\\python.exe benchmark_gpu\\analizar_sesion.py [carpeta_sesion]
(sin argumento usa la sesion mas reciente)
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from app.core.config import settings  # noqa: E402

carpeta = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(
    (RAIZ / "benchmark_gpu" / "resultados").glob("sesion_*"))[-1]
filas = list(csv.DictReader(open(carpeta / "cuadros.csv", encoding="utf-8")))
for r in filas:
    for k in ("heuristica", "modelo", "giro", "ms", "t"):
        r[k] = float(r[k])
    r["real"] = r["real"] == "True"

th_h, th_m = settings.LIVENESS_THRESHOLD, settings.ANTISPOOF_THRESHOLD
print(f"Sesion: {carpeta.name}  ({len(filas)} cuadros)   umbrales: heuristica {th_h}, modelo {th_m}\n")
print(f"{'fase':16s}{'n':>4s}{'heur. acepta':>14s}{'modelo acepta':>15s}{'giro min':>10s}{'giro max':>10s}"
      f"{'rango giro':>12s}{'ms':>7s}")
resumen = {}
for fase in dict.fromkeys(r["fase"] for r in filas):
    fs = [r for r in filas if r["fase"] == fase]
    g = np.array([r["giro"] for r in fs])
    h = 100 * np.mean([r["heuristica"] >= th_h for r in fs])
    m = 100 * np.mean([r["modelo"] >= th_m for r in fs])
    # rango robusto (percentiles 5-95) para no depender de un cuadro raro
    p5, p95 = np.percentile(g, 5), np.percentile(g, 95)
    resumen[fase] = {"n": len(fs), "heur_acepta": round(h, 1), "modelo_acepta": round(m, 1),
                     "giro_p5": round(p5, 3), "giro_p95": round(p95, 3),
                     "ms": round(float(np.mean([r["ms"] for r in fs])), 1)}
    print(f"{fase:16s}{len(fs):4d}{h:13.1f}%{m:14.1f}%{p5:10.2f}{p95:10.2f}{p95 - p5:12.2f}"
          f"{resumen[fase]['ms']:7.1f}")

(carpeta / "resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
