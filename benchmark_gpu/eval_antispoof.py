"""
Evalua el anti-spoofing con videos de webcam (UniqueData/web-camera-face-liveness-detection,
CC BY-NC-ND 4.0): 10 reales + 60 ataques (monitor, foto A4, foto recortada, mascara 2D,
silueta, mascara de silicona).

Compara: heuristica actual (liveness_service), modelo MiniFASNet (antispoof_service)
y ambos combinados (debe pasar los dos). Metricas ISO/IEC 30107-3:
  APCER = % de ataques aceptados como reales (menor = mas seguro)
  BPCER = % de personas reales rechazadas    (menor = mas comodo)

Uso (desde la raiz del proyecto):
  ..\\venv_gpu\\Scripts\\python.exe benchmark_gpu\\eval_antispoof.py
"""
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
DATASET = RAIZ.parent / "datasets" / "webcam_liveness" / "files"
SALIDA = Path(__file__).parent / "resultados"
CUADROS_POR_VIDEO = 12
VRAM_MIN_LIBRE_MB = 1200

sys.path.insert(0, str(RAIZ))
os.chdir(RAIZ)  # .env del proyecto

# Si Ollama u otro proceso ocupa la GPU, evaluar en CPU (mismo resultado, mas lento)
try:
    libre = int(subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True).stdout.split()[0])
except Exception:
    libre = 0
if libre < VRAM_MIN_LIBRE_MB:
    os.environ["USE_GPU"] = "false"
print(f"VRAM libre: {libre} MB -> {'GPU' if libre >= VRAM_MIN_LIBRE_MB else 'CPU'}")

from app.core.config import settings  # noqa: E402
from app.services.face_service import face_service  # noqa: E402
from app.services.liveness_service import liveness_service  # noqa: E402
from app.services.antispoof_service import AntiSpoofService  # noqa: E402

antispoof = AntiSpoofService()


def cuadros(video: Path, n: int):
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for idx in np.linspace(0, max(total - 1, 0), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        s = 640 / max(h, w)  # la camara del comedor es 640x480
        if s < 1:
            frame = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        yield int(idx), frame
    cap.release()


def tasa(xs):
    return round(100 * sum(xs) / len(xs), 1) if xs else float("nan")


def auc(reales, ataques):
    """Probabilidad de que un real puntue mas alto que un ataque (AUC ROC)."""
    r, a = np.array(reales), np.array(ataques)
    return round(float(((r[:, None] > a[None]).mean() + 0.5 * (r[:, None] == a[None]).mean())), 3)


def main():
    SALIDA.mkdir(exist_ok=True)
    filas = []
    for categoria in sorted(p.name for p in DATASET.iterdir()):
        for video in sorted((DATASET / categoria).glob("*.mp4")):
            sin_cara = 0
            for idx, frame in cuadros(video, CUADROS_POR_VIDEO):
                det = face_service.detect_face(frame)
                if det is None:
                    sin_cara += 1
                    continue
                heur = liveness_service.analyze(frame, det["bbox"])
                mod = antispoof.analyze(frame, det["bbox"])
                filas.append({
                    "categoria": categoria, "video": video.name, "cuadro": idx,
                    "real": categoria == "real",
                    "heuristica": heur["score"], "modelo": mod["prob_real"],
                })
            print(f"  {categoria:10s} {video.name:7s} sin cara: {sin_cara}/{CUADROS_POR_VIDEO}")

    with open(SALIDA / "antispoof_cuadros.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=filas[0].keys())
        w.writeheader()
        w.writerows(filas)

    th_h, th_m = settings.LIVENESS_THRESHOLD, settings.ANTISPOOF_THRESHOLD
    metodos = {
        "Heuristica actual": lambda r: r["heuristica"] >= th_h,
        "MiniFASNet": lambda r: r["modelo"] >= th_m,
        "Ambos (AND)": lambda r: r["heuristica"] >= th_h and r["modelo"] >= th_m,
    }

    # Por video: aceptado si la mayoria de sus cuadros pasa (como la votacion 3 de 5)
    por_video = defaultdict(list)
    for r in filas:
        por_video[(r["categoria"], r["video"])].append(r)

    resumen = {"umbral_heuristica": th_h, "umbral_modelo": th_m, "cuadros": len(filas), "metodos": {}}
    cats = sorted({r["categoria"] for r in filas})
    print(f"\nUmbrales: heuristica >= {th_h}, modelo >= {th_m}.  % ACEPTADO como real (por cuadro | por video)")
    print(f"{'categoria':12s}" + "".join(f"{m:>26s}" for m in metodos))
    for c in cats:
        linea = f"{c:12s}"
        for m, f in metodos.items():
            fc = [f(r) for r in filas if r["categoria"] == c]
            fv = [np.mean([f(r) for r in rs]) > 0.5 for (cc, _), rs in por_video.items() if cc == c]
            linea += f"{tasa(fc):>16.1f} | {tasa(fv):5.1f}  "
        print(linea)

    print()
    for m, f in metodos.items():
        reales = [f(r) for r in filas if r["real"]]
        ataques = [f(r) for r in filas if not r["real"]]
        v_reales = [np.mean([f(r) for r in rs]) > 0.5 for (c, _), rs in por_video.items() if c == "real"]
        v_ataques = [np.mean([f(r) for r in rs]) > 0.5 for (c, _), rs in por_video.items() if c != "real"]
        resumen["metodos"][m] = {
            "APCER_cuadro": tasa(ataques), "BPCER_cuadro": round(100 - tasa(reales), 1),
            "APCER_video": tasa(v_ataques), "BPCER_video": round(100 - tasa(v_reales), 1),
        }
        print(f"{m:18s} APCER {tasa(ataques):5.1f}%  BPCER {100 - tasa(reales):5.1f}%   "
              f"(por video: APCER {tasa(v_ataques):5.1f}%  BPCER {100 - tasa(v_reales):5.1f}%)")

    for col in ("heuristica", "modelo"):
        resumen[f"AUC_{col}"] = auc([r[col] for r in filas if r["real"]],
                                    [r[col] for r in filas if not r["real"]])
    print(f"\nAUC (independiente del umbral): heuristica {resumen['AUC_heuristica']}  "
          f"modelo {resumen['AUC_modelo']}")

    print("\nBarrido de umbral del modelo (por cuadro):")
    resumen["barrido_modelo"] = []
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        ap = tasa([r["modelo"] >= t for r in filas if not r["real"]])
        bp = round(100 - tasa([r["modelo"] >= t for r in filas if r["real"]]), 1)
        resumen["barrido_modelo"].append({"umbral": t, "APCER": ap, "BPCER": bp})
        print(f"  {t:.1f}: APCER {ap:5.1f}%  BPCER {bp:5.1f}%")

    (SALIDA / "antispoof_resumen.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado en {SALIDA}")


if __name__ == "__main__":
    main()
