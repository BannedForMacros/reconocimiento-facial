"""
Sesion guiada con la camara real para medir el anti-spoofing (real vs celular).

Abre una ventana con instrucciones; ESPACIO empieza cada fase, S la salta, Q sale.
Por cada cuadro guarda: score de la heuristica actual, prob. del modelo MiniFASNet y
el "giro" de la cabeza calculado con los 5 puntos de SCRFD (nariz respecto al centro
de los ojos; una foto plana no puede cambiarlo aunque se gire el celular).

Uso:  ..\\venv_gpu\\Scripts\\python.exe benchmark_gpu\\sesion_camara.py [--camara 1]
Salida: benchmark_gpu/resultados/sesion_<fecha>/ (cuadros.csv + algunas imagenes)
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
os.chdir(RAIZ)

from app.core.config import settings  # noqa: E402
from app.services.face_service import face_service  # noqa: E402
from app.services.liveness_service import liveness_service  # noqa: E402
from app.services.antispoof_service import AntiSpoofService  # noqa: E402

FASES = [
    ("real_frontal", True, 10, "Mira a la camara. Muevete natural, habla, parpadea."),
    ("real_giro", True, 12, "Gira la cabeza a la IZQUIERDA, luego a la DERECHA, despacio. Repite."),
    ("real_lejos", True, 8, "Alejate un poco y vuelve a acercarte."),
    ("celular_foto", False, 12, "Pon una FOTO TUYA en el celular frente a la camara (pantalla completa)."),
    ("celular_girado", False, 12, "Con la foto: gira el celular a izquierda y derecha. Intenta enganar."),
    ("celular_video", False, 12, "(Opcional) Reproduce un VIDEO tuyo en el celular. S = saltar."),
]


def giro(kps: np.ndarray) -> float:
    """(x nariz - x centro de ojos) / distancia entre ojos. ~0 de frente, +-0.3..0.6 girado."""
    ojo_i, ojo_d, nariz = kps[0], kps[1], kps[2]
    dist = np.linalg.norm(ojo_d - ojo_i) + 1e-6
    return float((nariz[0] - (ojo_i[0] + ojo_d[0]) / 2) / dist)


def texto(img, t, y, escala=0.7, color=(255, 255, 255)):
    cv2.putText(img, t, (12, y), cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, t, (12, y), cv2.FONT_HERSHEY_SIMPLEX, escala, color, 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camara", type=int, default=1)
    args = ap.parse_args()

    antispoof = AntiSpoofService()
    cap = cv2.VideoCapture(args.camara, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        sys.exit(f"No se pudo abrir la camara {args.camara}")

    salida = RAIZ / "benchmark_gpu" / "resultados" / time.strftime("sesion_%Y%m%d_%H%M")
    salida.mkdir(parents=True, exist_ok=True)
    filas, ventana = [], "Sesion anti-spoofing (ESPACIO empezar, S saltar, Q salir)"
    cv2.namedWindow(ventana, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(ventana, 960, 720)
    cv2.setWindowProperty(ventana, cv2.WND_PROP_TOPMOST, 1)

    for nombre, es_real, segundos, instruccion in FASES:
        estado, inicio, guardadas = "espera", 0.0, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            vista = frame.copy()
            t0 = time.perf_counter()
            det = face_service.detect_face(frame)
            info = None
            if det is not None:
                heur = liveness_service.analyze(frame, det["bbox"])
                mod = antispoof.analyze(frame, det["bbox"])
                g = giro(det["face"].kps)
                ms = (time.perf_counter() - t0) * 1000
                info = (heur["score"], mod["prob_real"], g, ms)
                x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
                cv2.rectangle(vista, (x1, y1), (x2, y2), (0, 255, 0), 2)
                for p in det["face"].kps.astype(int):
                    cv2.circle(vista, tuple(p), 3, (0, 255, 255), -1)

            texto(vista, f"FASE: {nombre}", 30, 0.8, (0, 255, 255))
            texto(vista, instruccion, 60, 0.55)
            if info:
                texto(vista, f"heuristica {info[0]:.2f} (>= {settings.LIVENESS_THRESHOLD})  "
                             f"modelo {info[1]:.2f}  giro {info[2]:+.2f}  {info[3]:.0f} ms", 465, 0.55)
            else:
                texto(vista, "sin cara", 465, 0.55, (0, 0, 255))

            if estado == "espera":
                texto(vista, "Preparate y pulsa ESPACIO", 100, 0.8, (0, 200, 255))
            else:
                resta = segundos - (time.time() - inicio)
                texto(vista, f"GRABANDO {resta:4.1f} s", 100, 0.8, (0, 0, 255))
                if info:
                    filas.append({
                        "fase": nombre, "real": es_real, "t": round(time.time() - inicio, 2),
                        "heuristica": info[0], "modelo": info[1], "giro": round(info[2], 3),
                        "det_score": round(det["det_score"], 3),
                        "ancho_cara": round(det["face_width"]), "ms": round(info[3], 1),
                    })
                    if guardadas < 3 and (time.time() - inicio) > guardadas * segundos / 3:
                        cv2.imwrite(str(salida / f"{nombre}_{guardadas}.jpg"), frame)
                        guardadas += 1
                if resta <= 0:
                    break

            cv2.imshow(ventana, vista)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                estado = "salir"
                break
            if k == ord("s"):
                break
            if k == ord(" ") and estado == "espera":
                estado, inicio = "grabando", time.time()
        if estado == "salir":
            break

    cap.release()
    cv2.destroyAllWindows()
    if filas:
        with open(salida / "cuadros.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=filas[0].keys())
            w.writeheader()
            w.writerows(filas)
    print(f"{len(filas)} cuadros guardados en {salida}")


if __name__ == "__main__":
    main()
