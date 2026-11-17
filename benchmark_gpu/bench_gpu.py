"""
Benchmark CPU vs GPU del pipeline de reconocimiento facial (InsightFace buffalo_l).

Mide lo mismo que hace /api/reconocer: detectar + embedding de la cara mas grande.
Compara 4 configuraciones:
  - CPU, 5 modelos  (config actual de face_service.py)
  - CPU, 2 modelos  (solo deteccion + reconocimiento: lo unico que usa el sistema)
  - GPU, 5 modelos
  - GPU, 2 modelos

Cuida la VRAM: no arranca en GPU si hay menos de VRAM_MIN_LIBRE_MB libres y limita
la arena de ONNX Runtime (otros procesos, como Ollama, comparten la tarjeta).

Uso:  venv_gpu\\Scripts\\python.exe benchmark_gpu\\bench_gpu.py [--iter 100]
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import cv2
import onnxruntime as ort

VRAM_MIN_LIBRE_MB = 1200
GPU_MEM_LIMIT = 1024 * 1024 * 1024  # 1 GB para la arena de ORT

AQUI = Path(__file__).parent
RESULTADOS = AQUI / "resultados"


def vram_libre_mb() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    ).stdout.strip()
    return int(out.splitlines()[0])


def vram_proceso_mb() -> float:
    """VRAM dedicada de ESTE proceso (contador de Windows; nvidia-smi da N/A en WDDM)."""
    ps = (
        f"(Get-Counter '\\GPU Process Memory(pid_{os.getpid()}_*)\\Dedicated Usage' "
        "-ErrorAction SilentlyContinue).CounterSamples | Measure-Object CookedValue -Sum | "
        "ForEach-Object { $_.Sum }"
    )
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                         capture_output=True, text=True).stdout.strip()
    try:
        return float(out.replace(",", ".")) / 1024 / 1024
    except ValueError:
        return float("nan")


def activar_cuda():
    """CUDA/cuDNN vienen por pip (nvidia-*-cu12). cuDNN carga sus sub-DLL por PATH
    (p. ej. cudnn_engines_tensor_ir64_9.dll), asi que preload_dlls() no basta."""
    import nvidia
    bins = [str(p) for d in nvidia.__path__ for p in Path(d).glob("*/bin")]
    os.environ["PATH"] = os.pathsep.join(bins) + os.pathsep + os.environ["PATH"]
    for b in bins:
        os.add_dll_directory(b)
    ort.preload_dlls()


def proveedores(gpu: bool):
    if not gpu:
        return ["CPUExecutionProvider"]
    return [
        ("CUDAExecutionProvider", {
            "device_id": 0,
            "gpu_mem_limit": GPU_MEM_LIMIT,
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_algo_search": "HEURISTIC",
            "cudnn_conv_use_max_workspace": "0",
        }),
        "CPUExecutionProvider",
    ]


def cargar(gpu: bool, solo_necesarios: bool):
    from insightface.app import FaceAnalysis
    kw = {}
    if solo_necesarios:
        kw["allowed_modules"] = ["detection", "recognition"]
    app = FaceAnalysis(name="buffalo_l", providers=proveedores(gpu), **kw)
    app.prepare(ctx_id=0 if gpu else -1, det_size=(640, 640))
    # Verificar que de verdad corre en GPU (si falla CUDA, ORT cae a CPU en silencio)
    usados = {m.session.get_providers()[0] for m in app.models.values()}
    return app, usados


def mayor(faces):
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def medir(app, img, iteraciones: int, calentamiento: int = 10):
    for _ in range(calentamiento):
        app.get(img)
    tiempos = []
    for _ in range(iteraciones):
        t0 = time.perf_counter()
        faces = app.get(img)
        tiempos.append((time.perf_counter() - t0) * 1000)
    tiempos.sort()
    return {
        "ms_media": round(statistics.mean(tiempos), 2),
        "ms_p50": round(tiempos[len(tiempos) // 2], 2),
        "ms_p95": round(tiempos[int(len(tiempos) * 0.95) - 1], 2),
        "fps": round(1000 / statistics.mean(tiempos), 1),
        "caras": len(faces),
    }, faces


def imagenes():
    from insightface.data import get_image
    grupo = get_image("t1")  # foto de grupo de ejemplo de InsightFace (6 caras)
    # Simula un cuadro de la webcam del comedor (640x480) con UNA persona cerca
    f = mayor(cargar_det_cpu().get(grupo))
    x1, y1, x2, y2 = [int(v) for v in f.bbox]
    cx, cy, lado = (x1 + x2) // 2, (y1 + y2) // 2, int((x2 - x1) * 2.6)
    y0, x0 = max(0, cy - lado // 2), max(0, cx - int(lado * 0.67))
    recorte = grupo[y0:y0 + lado, x0:x0 + int(lado * 1.33)]
    webcam = cv2.resize(recorte, (640, 480))
    return {"webcam_640x480_1cara": webcam, "grupo_t1": grupo}


_det = None
def cargar_det_cpu():
    global _det
    if _det is None:
        from insightface.app import FaceAnalysis
        _det = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"],
                            allowed_modules=["detection"])
        _det.prepare(ctx_id=-1, det_size=(640, 640))
    return _det


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", type=int, default=100)
    ap.add_argument("--solo-cpu", action="store_true")
    ap.add_argument("--solo-gpu", action="store_true")
    args = ap.parse_args()

    RESULTADOS.mkdir(exist_ok=True)
    imgs = imagenes()
    for n, im in imgs.items():
        cv2.imwrite(str(RESULTADOS / f"{n}.jpg"), im)

    configs = [] if args.solo_gpu else [("CPU", False, False), ("CPU", False, True)]
    if not args.solo_cpu:
        configs += [("GPU", True, False), ("GPU", True, True)]

    filas, embeddings_ref, vram = [], {}, {}
    for nombre, gpu, solo in configs:
        etiqueta = f"{nombre} {'2 modelos (det+rec)' if solo else '5 modelos (actual)'}"
        if gpu:
            activar_cuda()
            libre = vram_libre_mb()
            if libre < VRAM_MIN_LIBRE_MB:
                print(f"[ABORTO GPU] solo {libre} MB libres (< {VRAM_MIN_LIBRE_MB}); no compito con Ollama.")
                break
            print(f"VRAM libre antes: {libre} MB")
        app, usados = cargar(gpu, solo)
        print(f"\n== {etiqueta}  proveedores reales: {usados}")
        if gpu and usados != {"CUDAExecutionProvider"}:
            print("[ERROR] Algun modelo no corre en CUDA; resultado GPU invalido.")
        for nimg, img in imgs.items():
            r, faces = medir(app, img, args.iter)
            emb = mayor(faces).normed_embedding
            if nimg not in embeddings_ref:
                embeddings_ref[nimg] = emb
            r["sim_vs_cpu"] = round(float(np.dot(emb, embeddings_ref[nimg])), 5)
            r.update(config=etiqueta, imagen=nimg)
            filas.append(r)
            print(f"  {nimg:24s} {r['ms_media']:7.1f} ms  p95 {r['ms_p95']:7.1f}  "
                  f"{r['fps']:6.1f} FPS  caras={r['caras']}  sim={r['sim_vs_cpu']}")
        if gpu:
            vram[etiqueta] = round(vram_proceso_mb(), 0)
            print(f"  VRAM de este proceso: {vram[etiqueta]:.0f} MB")
        del app

    info = {
        "fecha": time.strftime("%Y-%m-%d %H:%M"),
        "cpu": "Intel i7-9700 (8 nucleos)", "gpu": "RTX 5060 Ti 16 GB",
        "onnxruntime": ort.__version__, "iteraciones": args.iter,
        "vram_mb": vram, "filas": filas,
    }
    (RESULTADOS / "resultados.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado en {RESULTADOS / 'resultados.json'}")


if __name__ == "__main__":
    sys.exit(main())
