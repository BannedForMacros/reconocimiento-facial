"""
Seleccion de proveedor ONNX Runtime: GPU (CUDA) si esta disponible, si no CPU.

- Requiere `onnxruntime-gpu` (ver requirements-gpu.txt). Con `onnxruntime` normal
  todo sigue funcionando en CPU, sin cambios.
- Limita la memoria de la arena CUDA: la tarjeta se comparte con otros procesos
  (p. ej. modelos de lenguaje) y ORT por defecto reserva todo lo que puede.
"""
import os
from pathlib import Path

import onnxruntime as ort
from loguru import logger

from app.core.config import settings

_cuda_lista = None


def _activar_cuda() -> bool:
    """En Windows, CUDA/cuDNN instalados por pip (nvidia-*-cu12) no estan en el PATH
    y cuDNN carga sus sub-DLL por nombre: hay que agregarlos antes de crear sesiones."""
    global _cuda_lista
    if _cuda_lista is not None:
        return _cuda_lista
    _cuda_lista = False
    if not settings.USE_GPU or "CUDAExecutionProvider" not in ort.get_available_providers():
        return False
    try:
        import nvidia  # paquetes nvidia-*-cu12
        bins = [str(p) for d in nvidia.__path__ for p in Path(d).glob("*/bin")]
        if os.name == "nt":
            os.environ["PATH"] = os.pathsep.join(bins) + os.pathsep + os.environ["PATH"]
            for b in bins:
                os.add_dll_directory(b)
    except ImportError:
        pass  # CUDA del sistema (p. ej. Linux con toolkit instalado)
    try:
        ort.preload_dlls()
    except Exception as e:  # versiones antiguas de ORT no lo tienen
        logger.debug(f"preload_dlls: {e}")
    _cuda_lista = True
    return True


def providers() -> list:
    if not _activar_cuda():
        return ["CPUExecutionProvider"]
    return [
        ("CUDAExecutionProvider", {
            "device_id": 0,
            "gpu_mem_limit": settings.GPU_MEM_LIMIT_MB * 1024 * 1024,
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_algo_search": "HEURISTIC",
            "cudnn_conv_use_max_workspace": "0",
        }),
        "CPUExecutionProvider",
    ]


def usa_gpu(session: ort.InferenceSession) -> bool:
    """ORT cae a CPU en silencio si CUDA falla: comprobar la sesion real."""
    return session.get_providers()[0] == "CUDAExecutionProvider"
