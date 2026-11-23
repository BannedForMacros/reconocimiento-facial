"""
Reconocimiento por RAFAGA: liveness multi-fotograma (MiniFASNet + profundidad
+ luz) y voto de identidad dentro de una sola peticion (~1 s de captura).
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.core.database import get_db
from app.routes.reconocimiento import _confirmar_asistencia, _cooldowns
from app.services.cache_service import EmbeddingCache
from app.services.face_service import face_service
from app.services.liveness_service import liveness_service
from app.services.multiframe_liveness import (
    PALETA, analizar_fondo, analizar_luz, analizar_profundidad,
    comparar_con_referencia, gestor_desafios, referencia_fondo,
)
from app.services.validacion_service import validacion_service

router = APIRouter(prefix="/api", tags=["rafaga"])

LOG_SENALES = Path("data") / "liveness_log.jsonl"

MENSAJES = {
    "NO_FACE": "No se detecto rostro",
    "LOW_QUALITY": "Rostro de baja calidad, mejora la iluminacion",
    "TOO_FAR": "Acercate mas a la camara",
    "LOW_LIGHT": "Falta iluminacion, acercate a la luz",
}


@router.get("/liveness/config")
async def liveness_config():
    """Parametros que el kiosco necesita para capturar la rafaga."""
    return {
        "modo": settings.LIVENESS_MODO,
        "fotogramas": settings.RAFAGA_FOTOGRAMAS,
        "intervalo_ms": settings.RAFAGA_INTERVALO_MS,
        "ms_por_color": settings.LUZ_MS_POR_COLOR,
    }


@router.post("/liveness/desafio")
async def crear_desafio():
    """Secuencia de colores aleatoria y de un solo uso (anti-video grabado)."""
    d = gestor_desafios.crear()
    return {
        "id": d.id,
        "colores": [PALETA[c] for c in d.secuencia],
        "ms_por_color": settings.LUZ_MS_POR_COLOR,
    }


def decodificar(blobs: list[bytes]) -> list[np.ndarray]:
    out = []
    for b in blobs:
        im = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
        if im is not None:
            out.append(im)
    return out


def analizar_rafaga(images: list[np.ndarray], colores: Optional[list[str]] = None,
                    referencia: Optional[np.ndarray] = None) -> dict:
    """
    Analisis completo de una rafaga (CPU; se ejecuta fuera del event loop).
    """
    validos, fallos = [], {"NO_FACE": 0, "LOW_QUALITY": 0, "TOO_FAR": 0}
    for i, im in enumerate(images):
        det = face_service.detect_face(im)
        if det is None:
            fallos["NO_FACE"] += 1
            continue
        if det["det_score"] < settings.DETECTION_CONFIDENCE:
            fallos["LOW_QUALITY"] += 1
            continue
        if det["face_width"] < settings.MIN_FACE_SIZE or det["face_height"] < settings.MIN_FACE_SIZE:
            fallos["TOO_FAR"] += 1
            continue
        validos.append((i, im, det))

    if len(validos) < settings.RAFAGA_MIN_VALIDOS:
        peor = max(fallos, key=fallos.get) if any(fallos.values()) else "NO_FACE"
        return {"estado": peor, "validos": len(validos)}

    # --- Capa 1: MiniFASNet en cada fotograma ---
    fas = [liveness_service.analyze(im, det["bbox"]) for _, im, det in validos]
    if sum(f["detail"] == "low_light" for f in fas) > len(fas) / 2:
        return {"estado": "LOW_LIGHT", "validos": len(validos)}
    fas_scores = [f["score"] for f in fas if f["detail"] == "ok"]
    fas_score = float(np.mean(fas_scores)) if fas_scores else 0.0

    # --- Capa 2: profundidad (paralaje de los 106 puntos) ---
    prof = analizar_profundidad([det["landmarks_106"] for _, _, det in validos])

    # --- Capa 3: fondo quieto (camara fija: nada alrededor de la cabeza se mueve en bloque) ---
    frames_v = [im for _, im, _ in validos]
    bboxes_v = [det["bbox"] for _, _, det in validos]
    fondo = analizar_fondo(frames_v, bboxes_v)

    # --- Capa 4: fondo de referencia (alrededor se ve el fondo memorizado del kiosco) ---
    ref = comparar_con_referencia(frames_v, bboxes_v, referencia)

    # --- Capa 5: luz (solo si hubo desafio de colores) ---
    luz = None
    if colores:
        luz = analizar_luz(
            [im for _, im, _ in validos],
            [det["bbox"] for _, _, det in validos],
            [colores[i] for i, _, _ in validos],
        )

    # --- Decision ---
    fas_ok = fas_score >= settings.LIVENESS_THRESHOLD
    prof_ok = prof["estado"] == "real" or (
        prof["estado"] == "inconcluso" and not settings.PROFUNDIDAD_INCONCLUSO_RECHAZA
    )
    luz_ok = luz is None or luz["estado"] == "real"
    fondo_ok = fondo["estado"] != "se_mueve" or not settings.FONDO_LAMINA_DECIDE
    ref_ok = ref["estado"] != "no_coincide"          # sin referencia aun: no bloquea
    capas_ok = prof_ok and fondo_ok and ref_ok and luz_ok
    es_real = fas_ok and (capas_ok or not settings.LIVENESS_CAPAS_ACTIVAS)

    return {
        "estado": "OK",
        "validos": len(validos),
        "detecciones": validos,
        "es_real": es_real,
        "senales": {
            "minifasnet": round(fas_score, 3),
            "minifasnet_ok": fas_ok,
            "profundidad": prof,
            "fondo": fondo,
            "referencia": ref,
            "luz": luz,
            "capas_activas": settings.LIVENESS_CAPAS_ACTIVAS,
            "capas_ok": capas_ok,
        },
    }


def registrar_senales(modo: str, senales: dict, resultado: str):
    """Registro para calibrar en produccion (modo sombra). No guarda imagenes."""
    try:
        LOG_SENALES.parent.mkdir(parents=True, exist_ok=True)
        with LOG_SENALES.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": datetime.now().isoformat(timespec="seconds"),
                                "modo": modo, "resultado": resultado, **senales}) + "\n")
    except Exception as e:  # el registro nunca debe tumbar el reconocimiento
        logger.warning(f"No se pudo registrar senales de liveness: {e}")


def colores_de_desafio(desafio_id: Optional[str], indices: Optional[str], n: int):
    """Traduce los indices que envia el kiosco a los colores guardados en el servidor."""
    if not desafio_id:
        return None, None
    d = gestor_desafios.consumir(desafio_id)
    if d is None:
        return None, "desafio_invalido"
    try:
        cols = [d.secuencia[int(i)] for i in json.loads(indices or "[]")]
    except (ValueError, IndexError, TypeError):
        return None, "indices_invalidos"
    if len(cols) != n:
        return None, "indices_invalidos"
    return cols, None


@router.post("/reconocer_rafaga")
async def reconocer_rafaga(
    request: Request,
    fotos: List[UploadFile] = File(...),
    desafio_id: Optional[str] = Form(None),
    indices: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
):
    cache = EmbeddingCache.get_instance()
    images = decodificar([await f.read() for f in fotos])
    if len(images) < settings.RAFAGA_MIN_VALIDOS:
        return {"success": False, "status": "ERROR", "message": "Rafaga incompleta"}

    colores, err = colores_de_desafio(desafio_id, indices, len(images))
    if err:
        return {"success": False, "status": "ERROR", "message": "Desafio de luz invalido o vencido"}
    if settings.LIVENESS_MODO == "luz" and settings.LIVENESS_CAPAS_ACTIVAS and colores is None:
        return {"success": False, "status": "ERROR", "message": "Falta el desafio de luz"}

    kiosco = request.client.host if request.client else "default"
    r = await run_in_threadpool(analizar_rafaga, images, colores, referencia_fondo.obtener(kiosco))
    modo = "luz" if colores else "profundidad"

    if r["estado"] != "OK":
        return {"success": False, "status": r["estado"], "message": MENSAJES.get(r["estado"], "")}

    senales = r["senales"]
    if not r["es_real"]:
        registrar_senales(modo, senales, "LIVENESS_FAILED")
        await validacion_service.registrar_intento(
            db, None, "reconocimiento", "rechazado", "spoofing_detectado",
            score_liveness=senales["minifasnet"], es_suplantacion=True,
        )
        return {"success": False, "status": "LIVENESS_FAILED",
                "message": "Intento de suplantacion detectado", "liveness": senales}

    # --- Identidad: mayoria de los fotogramas de la rafaga (reemplaza los 3 votos) ---
    votos = {}
    for _, _, det in r["detecciones"]:
        m = cache.find_match(det["embedding"], settings.MATCH_THRESHOLD)
        if m:
            votos.setdefault(m["comensal_id"], []).append((m["distance"], det["embedding"]))
    necesarios = int(np.ceil(r["validos"] * 0.6))
    if not votos or max(len(v) for v in votos.values()) < necesarios:
        registrar_senales(modo, senales, "NO_MATCH")
        return {"success": False, "status": "NO_MATCH", "message": "Rostro no reconocido"}

    cid = max(votos, key=lambda k: len(votos[k]))
    dist = float(np.mean([d for d, _ in votos[cid]]))
    mejor_emb = min(votos[cid], key=lambda x: x[0])[1]

    # Cooldown compartido con /reconocer
    now = time.time()
    cooldown_secs = settings.ATTENDANCE_COOLDOWN_MINUTES * 60
    if cid in _cooldowns and now - _cooldowns[cid] < cooldown_secs:
        rest = int(cooldown_secs - (now - _cooldowns[cid]))
        info = cache._data.get(cid, {})
        registrar_senales(modo, senales, "COOLDOWN")
        return {"success": False, "status": "COOLDOWN",
                "message": f"Ya registraste asistencia. Espera {rest // 60}m {rest % 60}s.",
                "comensal": {"nombre": info.get("nombre"), "apellido": info.get("apellido")}}
    _cooldowns[cid] = now

    registrar_senales(modo, senales, "MATCH")
    return await _confirmar_asistencia(db, cache, cid, dist, senales["minifasnet"], mejor_emb, senales)
