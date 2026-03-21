from fastapi import APIRouter, UploadFile, File, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from app.core.database import get_db
from app.core.config import settings
from app.models.marcacion import Marcacion
from app.models.comensal import Comensal
from app.models.asistencia import Asistencia
from app.services.face_service import face_service
from app.services.liveness_service import liveness_service
from app.services.cache_service import EmbeddingCache
from app.services.validacion_service import validacion_service
from app.services.embedding_service import embedding_service
from datetime import datetime, time as dt_time
from collections import deque
import time
import cv2
import numpy as np

router = APIRouter(prefix="/api", tags=["reconocimiento"])

# ============================================================
# Sistema de Votacion - estado en memoria
# ============================================================
_vote_buffers = {}   # session_key -> deque of {uid, dist, ts}
_cooldowns = {}      # comensal_id -> timestamp de ultima confirmacion
VOTE_MAX_AGE = 10    # votos mas viejos que esto (segundos) se descartan
_last_cleanup = 0    # timestamp del ultimo cleanup global
CLEANUP_INTERVAL = 300  # limpiar buffers inactivos cada 5 minutos

# ============================================================
# Cache de horarios de comida (evita query a BD en cada frame)
# ============================================================
_meal_config_cache = None     # dict con horarios parseados
_meal_config_ts = 0           # timestamp de ultima carga
MEAL_CACHE_TTL = 600          # 10 minutos de TTL


def _get_session_key(request: Request) -> str:
    return request.client.host if request.client else "default"


def _cleanup_stale():
    """Limpia buffers y cooldowns inactivos para evitar memory leaks."""
    global _last_cleanup
    now = time.time()
    if (now - _last_cleanup) < CLEANUP_INTERVAL:
        return
    _last_cleanup = now

    # Limpiar buffers de sesiones inactivas (sin votos recientes)
    stale_sessions = [
        s for s, buf in _vote_buffers.items()
        if not buf or (now - buf[-1]["ts"]) > VOTE_MAX_AGE * 6
    ]
    for s in stale_sessions:
        del _vote_buffers[s]

    # Limpiar cooldowns expirados
    cooldown_secs = settings.ATTENDANCE_COOLDOWN_MINUTES * 60
    expired = [uid for uid, ts in _cooldowns.items() if (now - ts) > cooldown_secs]
    for uid in expired:
        del _cooldowns[uid]


def _add_vote(session: str, comensal_id, distance: float) -> dict:
    """Agrega un voto al buffer y evalua si se alcanzo el threshold."""
    now = time.time()
    _cleanup_stale()

    if session not in _vote_buffers:
        _vote_buffers[session] = deque(maxlen=settings.VOTING_WINDOW)

    buffer = _vote_buffers[session]

    # Descartar votos viejos
    while buffer and (now - buffer[0]["ts"]) > VOTE_MAX_AGE:
        buffer.popleft()

    buffer.append({"uid": comensal_id, "dist": distance, "ts": now})

    if comensal_id is None:
        return {"confirmed": False, "status": "NO_MATCH"}

    # Contar votos por usuario
    votes = {}
    for v in buffer:
        if v["uid"] is not None:
            votes[v["uid"]] = votes.get(v["uid"], 0) + 1

    if not votes:
        return {"confirmed": False, "status": "NO_MATCH"}

    best_uid = max(votes, key=votes.get)
    best_count = votes[best_uid]

    if best_count >= settings.VOTING_THRESHOLD:
        # Verificar cooldown
        cooldown_secs = settings.ATTENDANCE_COOLDOWN_MINUTES * 60
        if best_uid in _cooldowns:
            elapsed = now - _cooldowns[best_uid]
            if elapsed < cooldown_secs:
                remaining = int(cooldown_secs - elapsed)
                return {
                    "confirmed": False,
                    "status": "COOLDOWN",
                    "comensal_id": best_uid,
                    "remaining": remaining,
                }

        # Calcular distancia promedio de los votos confirmados
        matching = [v["dist"] for v in buffer if v["uid"] == best_uid]
        avg_dist = float(np.mean(matching)) if matching else distance

        # Confirmar y registrar cooldown
        _cooldowns[best_uid] = now
        buffer.clear()

        return {
            "confirmed": True,
            "status": "MATCH_CONFIRMED",
            "comensal_id": best_uid,
            "votes": best_count,
            "avg_distance": avg_dist,
        }

    return {
        "confirmed": False,
        "status": "SCANNING",
        "progress": f"{best_count}/{settings.VOTING_THRESHOLD}",
        "comensal_id": best_uid,
    }


# ============================================================
# Determinar comida segun horario (con cache)
# ============================================================
async def _load_meal_config(db: AsyncSession):
    """Carga horarios de comida desde BD y los cachea."""
    global _meal_config_cache, _meal_config_ts
    from app.models.configuracion import Configuracion

    config = {}
    result = await db.execute(select(Configuracion))
    for conf in result.scalars():
        config[conf.clave] = conf.valor

    def parse_time(t_str):
        return datetime.strptime(t_str, "%H:%M").time()

    parsed = {}
    try:
        for meal in ("desayuno", "almuerzo", "cena"):
            parsed[meal] = (
                parse_time(config[f"horario_{meal}_inicio"]),
                parse_time(config[f"horario_{meal}_fin"]),
            )
    except (KeyError, ValueError):
        parsed = {}

    _meal_config_cache = parsed
    _meal_config_ts = time.time()


async def determinar_comida(hora: dt_time, db: AsyncSession):
    global _meal_config_cache, _meal_config_ts

    # Recargar si el cache expiro o no existe
    if _meal_config_cache is None or (time.time() - _meal_config_ts) > MEAL_CACHE_TTL:
        await _load_meal_config(db)

    if not _meal_config_cache:
        return None

    for meal, (inicio, fin) in _meal_config_cache.items():
        if inicio <= hora <= fin:
            return meal
    return None


# ============================================================
# Endpoint principal de reconocimiento
# ============================================================
@router.post("/reconocer")
async def reconocer_rostro(
    request: Request,
    foto: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    session_key = _get_session_key(request)
    cache = EmbeddingCache.get_instance()

    # --- Leer imagen ---
    contents = await foto.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return {"success": False, "status": "ERROR", "message": "Imagen invalida"}

    # --- 1. Detectar rostro ---
    detection = face_service.detect_face(image)
    if detection is None:
        return {"success": False, "status": "NO_FACE", "message": "No se detecto rostro"}

    # --- 2. Validar calidad ---
    if detection["det_score"] < settings.DETECTION_CONFIDENCE:
        return {
            "success": False,
            "status": "LOW_QUALITY",
            "message": "Rostro de baja calidad, mejora la iluminacion",
        }

    if (
        detection["face_width"] < settings.MIN_FACE_SIZE
        or detection["face_height"] < settings.MIN_FACE_SIZE
    ):
        return {
            "success": False,
            "status": "TOO_FAR",
            "message": "Acercate mas a la camara",
        }

    # --- 3. LIVENESS primero (si falla, no se intenta reconocimiento) ---
    face_crop = face_service.get_face_crop(image, detection["bbox"])
    liveness = liveness_service.analyze(face_crop)

    if not liveness["is_live"]:
        await validacion_service.registrar_intento(
            db,
            None,
            "reconocimiento",
            "rechazado",
            "spoofing_detectado",
            score_liveness=liveness["score"],
            es_suplantacion=True,
        )
        return {
            "success": False,
            "status": "LIVENESS_FAILED",
            "message": "Intento de suplantacion detectado",
            "liveness_score": liveness["score"],
        }

    # --- 4. Buscar match en cache (vectorizado) ---
    match = cache.find_match(detection["embedding"], settings.MATCH_THRESHOLD)

    if match is None:
        _add_vote(session_key, None, 1.0)
        return {"success": False, "status": "NO_MATCH", "message": "Rostro no reconocido"}

    # --- 5. Sistema de votacion ---
    vote = _add_vote(session_key, match["comensal_id"], match["distance"])

    if vote["status"] == "SCANNING":
        return {
            "success": False,
            "status": "SCANNING",
            "message": f"Verificando identidad... ({vote['progress']})",
            "partial": {
                "nombre": f"{match['nombre']} {match['apellido']}",
                "confianza": round((1 - match["distance"]) * 100, 1),
            },
        }

    if vote["status"] == "COOLDOWN":
        mins = vote["remaining"] // 60
        secs = vote["remaining"] % 60
        return {
            "success": False,
            "status": "COOLDOWN",
            "message": f"Ya registraste asistencia. Espera {mins}m {secs}s.",
            "comensal": {
                "nombre": match["nombre"],
                "apellido": match["apellido"],
            },
        }

    if vote["status"] == "NO_MATCH":
        return {"success": False, "status": "NO_MATCH", "message": "Rostro no reconocido"}

    # --- 6. MATCH CONFIRMADO → Registrar asistencia ---
    comensal_id = vote["comensal_id"]
    comensal = await db.get(Comensal, comensal_id)

    if not comensal:
        return {"success": False, "status": "ERROR", "message": "Comensal no encontrado en BD"}

    avg_distance = vote["avg_distance"]
    confianza_pct = (1 - avg_distance) * 100
    ahora = datetime.now()
    comida = await determinar_comida(ahora.time(), db)

    # Validar (comida ya consumida, confianza minima, etc.)
    validacion = await validacion_service.validar_marcacion(
        db, comensal_id, comida, confianza_pct, liveness["score"]
    )

    if not validacion["valido"]:
        await validacion_service.registrar_intento(
            db,
            comensal_id,
            "reconocimiento",
            "rechazado",
            validacion["mensaje"],
            confianza_pct,
            liveness["score"],
        )
        return {
            "success": False,
            "status": "REJECTED",
            "message": validacion["mensaje"],
        }

    # Crear marcacion
    marcacion = Marcacion(
        comensal_id=comensal_id,
        tipo="entrada",
        confianza=confianza_pct / 100,
        score_liveness=liveness["score"],
        es_valida=True,
    )
    db.add(marcacion)

    # Crear/actualizar asistencia del dia
    asistencia_q = await db.execute(
        select(Asistencia).where(
            and_(
                Asistencia.comensal_id == comensal_id,
                Asistencia.fecha == ahora.date(),
            )
        )
    )
    asistencia = asistencia_q.scalar_one_or_none()

    if not asistencia:
        asistencia = Asistencia(
            comensal_id=comensal_id,
            fecha=ahora.date(),
            hora_entrada=ahora.time(),
        )
        db.add(asistencia)

    if comida == "desayuno":
        asistencia.desayuno = True
    elif comida == "almuerzo":
        asistencia.almuerzo = True
    elif comida == "cena":
        asistencia.cena = True

    comensal.total_marcaciones = (comensal.total_marcaciones or 0) + 1
    comensal.ultima_marcacion = ahora

    # Registrar intento exitoso
    await validacion_service.registrar_intento(
        db, comensal_id, "reconocimiento", "exitoso", None, confianza_pct, liveness["score"]
    )

    await db.commit()

    # --- 7. AUTO-ENRIQUECIMIENTO ---
    # Si la distancia fue alta (apariencia cambio: lentes, gorra, barba, etc.),
    # mezclar el embedding actual con el promedio para que el sistema "aprenda"
    # la nueva apariencia y no falle la proxima vez.
    if avg_distance > settings.ENRICHMENT_THRESHOLD:
        enriched = cache.enrich_embedding(comensal_id, detection["embedding"], weight=0.15)
        if enriched:
            # Persistir el embedding actualizado en BD (async, no bloqueante)
            updated_emb = cache._data[comensal_id]["embedding"]
            await embedding_service.update_average_embedding(db, comensal_id, updated_emb)

    return {
        "success": True,
        "status": "MATCH_CONFIRMED",
        "message": f"Bienvenido, {comensal.nombre} {comensal.apellido}",
        "comensal": {
            "dni": comensal.dni,
            "nombre": comensal.nombre,
            "apellido": comensal.apellido,
        },
        "confianza": round(confianza_pct, 2),
        "comida": comida or "fuera de horario",
        "liveness_score": round(liveness["score"], 2),
    }
