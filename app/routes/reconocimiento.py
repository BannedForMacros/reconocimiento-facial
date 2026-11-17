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
from app.services.antispoof_service import antispoof_service
from app.services.reto_service import reto_service
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
# Deteccion de movimiento entre frames (anti-spoofing temporal)
# Una foto estatica en pantalla tiene variacion ~0 entre frames.
# Una persona real tiene micro-movimientos naturales (respiracion,
# temblor, parpadeo) que generan variacion de ~3-15 pixeles promedio.
# ============================================================
_face_history = {}          # session_key -> list of (gray_face_64x64, timestamp)
MOTION_HISTORY_SIZE = 4     # cuantos frames guardar
MOTION_MIN_DIFF = 1.8       # variacion minima para considerar "con movimiento"
MOTION_FRAMES_NEEDED = 3    # cuantos frames necesitamos antes de fallar por motion

def _check_motion(session: str, face_bgr: np.ndarray) -> dict:
    """
    Compara el frame actual con los anteriores para detectar movimiento.
    Retorna: {'score': float, 'frames': int, 'ready': bool}
    """
    gray = cv2.cvtColor(cv2.resize(face_bgr, (64, 64)), cv2.COLOR_BGR2GRAY)
    now = time.time()

    if session not in _face_history:
        _face_history[session] = []

    history = _face_history[session]

    # Descartar frames muy viejos (mas de 15s = nueva sesion)
    history[:] = [(f, t) for f, t in history if (now - t) < 15.0]
    history.append((gray, now))

    if len(history) > MOTION_HISTORY_SIZE:
        history.pop(0)

    if len(history) < 2:
        return {"score": 1.0, "frames": len(history), "ready": False}

    diffs = []
    for i in range(1, len(history)):
        diff = np.mean(np.abs(history[i][0].astype(np.float32) - history[i-1][0].astype(np.float32)))
        diffs.append(diff)

    avg_diff = float(np.mean(diffs))
    # Clamp: 0 diff -> 0.0, MOTION_MIN_DIFF -> empieza a subir, 8.0 -> 1.0
    motion_score = float(np.clip((avg_diff - 0.5) / 7.5, 0.0, 1.0))
    ready = len(history) >= MOTION_FRAMES_NEEDED

    return {"score": motion_score, "avg_diff": round(avg_diff, 2), "frames": len(history), "ready": ready}


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


def _add_vote(session: str, comensal_id, distance: float, permitir_confirmar: bool = True) -> dict:
    """Agrega un voto al buffer y evalua si se alcanzo el threshold.
    permitir_confirmar=False (reto activo sin cumplir): no confirma ni inicia cooldown."""
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

        if not permitir_confirmar:
            return {
                "confirmed": False,
                "status": "CHALLENGE",
                "progress": f"{best_count}/{settings.VOTING_THRESHOLD}",
                "comensal_id": best_uid,
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

    # --- 2. Validar tamano (la confianza del detector se valida despues del reto:
    #        al girar la cabeza baja de 0.80 y esos cuadros si sirven para el reto) ---
    if (
        detection["face_width"] < settings.MIN_FACE_SIZE
        or detection["face_height"] < settings.MIN_FACE_SIZE
    ):
        return {
            "success": False,
            "status": "TOO_FAR",
            "message": "Acercate mas a la camara",
        }

    # --- 3. LIVENESS ---
    liveness = liveness_service.analyze(image, detection["bbox"])

    # El motion check solo se usa para logging/debug, NO afecta la decision.
    # Anteriormente sumaba al combined_score, lo que permitia que un celular
    # en movimiento pasara el threshold aunque tuviera liveness muy bajo.
    face_crop = face_service.get_face_crop(image, detection["bbox"])
    motion = _check_motion(session_key, face_crop)

    # Con el modelo MiniFASNet activo, el decide (la heuristica queda solo en el log):
    # con la camara real rechazaba al usuario al girar (6 % aceptado) y aceptaba el
    # celular girado (95 %). Ver benchmark_gpu/RESULTADOS.md.
    modelo = antispoof_service.analyze(image, detection["bbox"]) if antispoof_service else None
    if modelo and modelo["prob_real"] is not None:
        combined_score = modelo["prob_real"]
        is_live = modelo["is_live"]
    else:
        combined_score = liveness["score"]
        is_live = combined_score >= settings.LIVENESS_THRESHOLD

    from loguru import logger as _log
    _log.debug(
        f"LIVENESS heuristica={liveness['score']:.3f} scores={liveness.get('scores',{})} "
        f"modelo={modelo['prob_real'] if modelo else None} "
        f"motion_ref={motion['score']:.2f} avg_diff={motion.get('avg_diff',0):.2f} "
        f"threshold={settings.LIVENESS_THRESHOLD}"
    )

    if not is_live:
        await validacion_service.registrar_intento(
            db,
            None,
            "reconocimiento",
            "rechazado",
            "spoofing_detectado",
            score_liveness=combined_score,
            es_suplantacion=True,
        )
        return {
            "success": False,
            "status": "LIVENESS_FAILED",
            "message": "Intento de suplantacion detectado",
            "liveness_score": round(combined_score, 3),
            "liveness_detail": {
                "modelo": modelo["prob_real"] if modelo else None,
                "texture_color": round(liveness["score"], 3),
                "motion": round(motion["score"], 3),
                "motion_diff": motion.get("avg_diff", 0),
                "frames_analyzed": motion["frames"],
            },
        }

    # --- 3b. Prueba de vida ACTIVA: solo avanza con cuadros que pasaron el modelo ---
    reto = reto_service.actualizar(session_key, detection["face"].kps) if settings.RETO_ENABLED else None
    reto_pendiente = reto is not None and not reto["cumplido"]

    # --- 3c. Calidad para identificar (no para el reto) ---
    if detection["det_score"] < settings.DETECTION_CONFIDENCE:
        if reto_pendiente:
            return {"success": False, "status": "SCANNING", "message": reto["instruccion"], "reto": reto}
        return {
            "success": False,
            "status": "LOW_QUALITY",
            "message": "Rostro de baja calidad, mejora la iluminacion",
        }

    # --- 4. Buscar match en cache (vectorizado) ---
    match = cache.find_match(detection["embedding"], settings.MATCH_THRESHOLD)

    if match is None:
        _add_vote(session_key, None, 1.0)
        if reto_pendiente and reto["fase"] == "girado":
            # De perfil el embedding se aleja: no cortar el reto a la mitad
            return {"success": False, "status": "SCANNING", "message": reto["instruccion"], "reto": reto}
        return {"success": False, "status": "NO_MATCH", "message": "Rostro no reconocido", "liveness_score": liveness["score"]}

    # --- 5. Sistema de votacion ---
    vote = _add_vote(session_key, match["comensal_id"], match["distance"],
                     permitir_confirmar=not reto_pendiente)

    if vote["status"] in ("SCANNING", "CHALLENGE"):
        # Mientras el reto no se cumpla, la instruccion reemplaza al progreso
        # (la pagina ya muestra data.message en el estado SCANNING)
        mensaje = reto["instruccion"] if reto_pendiente else f"Verificando identidad... ({vote['progress']})"
        return {
            "success": False,
            "status": "SCANNING",
            "message": mensaje,
            "reto": reto,
            "liveness_score": round(combined_score, 3),
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
    reto_service.consumir(session_key)
    comensal = await db.get(Comensal, comensal_id)

    if not comensal:
        return {"success": False, "status": "ERROR", "message": "Comensal no encontrado en BD"}

    avg_distance = vote["avg_distance"]
    confianza_pct = (1 - avg_distance) * 100
    ahora = datetime.now()
    comida = await determinar_comida(ahora.time(), db)

    # Validar (comida ya consumida, confianza minima, etc.)
    validacion = await validacion_service.validar_marcacion(
        db, comensal_id, comida, confianza_pct, combined_score
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
        score_liveness=combined_score,
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
        db, comensal_id, "reconocimiento", "exitoso", None, confianza_pct, combined_score
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
        "liveness_score": round(combined_score, 3),
    }
