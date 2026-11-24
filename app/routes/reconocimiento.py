"""
Registro de asistencia comun al kiosco (POST /api/reconocer_rafaga, en rafaga.py).

Contiene el cooldown en memoria, los horarios de comida y _confirmar_asistencia.
"""
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from app.core.config import settings
from app.models.marcacion import Marcacion
from app.models.comensal import Comensal
from app.models.asistencia import Asistencia
from app.services.cache_service import EmbeddingCache
from app.services.validacion_service import validacion_service
from app.services.embedding_service import embedding_service
from datetime import datetime, time as dt_time
import time
import numpy as np

# comensal_id -> timestamp de la ultima asistencia registrada (lo consulta rafaga.py)
_cooldowns = {}

# ============================================================
# Cache de horarios de comida (evita query a BD en cada intento)
# ============================================================
_meal_config_cache = None     # dict con horarios parseados
_meal_config_ts = 0           # timestamp de ultima carga
MEAL_CACHE_TTL = 600          # 10 minutos de TTL


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
# Registro de asistencia
# ============================================================
async def _confirmar_asistencia(
    db: AsyncSession,
    cache: EmbeddingCache,
    comensal_id: int,
    avg_distance: float,
    liveness_score: float,
    embedding: np.ndarray,
    liveness_detalle: dict | None = None,
) -> dict:
    comensal = await db.get(Comensal, comensal_id)

    if not comensal:
        return {"success": False, "status": "ERROR", "message": "Comensal no encontrado en BD"}

    confianza_pct = (1 - avg_distance) * 100
    ahora = datetime.now()
    comida = await determinar_comida(ahora.time(), db)

    # Validar (comida ya consumida, confianza minima, etc.)
    validacion = await validacion_service.validar_marcacion(
        db, comensal_id, comida, confianza_pct, liveness_score
    )

    if not validacion["valido"]:
        await validacion_service.registrar_intento(
            db,
            comensal_id,
            "reconocimiento",
            "rechazado",
            validacion["mensaje"],
            confianza_pct,
            liveness_score,
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
        score_liveness=liveness_score,
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
        db, comensal_id, "reconocimiento", "exitoso", None, confianza_pct, liveness_score
    )

    await db.commit()

    # Cooldown solo cuando la asistencia quedo registrada: un rechazo
    # (fuera de horario, comida ya consumida) no debe bloquear 30 minutos.
    _cooldowns[comensal_id] = time.time()

    # --- 7. AUTO-ENRIQUECIMIENTO ---
    # Si la distancia fue alta (apariencia cambio: lentes, gorra, barba, etc.),
    # mezclar el embedding actual con el promedio para que el sistema "aprenda"
    # la nueva apariencia y no falle la proxima vez.
    if avg_distance > settings.ENRICHMENT_THRESHOLD:
        updated_emb = cache.enrich_embedding(comensal_id, embedding, weight=0.15)
        if updated_emb is not None:
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
        "liveness_score": round(liveness_score, 2),
        "liveness": liveness_detalle,
    }
