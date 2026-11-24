from starlette.concurrency import run_in_threadpool
from fastapi import APIRouter, Request, Depends
from fastapi.responses import JSONResponse
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.core.database import get_db
from app.core.config import settings
from app.models.comensal import Comensal
from app.services.face_service import face_service
from app.services.liveness_service import liveness_service
from app.services.embedding_service import embedding_service
from app.services.cache_service import EmbeddingCache
from app.services.multiframe_liveness import referencia_fondo
import cv2
import base64
import numpy as np

router = APIRouter(prefix="/api", tags=["registro"])


@router.post("/validar_rostro")
async def validar_rostro(request: Request):
    """
    Endpoint de validacion en vivo. El frontend envia un frame y recibe
    feedback instantaneo sobre deteccion, calidad, tamano y liveness.
    Se usa ANTES de iniciar la captura para que el usuario se posicione bien.
    """
    try:
        body = await request.json()
        frame_b64 = body.get("frame", "")
    except Exception:
        return {"ok": False, "msg": "Datos invalidos"}

    try:
        data_part = frame_b64.split(",")[1] if "," in frame_b64 else frame_b64
        img_data = base64.b64decode(data_part)
        nparr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return {"ok": False, "msg": "Imagen invalida"}

        detection = await run_in_threadpool(face_service.detect_face, image)
        if detection is None:
            # Kiosco vacio: aprender/actualizar su fondo de referencia
            if body.get("kiosco"):
                kiosco = request.client.host if request.client else "default"
                referencia_fondo.actualizar(kiosco, image)
            return {"ok": False, "msg": "No se detecta rostro", "code": "NO_FACE"}

        det_score = detection["det_score"]
        fw = detection["face_width"]
        fh = detection["face_height"]

        if det_score < settings.DETECTION_CONFIDENCE:
            return {
                "ok": False,
                "msg": f"Confianza baja ({det_score:.2f}). Mejora iluminacion.",
                "code": "LOW_CONF",
                "det_score": round(det_score, 2),
            }

        if fw < settings.MIN_FACE_SIZE or fh < settings.MIN_FACE_SIZE:
            return {
                "ok": False,
                "msg": "Rostro muy pequeno. Acercate a la camara.",
                "code": "TOO_SMALL",
                "face_size": round(min(fw, fh)),
            }

        # Liveness check
        liveness = await run_in_threadpool(liveness_service.analyze, image, detection["bbox"])
        if liveness["detail"] == "low_light":
            return {
                "ok": False,
                "msg": "Falta iluminacion. Acercate a una fuente de luz.",
                "code": "LOW_LIGHT",
                "det_score": round(det_score, 2),
            }

        return {
            "ok": True,
            "msg": "Rostro detectado correctamente",
            "det_score": round(det_score, 2),
            "face_size": round(min(fw, fh)),
            "liveness": round(liveness["score"], 2),
            "liveness_ok": liveness["is_live"],
        }

    except Exception as e:
        logger.exception(f"Error en validar_rostro: {e}")
        return {"ok": False, "msg": "Error al procesar la imagen"}


@router.post("/registrar_multiframe")
async def registrar_multiframe(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Registro de comensal con captura multi-frame (15 fotos).
    Pipeline: deteccion → calidad → liveness → embedding → promedio → anti-duplicado → guardar.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            {"success": False, "message": "Datos invalidos"}, status_code=400
        )

    frames_b64 = body.get("frames", [])
    dni = body.get("dni", "").strip()
    nombre = body.get("nombre", "").strip()
    apellido = body.get("apellido", "").strip()
    email = (body.get("email") or "").strip() or None

    # --- Validaciones basicas ---
    if not dni or not nombre or not apellido:
        return JSONResponse(
            {"success": False, "message": "DNI, nombre y apellido son obligatorios"},
            status_code=400,
        )

    if len(dni) != 8 or not dni.isdigit():
        return JSONResponse(
            {"success": False, "message": "DNI debe tener exactamente 8 digitos"},
            status_code=400,
        )

    if len(frames_b64) < settings.FRAMES_FOR_REGISTRATION:
        return JSONResponse(
            {
                "success": False,
                "message": f"Se requieren minimo {settings.FRAMES_FOR_REGISTRATION} capturas, se recibieron {len(frames_b64)}",
            },
            status_code=400,
        )

    # --- Verificar DNI unico ---
    existing = await db.execute(select(Comensal).where(Comensal.dni == dni))
    if existing.scalar_one_or_none():
        return JSONResponse(
            {"success": False, "message": "DNI ya registrado"}, status_code=400
        )

    # --- Procesar cada frame ---
    valid_embeddings = []
    rejection_reasons = {"no_face": 0, "low_conf": 0, "small": 0, "low_light": 0, "spoof": 0, "error": 0}

    for frame_b64 in frames_b64:
        try:
            # Decodificar base64
            data_part = frame_b64.split(",")[1] if "," in frame_b64 else frame_b64
            img_data = base64.b64decode(data_part)
            nparr = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                rejection_reasons["error"] += 1
                continue

            # Detectar rostro
            detection = await run_in_threadpool(face_service.detect_face, image)
            if detection is None:
                rejection_reasons["no_face"] += 1
                continue

            # Verificar confianza de deteccion
            if detection["det_score"] < settings.DETECTION_CONFIDENCE:
                rejection_reasons["low_conf"] += 1
                continue

            # Verificar tamano minimo del rostro
            if (
                detection["face_width"] < settings.MIN_FACE_SIZE
                or detection["face_height"] < settings.MIN_FACE_SIZE
            ):
                rejection_reasons["small"] += 1
                continue

            # Liveness check
            liveness = await run_in_threadpool(liveness_service.analyze, image, detection["bbox"])
            if liveness["detail"] == "low_light":
                rejection_reasons["low_light"] += 1
                continue
            if not liveness["is_live"]:
                rejection_reasons["spoof"] += 1
                continue

            valid_embeddings.append(detection["embedding"])

        except Exception:
            rejection_reasons["error"] += 1
            continue

    # --- Verificar minimo de frames validos ---
    if len(valid_embeddings) < settings.MIN_VALID_FRAMES:
        detail = []
        if rejection_reasons["no_face"] > 0:
            detail.append(f"{rejection_reasons['no_face']} sin rostro")
        if rejection_reasons["low_conf"] > 0:
            detail.append(f"{rejection_reasons['low_conf']} baja confianza")
        if rejection_reasons["small"] > 0:
            detail.append(f"{rejection_reasons['small']} rostro muy pequeno")
        if rejection_reasons["low_light"] > 0:
            detail.append(f"{rejection_reasons['low_light']} con poca luz")
        if rejection_reasons["spoof"] > 0:
            detail.append(f"{rejection_reasons['spoof']} liveness fallido")

        return JSONResponse(
            {
                "success": False,
                "message": (
                    f"Solo {len(valid_embeddings)} capturas validas de {len(frames_b64)}. "
                    f"Se requieren minimo {settings.MIN_VALID_FRAMES}. "
                    f"Rechazos: {', '.join(detail) if detail else 'errores varios'}. "
                    f"Asegurate de tener buena iluminacion y mirar a la camara."
                ),
            },
            status_code=400,
        )

    # --- Calcular embedding promedio (L2 normalizado) ---
    avg_embedding = np.mean(valid_embeddings, axis=0).astype(np.float32)
    norm = np.linalg.norm(avg_embedding)
    if norm > 0:
        avg_embedding = avg_embedding / norm

    # --- ANTI-DUPLICADO: verificar si el rostro ya existe ---
    cache = EmbeddingCache.get_instance()
    duplicate = cache.check_duplicate(avg_embedding, settings.DUPLICATE_THRESHOLD)
    if duplicate:
        return JSONResponse(
            {
                "success": False,
                "message": (
                    f"Este rostro ya esta registrado "
                    f"(coincide con {duplicate['nombre']} {duplicate['apellido']}, "
                    f"DNI: {duplicate['dni']})"
                ),
            },
            status_code=400,
        )

    # --- Crear comensal ---
    comensal = Comensal(
        dni=dni,
        nombre=nombre,
        apellido=apellido,
        email=email,
    )
    db.add(comensal)
    await db.flush()

    # --- Guardar embeddings (promedio + individuales) ---
    await embedding_service.save_batch(
        db, comensal.id, valid_embeddings, avg_embedding, "registro_webcam"
    )

    await db.commit()

    # --- Actualizar cache en memoria ---
    cache.add(
        comensal.id,
        {"nombre": nombre, "apellido": apellido, "dni": dni},
        avg_embedding,
    )

    return {
        "success": True,
        "message": f"{nombre} {apellido} registrado correctamente",
        "frames_validos": len(valid_embeddings),
        "frames_totales": len(frames_b64),
    }
