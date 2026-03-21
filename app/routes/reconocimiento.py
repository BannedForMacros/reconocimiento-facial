from fastapi import APIRouter, UploadFile, File, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from app.core.database import get_db
from app.models.marcacion import Marcacion
from app.models.comensal import Comensal
from app.services.face_service import face_service
from app.services.embedding_service import embedding_service
from app.services.validacion_service import validacion_service
from datetime import datetime, time as dt_time
import cv2
import numpy as np

router = APIRouter(prefix="/api", tags=["reconocimiento"])

async def determinar_comida(hora: dt_time, db: AsyncSession):
    from app.models.configuracion import Configuracion
    
    config = {}
    result = await db.execute(select(Configuracion))
    for conf in result.scalars():
        config[conf.clave] = conf.valor
    
    def parse_time(t_str):
        return datetime.strptime(t_str, "%H:%M").time()
    
    if parse_time(config['horario_desayuno_inicio']) <= hora <= parse_time(config['horario_desayuno_fin']):
        return 'desayuno'
    elif parse_time(config['horario_almuerzo_inicio']) <= hora <= parse_time(config['horario_almuerzo_fin']):
        return 'almuerzo'
    elif parse_time(config['horario_cena_inicio']) <= hora <= parse_time(config['horario_cena_fin']):
        return 'cena'
    return None

@router.post("/reconocer")
async def reconocer_rostro(
    foto: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    contents = await foto.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    result = face_service.detect_face(image)
    
    if result is None:
        return {"success": False, "message": "No se detectó rostro"}
    
    if result["is_spoofing"]:
        await validacion_service.registrar_intento(
            db, None, "reconocimiento", "rechazado", "spoofing_detectado",
            score_liveness=result["liveness_score"], es_suplantacion=True
        )
        return {
            "success": False, 
            "message": "⚠️ ALERTA: Intento de spoofing detectado",
            "liveness_score": result["liveness_score"]
        }
    
    match = await embedding_service.find_match(db, result["embedding"])
    
    if not match:
        return {"success": False, "message": "Rostro no reconocido"}
    
    comensal = match["comensal"]
    ahora = datetime.now()
    confianza_pct = (1 - match["distance"]) * 100
    
    comida = await determinar_comida(ahora.time(), db)
    
    # ===== VALIDACIONES CRÍTICAS =====
    validacion = await validacion_service.validar_marcacion(
        db, comensal.id, comida, confianza_pct, result["liveness_score"]
    )
    
    if not validacion["valido"]:
        # Registrar intento rechazado
        ultima_marcacion = await validacion_service._obtener_ultima_marcacion(db, comensal.id)
        tiempo_desde = None
        if ultima_marcacion:
            tiempo_desde = int((ahora - ultima_marcacion.fecha_hora).total_seconds())
        
        await validacion_service.registrar_intento(
            db, comensal.id, "reconocimiento", "rechazado",
            validacion["mensaje"], confianza_pct, result["liveness_score"],
            es_doble_marcacion=(validacion["motivo"] == "doble_marcacion"),
            tiempo_desde_ultima=tiempo_desde
        )
        
        return {
            "success": False,
            "message": f"❌ {validacion['mensaje']}",
            "motivo": validacion["motivo"],
            "detalles": validacion
        }
    
    # Marcación válida - registrar
    marcacion = Marcacion(
        comensal_id=comensal.id,
        tipo="entrada",
        confianza=confianza_pct / 100,
        score_liveness=result["liveness_score"],
        embedding_usado_id=match["embedding_id"],
        es_valida=True
    )
    db.add(marcacion)
    
    # Actualizar asistencia
    from app.models.asistencia import Asistencia
    
    asistencia_query = await db.execute(
        select(Asistencia).where(
            and_(Asistencia.comensal_id == comensal.id, Asistencia.fecha == ahora.date())
        )
    )
    asistencia = asistencia_query.scalar_one_or_none()
    
    if not asistencia:
        asistencia = Asistencia(
            comensal_id=comensal.id,
            fecha=ahora.date(),
            hora_entrada=ahora.time()
        )
        db.add(asistencia)
    
    if comida == 'desayuno':
        asistencia.desayuno = True
    elif comida == 'almuerzo':
        asistencia.almuerzo = True
    elif comida == 'cena':
        asistencia.cena = True
    
    comensal.total_marcaciones += 1
    comensal.ultima_marcacion = ahora
    
    # Registrar intento exitoso
    await validacion_service.registrar_intento(
        db, comensal.id, "reconocimiento", "exitoso",
        None, confianza_pct, result["liveness_score"]
    )
    
    await db.commit()
    
    return {
        "success": True,
        "message": f"✅ Bienvenido, {comensal.nombre} {comensal.apellido}",
        "comida": comida or "fuera de horario",
        "comensal": {
            "dni": comensal.dni,
            "nombre": comensal.nombre,
            "apellido": comensal.apellido
        },
        "confianza": round(confianza_pct, 2),
        "liveness_score": round(result["liveness_score"], 2)
    }