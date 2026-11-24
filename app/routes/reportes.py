from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from app.core.database import get_db
from app.models.marcacion import Marcacion
from app.models.comensal import Comensal
from app.models.asistencia import Asistencia
from app.models.intento_acceso import IntentoAcceso
from app.routes.auth import get_current_user
from datetime import datetime, date, timedelta

# Todos los reportes exigen el token del login (el dashboard ya lo envia)
router = APIRouter(prefix="/api", tags=["reportes"], dependencies=[Depends(get_current_user)])

@router.get("/kpis")
async def get_kpis(db: AsyncSession = Depends(get_db)):
    hoy = date.today()
    
    # Total comensales
    total_comensales = await db.execute(select(func.count(Comensal.id)))
    total = total_comensales.scalar()
    
    # Marcaciones hoy
    marcaciones_hoy = await db.execute(
        select(func.count(Marcacion.id)).where(
            func.date(Marcacion.fecha_hora) == hoy
        )
    )
    hoy_count = marcaciones_hoy.scalar()
    
    # Marcaciones semana
    inicio_semana = hoy - timedelta(days=hoy.weekday())
    marcaciones_semana = await db.execute(
        select(func.count(Marcacion.id)).where(
            func.date(Marcacion.fecha_hora) >= inicio_semana
        )
    )
    semana_count = marcaciones_semana.scalar()
    
    # Promedio diario
    promedio = round(semana_count / 7, 1) if semana_count > 0 else 0
    
    return {
        "total_comensales": total,
        "marcaciones_hoy": hoy_count,
        "marcaciones_semana": semana_count,
        "promedio_diario": promedio
    }

@router.get("/marcaciones_recientes")
async def get_marcaciones_recientes(
    limite: int = 20,
    db: AsyncSession = Depends(get_db)
):
    query = select(Marcacion, Comensal).join(
        Comensal, Marcacion.comensal_id == Comensal.id
    ).order_by(Marcacion.fecha_hora.desc()).limit(limite)
    
    result = await db.execute(query)
    marcaciones = []
    
    for marcacion, comensal in result:
        marcaciones.append({
            "id": marcacion.id,
            "fecha_hora": marcacion.fecha_hora.strftime("%Y-%m-%d %H:%M:%S"),
            "tipo": marcacion.tipo,
            "comensal": f"{comensal.nombre} {comensal.apellido}",
            "dni": comensal.dni,
            "confianza": round(marcacion.confianza * 100, 1) if marcacion.confianza else 0,
            "liveness": round(marcacion.score_liveness, 2) if marcacion.score_liveness else 0
        })
    
    return marcaciones

@router.get("/distribucion_horarios")
async def get_distribucion_horarios(
    fecha: str = None,
    db: AsyncSession = Depends(get_db)
):
    if not fecha:
        fecha = date.today()
    else:
        try:
            fecha = datetime.strptime(fecha, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(status_code=400, detail="Fecha invalida, use AAAA-MM-DD")
    
    result = await db.execute(
        select(Asistencia).where(Asistencia.fecha == fecha)
    )
    asistencias = result.scalars().all()
    
    desayuno = sum(1 for a in asistencias if a.desayuno)
    almuerzo = sum(1 for a in asistencias if a.almuerzo)
    cena = sum(1 for a in asistencias if a.cena)
    
    return {
        "fecha": str(fecha),
        "desayuno": desayuno,
        "almuerzo": almuerzo,
        "cena": cena,
        "total": len(asistencias)
    }

@router.get("/intentos_sospechosos")
async def get_intentos_sospechosos(
    limite: int = 50,
    db: AsyncSession = Depends(get_db)
):
    """Obtiene intentos de acceso sospechosos"""
    query = select(IntentoAcceso, Comensal).outerjoin(
        Comensal, IntentoAcceso.comensal_id == Comensal.id
    ).where(
        (IntentoAcceso.es_suplantacion == True) | 
        (IntentoAcceso.es_doble_marcacion == True) |
        (IntentoAcceso.resultado == 'rechazado')
    ).order_by(IntentoAcceso.fecha_hora.desc()).limit(limite)
    
    result = await db.execute(query)
    intentos = []
    
    for intento, comensal in result:
        intentos.append({
            "id": intento.id,
            "fecha_hora": intento.fecha_hora.strftime("%Y-%m-%d %H:%M:%S"),
            "comensal": f"{comensal.nombre} {comensal.apellido}" if comensal else "Desconocido",
            "dni": comensal.dni if comensal else "N/A",
            "tipo_intento": intento.tipo_intento,
            "resultado": intento.resultado,
            "motivo": intento.motivo_rechazo,
            "es_suplantacion": intento.es_suplantacion,
            "es_doble_marcacion": intento.es_doble_marcacion,
            "confianza": round(intento.confianza, 2) if intento.confianza else 0
        })
    
    return intentos

@router.get("/estadisticas_seguridad")
async def get_estadisticas_seguridad(db: AsyncSession = Depends(get_db)):
    """KPIs de seguridad"""
    hoy = date.today()
    hace_7_dias = hoy - timedelta(days=7)
    
    # Total intentos de suplantación (últimos 7 días)
    result = await db.execute(
        select(func.count(IntentoAcceso.id)).where(
            and_(
                IntentoAcceso.es_suplantacion == True,
                func.date(IntentoAcceso.fecha_hora) >= hace_7_dias
            )
        )
    )
    suplantaciones = result.scalar()
    
    # Total dobles marcaciones
    result = await db.execute(
        select(func.count(IntentoAcceso.id)).where(
            and_(
                IntentoAcceso.es_doble_marcacion == True,
                func.date(IntentoAcceso.fecha_hora) >= hace_7_dias
            )
        )
    )
    dobles_marcaciones = result.scalar()
    
    # Intentos rechazados
    result = await db.execute(
        select(func.count(IntentoAcceso.id)).where(
            and_(
                IntentoAcceso.resultado == 'rechazado',
                func.date(IntentoAcceso.fecha_hora) >= hace_7_dias
            )
        )
    )
    rechazados = result.scalar()
    
    # Intentos exitosos
    result = await db.execute(
        select(func.count(IntentoAcceso.id)).where(
            and_(
                IntentoAcceso.resultado == 'exitoso',
                func.date(IntentoAcceso.fecha_hora) >= hace_7_dias
            )
        )
    )
    exitosos = result.scalar()
    
    return {
        "periodo": "Últimos 7 días",
        "intentos_suplantacion": suplantaciones,
        "dobles_marcaciones": dobles_marcaciones,
        "intentos_rechazados": rechazados,
        "intentos_exitosos": exitosos,
        "tasa_rechazo": round((rechazados / (rechazados + exitosos) * 100), 2) if (rechazados + exitosos) > 0 else 0
    }

@router.get("/lista_comensales")
async def get_lista_comensales(db: AsyncSession = Depends(get_db)):
    """Lista completa de comensales registrados"""
    result = await db.execute(
        select(Comensal).order_by(Comensal.nombre)
    )
    comensales = result.scalars().all()
    
    return [
        {
            "id": c.id,
            "dni": c.dni,
            "nombre_completo": f"{c.nombre} {c.apellido}",
            "email": c.email,
            "fecha_registro": c.fecha_registro.strftime("%Y-%m-%d"),
            "total_marcaciones": c.total_marcaciones,
            "ultima_marcacion": c.ultima_marcacion.strftime("%Y-%m-%d %H:%M") if c.ultima_marcacion else "Nunca"
        }
        for c in comensales
    ]