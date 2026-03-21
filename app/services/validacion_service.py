from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, desc
from app.models.marcacion import Marcacion
from app.models.asistencia import Asistencia
from app.models.intento_acceso import IntentoAcceso
from datetime import datetime, timedelta
from typing import Optional, Dict

class ValidacionService:
    
    COOLDOWN_SEGUNDOS = 300  # 5 minutos
    
    async def validar_marcacion(
        self, 
        db: AsyncSession, 
        comensal_id: int,
        comida: str,
        confianza: float,
        score_liveness: float
    ) -> Dict:
        """Valida si se puede realizar la marcación"""
        ahora = datetime.now()
        
        # 1. ELIMINAR COOLDOWN - Solo verificar comida consumida
        ya_consumio = await self._verificar_comida_consumida(db, comensal_id, ahora.date(), comida)
        
        if ya_consumio:
            return {
                "valido": False,
                "motivo": "comida_ya_consumida",
                "mensaje": f"Ya consumiste {comida} hoy. Contacta a soporte si necesitas ayuda.",
                "comida": comida
            }
        
        # 2. Verificar horario
        if not comida:
            return {
                "valido": False,
                "motivo": "fuera_horario",
                "mensaje": "Fuera de horario de atención",
                "hora_actual": ahora.time().strftime("%H:%M")
            }
        
        # 3. Validar confianza mínima
        if confianza < 80:
            return {
                "valido": False,
                "motivo": "baja_confianza",
                "mensaje": "Confianza de reconocimiento insuficiente. Intenta de nuevo.",
                "confianza": confianza
            }
        
        return {
            "valido": True,
            "motivo": None,
            "mensaje": "Marcación válida"
        }
    
    async def _obtener_ultima_marcacion(
        self, 
        db: AsyncSession, 
        comensal_id: int
    ) -> Optional[Marcacion]:
        """Obtiene la última marcación del comensal"""
        result = await db.execute(
            select(Marcacion)
            .where(Marcacion.comensal_id == comensal_id)
            .order_by(desc(Marcacion.fecha_hora))
            .limit(1)
        )
        return result.scalar_one_or_none()
    
    async def _verificar_comida_consumida(
        self,
        db: AsyncSession,
        comensal_id: int,
        fecha: datetime.date,
        comida: str
    ) -> bool:
        """Verifica si ya consumió esta comida hoy"""
        result = await db.execute(
            select(Asistencia).where(
                and_(
                    Asistencia.comensal_id == comensal_id,
                    Asistencia.fecha == fecha
                )
            )
        )
        asistencia = result.scalar_one_or_none()
        
        if not asistencia:
            return False
        
        if comida == 'desayuno':
            return asistencia.desayuno
        elif comida == 'almuerzo':
            return asistencia.almuerzo
        elif comida == 'cena':
            return asistencia.cena
        
        return False
    
    async def registrar_intento(
        self,
        db: AsyncSession,
        comensal_id: Optional[int],
        tipo_intento: str,
        resultado: str,
        motivo_rechazo: Optional[str] = None,
        confianza: Optional[float] = None,
        score_liveness: Optional[float] = None,
        es_suplantacion: bool = False,
        es_doble_marcacion: bool = False,
        tiempo_desde_ultima: Optional[int] = None
    ):
        """Registra intento de acceso para auditoría"""
        intento = IntentoAcceso(
            comensal_id=comensal_id,
            tipo_intento=tipo_intento,
            resultado=resultado,
            motivo_rechazo=motivo_rechazo,
            confianza=confianza,
            score_liveness=score_liveness,
            es_suplantacion=es_suplantacion,
            es_doble_marcacion=es_doble_marcacion,
            tiempo_desde_ultima_marcacion=tiempo_desde_ultima
        )
        db.add(intento)
        await db.commit()

validacion_service = ValidacionService()