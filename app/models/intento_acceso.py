from sqlalchemy import Column, Integer, String, Float, TIMESTAMP, ForeignKey, Boolean, Text
from sqlalchemy.sql import func
from app.core.database import Base

class IntentoAcceso(Base):
    __tablename__ = "intentos_acceso"
    
    id = Column(Integer, primary_key=True, index=True)
    comensal_id = Column(Integer, ForeignKey("comensales.id"))
    fecha_hora = Column(TIMESTAMP, server_default=func.now())
    tipo_intento = Column(String(50), nullable=False)
    resultado = Column(String(20), nullable=False)
    motivo_rechazo = Column(Text)
    confianza = Column(Float)
    score_liveness = Column(Float)
    ip_address = Column(String(45))
    es_suplantacion = Column(Boolean, default=False)
    es_doble_marcacion = Column(Boolean, default=False)
    tiempo_desde_ultima_marcacion = Column(Integer)