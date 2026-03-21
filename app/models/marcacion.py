from sqlalchemy import Column, Integer, String, Float, TIMESTAMP, ForeignKey, Boolean
from sqlalchemy.sql import func
from app.core.database import Base

class Marcacion(Base):
    __tablename__ = "marcaciones"
    
    id = Column(Integer, primary_key=True, index=True)
    comensal_id = Column(Integer, ForeignKey("comensales.id"))
    tipo = Column(String(10), nullable=False)
    fecha_hora = Column(TIMESTAMP, server_default=func.now())
    confianza = Column(Float)
    score_liveness = Column(Float)
    embedding_usado_id = Column(Integer, ForeignKey("embeddings.id"))
    ubicacion = Column(String(50), default="entrada_principal")
    
    # NUEVO: Control de suplantación
    es_valida = Column(Boolean, default=True)
    motivo_rechazo = Column(String(200))