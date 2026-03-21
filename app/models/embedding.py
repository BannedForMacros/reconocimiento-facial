from sqlalchemy import Column, Integer, Float, String, Boolean, TIMESTAMP, ForeignKey
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from app.core.database import Base

class Embedding(Base):
    __tablename__ = "embeddings"
    
    id = Column(Integer, primary_key=True, index=True)
    comensal_id = Column(Integer, ForeignKey("comensales.id", ondelete="CASCADE"), nullable=False)
    embedding = Column(Vector(512), nullable=False)
    calidad = Column(Float, nullable=False)
    fuente = Column(String(20), nullable=False)
    tiene_lentes = Column(Boolean, default=False)
    tiene_barba = Column(Boolean, default=False)
    iluminacion = Column(String(20))
    angulo_estimado = Column(Float)
    es_principal = Column(Boolean, default=False)
    usado_en_matches = Column(Integer, default=0)
    fecha_creacion = Column(TIMESTAMP, server_default=func.now())