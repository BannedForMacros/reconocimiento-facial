from sqlalchemy import Column, Integer, String, Boolean, TIMESTAMP, Text
from sqlalchemy.sql import func
from app.core.database import Base

class Comensal(Base):
    __tablename__ = "comensales"
    
    id = Column(Integer, primary_key=True, index=True)
    dni = Column(String(8), unique=True, nullable=False, index=True)
    nombre = Column(String(100), nullable=False)
    apellido = Column(String(100), nullable=False)
    email = Column(String(150))
    foto_registro = Column(Text)
    estado = Column(Boolean, default=True)
    fecha_registro = Column(TIMESTAMP, server_default=func.now())
    total_marcaciones = Column(Integer, default=0)
    ultima_marcacion = Column(TIMESTAMP)