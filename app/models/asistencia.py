from sqlalchemy import Column, Integer, Date, Time, Boolean, ForeignKey, UniqueConstraint
from app.core.database import Base

class Asistencia(Base):
    __tablename__ = "asistencias"
    
    id = Column(Integer, primary_key=True, index=True)
    comensal_id = Column(Integer, ForeignKey("comensales.id"), nullable=False)
    fecha = Column(Date, nullable=False)
    hora_entrada = Column(Time)
    hora_salida = Column(Time)
    desayuno = Column(Boolean, default=False)
    almuerzo = Column(Boolean, default=False)
    cena = Column(Boolean, default=False)
    
    __table_args__ = (UniqueConstraint('comensal_id', 'fecha', name='_comensal_fecha_uc'),)