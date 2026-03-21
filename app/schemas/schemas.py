from pydantic import BaseModel
from typing import Optional

class ComensalCreate(BaseModel):
    dni: str
    nombre: str
    apellido: str
    email: Optional[str] = None

class ComensalResponse(BaseModel):
    id: int
    dni: str
    nombre: str
    apellido: str
    
    class Config:
        from_attributes = True