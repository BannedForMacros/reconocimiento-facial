from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/comedor_db"
    SECRET_KEY: str = "cambiar-en-produccion-clave-super-secreta"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 480
    FACE_THRESHOLD: float = 0.60
    LIVENESS_THRESHOLD: float = 0.3
    MAX_EMBEDDINGS: int = 8
    EMBEDDING_DIM: int = 512
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()  # ← AGREGAR ESTA LÍNEA