from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/comedor_db"
    SECRET_KEY: str = "cambiar-en-produccion-clave-super-secreta"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 480
    EMBEDDING_DIM: int = 512

    # === Reconocimiento Facial ===
    MATCH_THRESHOLD: float = 0.45
    DUPLICATE_THRESHOLD: float = 0.50
    DETECTION_CONFIDENCE: float = 0.80
    MIN_FACE_SIZE: int = 80
    ENRICHMENT_THRESHOLD: float = 0.30  # si distancia > esto, enriquecer embedding
    FRAMES_FOR_REGISTRATION: int = 15
    MIN_VALID_FRAMES: int = 10

    # === Liveness (Anti-Spoofing) ===
    LIVENESS_THRESHOLD: float = 0.45

    # Rendimiento: tamano de entrada del detector (640 = original, 480/320 = mas rapido)
    DET_SIZE: int = 480

    # === Sistema de Votacion ===
    VOTING_WINDOW: int = 5
    VOTING_THRESHOLD: int = 3

    # === Cooldown ===
    ATTENDANCE_COOLDOWN_MINUTES: int = 30

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()