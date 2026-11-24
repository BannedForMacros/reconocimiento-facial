from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/comedor_db"
    SECRET_KEY: str = "cambiar-en-produccion-clave-super-secreta"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 480

    # === Reconocimiento Facial ===
    MATCH_THRESHOLD: float = 0.45
    DUPLICATE_THRESHOLD: float = 0.50
    DETECTION_CONFIDENCE: float = 0.80
    MIN_FACE_SIZE: int = 80
    ENRICHMENT_THRESHOLD: float = 0.30  # si distancia > esto, enriquecer embedding
    FRAMES_FOR_REGISTRATION: int = 15
    MIN_VALID_FRAMES: int = 10

    # === Liveness (Anti-Spoofing, MiniFASNet) ===
    # Probabilidad minima de "rostro real" (0-1). Calibrar con datos propios.
    LIVENESS_THRESHOLD: float = 0.50
    # Brillo medio minimo del rostro (0-255). Por debajo se pide mas luz.
    LIVENESS_MIN_BRIGHTNESS: float = 60.0

    # === Liveness multi-fotograma (rafaga) ===
    # Modo del kiosco: "profundidad" (invisible) o "luz" (destello de colores + profundidad)
    LIVENESS_MODO: str = "profundidad"
    # False = modo sombra: las capas nuevas solo miden y registran, no rechazan.
    LIVENESS_CAPAS_ACTIVAS: bool = False
    RAFAGA_FOTOGRAMAS: int = 5          # fotogramas por intento (modo profundidad)
    RAFAGA_INTERVALO_MS: int = 180      # separacion entre fotogramas
    RAFAGA_MIN_VALIDOS: int = 3         # fotogramas con rostro valido requeridos
    # Profundidad (medido con CREMA-D: real 98.9% 3D, fotos 97.8% planas)
    PROFUNDIDAD_UMBRAL: float = 0.35         # ratio no-planaridad/movimiento minimo para "3D"
    PROFUNDIDAD_MOVIMIENTO_MIN: float = 12.0 # foto quieta = ruido <= ~8; debajo de esto no se decide
    PROFUNDIDAD_INCONCLUSO_RECHAZA: bool = False  # True = exigir un minimo de movimiento
    # Fondo quieto (camara fija): el anillo alrededor de la cabeza no debe moverse
    FONDO_MIN_PUNTOS: int = 12               # puntos con textura necesarios para decidir
    FONDO_TEXTURA_MIN: float = 0.002         # textura minima absoluta (pared lisa = no cuenta)
    FONDO_DESPLAZAMIENTO_MIN: float = 0.6    # px: por debajo es ruido de sensor
    FONDO_LAMINA_MIN: int = 8                # puntos minimos moviendose juntos como lamina
    FONDO_COHERENCIA_MIN: float = 0.15       # fraccion minima del anillo que forma la lamina
    FONDO_LAMINA_POR_LADO: int = 3           # la lamina debe rodear la cara (puntos a cada lado)
    # Solo se registra: con 2 personas en la fila genera falsas alarmas (-23%) y el fondo
    # de referencia ya cubre lo mismo. Activar solo si el kiosco no tiene fila detras.
    FONDO_LAMINA_DECIDE: bool = False
    # Fondo de referencia (el kiosco memoriza su fondo vacio)
    FONDO_REF_ANCHO: float = 1.0             # anillo interior (el aparato siempre lo tapa)
    FONDO_REF_BLOQUE: int = 16               # tamano de bloque para comparar textura (px)
    FONDO_REF_TEXTURA_MIN: float = 4.0       # desviacion minima del bloque en la referencia
    FONDO_REF_ZNCC_MIN: float = 0.6          # correlacion minima para "mismo fondo"
    FONDO_REF_MIN_BLOQUES: int = 6           # bloques con textura necesarios para decidir
    FONDO_REF_MIN: float = 0.30              # fraccion de bloques que coincide (real >=0.39, ataques <=0.25)
    FONDO_REF_APRENDIZAJE: float = 0.05      # velocidad de actualizacion del fondo
    FONDO_REF_MIN_MUESTRAS: int = 5          # fotogramas vacios antes de usarlo
    # Luz
    LUZ_NUM_COLORES: int = 4
    LUZ_MS_POR_COLOR: int = 360         # < 3 cambios por segundo (WCAG 2.3.1)
    LUZ_CORRELACION_MIN: float = 0.5
    LUZ_RESPUESTA_MIN: float = 0.02
    # Rendimiento: tamano de entrada del detector (640 = original, 480/320 = mas rapido)
    DET_SIZE: int = 480

    # === Cooldown ===
    ATTENDANCE_COOLDOWN_MINUTES: int = 30

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()