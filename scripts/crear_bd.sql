-- ============================================================
--  Esquema de comedor_db, reconstruido a partir de app/models/*.py
--  Ejecutar conectado a la base comedor_db:
--    psql -U postgres -d comedor_db -f scripts/crear_bd.sql
-- ============================================================

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------- usuarios (app/models/usuario.py) ----------
CREATE TABLE IF NOT EXISTS usuarios (
    id              SERIAL PRIMARY KEY,
    username        VARCHAR(50)  NOT NULL UNIQUE,
    password_hash   VARCHAR      NOT NULL,
    nombre          VARCHAR(150) NOT NULL,
    rol             VARCHAR(20)  DEFAULT 'operador',
    email           VARCHAR(150),
    activo          BOOLEAN      DEFAULT TRUE,
    fecha_creacion  TIMESTAMP    DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_usuarios_id ON usuarios (id);

-- ---------- comensales (app/models/comensal.py) ----------
CREATE TABLE IF NOT EXISTS comensales (
    id                 SERIAL PRIMARY KEY,
    dni                VARCHAR(8)   NOT NULL UNIQUE,
    nombre             VARCHAR(100) NOT NULL,
    apellido           VARCHAR(100) NOT NULL,
    email              VARCHAR(150),
    foto_registro      TEXT,
    estado             BOOLEAN      DEFAULT TRUE,
    fecha_registro     TIMESTAMP    DEFAULT now(),
    total_marcaciones  INTEGER      DEFAULT 0,
    ultima_marcacion   TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_comensales_id  ON comensales (id);
CREATE INDEX IF NOT EXISTS ix_comensales_dni ON comensales (dni);

-- ---------- embeddings (app/models/embedding.py) ----------
CREATE TABLE IF NOT EXISTS embeddings (
    id                SERIAL PRIMARY KEY,
    comensal_id       INTEGER     NOT NULL REFERENCES comensales(id) ON DELETE CASCADE,
    embedding         vector(512) NOT NULL,
    calidad           DOUBLE PRECISION NOT NULL,
    fuente            VARCHAR(20) NOT NULL,
    tiene_lentes      BOOLEAN     DEFAULT FALSE,
    tiene_barba       BOOLEAN     DEFAULT FALSE,
    iluminacion       VARCHAR(20),
    angulo_estimado   DOUBLE PRECISION,
    es_principal      BOOLEAN     DEFAULT FALSE,
    es_promedio       BOOLEAN     DEFAULT FALSE,
    usado_en_matches  INTEGER     DEFAULT 0,
    fecha_creacion    TIMESTAMP   DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_embeddings_id ON embeddings (id);
CREATE INDEX IF NOT EXISTS ix_embeddings_comensal ON embeddings (comensal_id);

-- ---------- asistencias (app/models/asistencia.py) ----------
CREATE TABLE IF NOT EXISTS asistencias (
    id            SERIAL PRIMARY KEY,
    comensal_id   INTEGER NOT NULL REFERENCES comensales(id),
    fecha         DATE    NOT NULL,
    hora_entrada  TIME,
    hora_salida   TIME,
    desayuno      BOOLEAN DEFAULT FALSE,
    almuerzo      BOOLEAN DEFAULT FALSE,
    cena          BOOLEAN DEFAULT FALSE,
    CONSTRAINT _comensal_fecha_uc UNIQUE (comensal_id, fecha)
);
CREATE INDEX IF NOT EXISTS ix_asistencias_id ON asistencias (id);

-- ---------- marcaciones (app/models/marcacion.py) ----------
CREATE TABLE IF NOT EXISTS marcaciones (
    id                  SERIAL PRIMARY KEY,
    comensal_id         INTEGER REFERENCES comensales(id),
    tipo                VARCHAR(10) NOT NULL,
    fecha_hora          TIMESTAMP   DEFAULT now(),
    confianza           DOUBLE PRECISION,
    score_liveness      DOUBLE PRECISION,
    embedding_usado_id  INTEGER REFERENCES embeddings(id),
    ubicacion           VARCHAR(50) DEFAULT 'entrada_principal',
    es_valida           BOOLEAN     DEFAULT TRUE,
    motivo_rechazo      VARCHAR(200)
);
CREATE INDEX IF NOT EXISTS ix_marcaciones_id ON marcaciones (id);

-- ---------- intentos_acceso (app/models/intento_acceso.py) ----------
CREATE TABLE IF NOT EXISTS intentos_acceso (
    id                             SERIAL PRIMARY KEY,
    comensal_id                    INTEGER REFERENCES comensales(id),
    fecha_hora                     TIMESTAMP   DEFAULT now(),
    tipo_intento                   VARCHAR(50) NOT NULL,
    resultado                      VARCHAR(20) NOT NULL,
    motivo_rechazo                 TEXT,
    confianza                      DOUBLE PRECISION,
    score_liveness                 DOUBLE PRECISION,
    ip_address                     VARCHAR(45),
    es_suplantacion                BOOLEAN DEFAULT FALSE,
    es_doble_marcacion             BOOLEAN DEFAULT FALSE,
    tiempo_desde_ultima_marcacion  INTEGER
);
CREATE INDEX IF NOT EXISTS ix_intentos_acceso_id ON intentos_acceso (id);

-- ---------- configuracion (app/models/configuracion.py) ----------
CREATE TABLE IF NOT EXISTS configuracion (
    clave        VARCHAR(100) PRIMARY KEY,
    valor        TEXT NOT NULL,
    descripcion  TEXT
);

-- ============================================================
--  Datos iniciales
-- ============================================================

-- Horarios leidos por app/routes/reconocimiento.py (_load_meal_config), formato HH:MM.
-- Sin estas filas la app no puede asignar desayuno/almuerzo/cena.
INSERT INTO configuracion (clave, valor, descripcion) VALUES
    ('horario_desayuno_inicio', '06:00', 'Inicio del desayuno'),
    ('horario_desayuno_fin',    '09:00', 'Fin del desayuno'),
    ('horario_almuerzo_inicio', '11:30', 'Inicio del almuerzo'),
    ('horario_almuerzo_fin',    '15:00', 'Fin del almuerzo'),
    ('horario_cena_inicio',     '17:30', 'Inicio de la cena'),
    ('horario_cena_fin',        '20:30', 'Fin de la cena')
ON CONFLICT (clave) DO NOTHING;

-- Usuario admin / admin123 (hash SHA-256, como en app/core/security.py y generar_hash.py)
INSERT INTO usuarios (username, password_hash, nombre, rol, email, activo) VALUES
    ('admin', '240be518fabd2724ddb6f04eeb1da5967448d7e831c08c8fa822809f74c720a9',
     'Administrador', 'admin', NULL, TRUE)
ON CONFLICT (username) DO NOTHING;
