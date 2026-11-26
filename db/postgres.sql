-- Esquema PostgreSQL del sistema de reconocimiento facial para eventos.
-- Idempotente: se puede ejecutar varias veces sin perder datos.
--   docker compose up -d                  (lo aplica solo al crear el volumen)
--   python -m src.db_postgres init        (o aplicarlo a mano sobre una base existente)
--
-- Las fotos siguen en fotos_registrados/; aquí se guarda solo el nombre del archivo.
-- Los vectores faciales son datos biométricos: esta base debe quedarse en la laptop
-- del evento (docker-compose.yml la expone solo en 127.0.0.1).

-- Personas registradas para el evento.
CREATE TABLE IF NOT EXISTS asistentes (
    id          SERIAL PRIMARY KEY,
    nombre      TEXT        NOT NULL CHECK (btrim(nombre) <> ''),
    profesion   TEXT        NOT NULL DEFAULT '',
    foto        TEXT        NOT NULL UNIQUE,          -- archivo dentro de fotos_registrados/
    activo      BOOLEAN     NOT NULL DEFAULT TRUE,     -- FALSE = no se reconoce (sin borrar historial)
    creado_en   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Vector facial de la foto de cada asistente, por modelo y detector.
-- Reemplaza a embeddings_cache.pkl: se recalcula solo si la foto cambia (foto_mtime_ns).
CREATE TABLE IF NOT EXISTS embeddings (
    asistente_id   INTEGER     NOT NULL REFERENCES asistentes (id) ON DELETE CASCADE,
    modelo         TEXT        NOT NULL,                -- p. ej. 'Facenet512'
    detector       TEXT        NOT NULL,                -- p. ej. 'opencv'
    foto_mtime_ns  BIGINT      NOT NULL,
    vector         REAL[]      NOT NULL CHECK (cardinality(vector) > 0),
    actualizado_en TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (asistente_id, modelo, detector)
);

-- Registro de asistencia: cada vez que alguien registrado pasa frente a la cámara.
-- main.py anota como máximo una pasada por persona cada PASADA_COOLDOWN_S segundos.
CREATE TABLE IF NOT EXISTS pasadas (
    id            BIGSERIAL   PRIMARY KEY,
    asistente_id  INTEGER     NOT NULL REFERENCES asistentes (id) ON DELETE CASCADE,
    score         REAL,                                 -- similitud (1 - distancia coseno)
    visto_en      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_pasadas_visto_en ON pasadas (visto_en);
CREATE INDEX IF NOT EXISTS idx_pasadas_asistente ON pasadas (asistente_id, visto_en);

-- Resumen de asistencia por persona (primera y última vez visto).
CREATE OR REPLACE VIEW resumen_asistencia AS
SELECT a.id,
       a.nombre,
       a.profesion,
       count(p.id)      AS pasadas,
       min(p.visto_en)  AS primera_vez,
       max(p.visto_en)  AS ultima_vez
FROM asistentes a
LEFT JOIN pasadas p ON p.asistente_id = a.id
GROUP BY a.id, a.nombre, a.profesion;
