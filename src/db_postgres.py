"""Backend PostgreSQL (opcional) para asistentes, vectores faciales y registro de pasadas.

Se activa definiendo DATABASE_URL; sin ella el sistema sigue en modo CSV.
Esquema: db/postgres.sql.

CLI:
    python -m src.db_postgres init                        # crea/actualiza las tablas
    python -m src.db_postgres importar data/asistentes.csv
    python -m src.db_postgres asistentes                  # lista asistentes
    python -m src.db_postgres pasadas                     # resumen de asistencia
"""
import argparse
import csv
import os
import time
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np

from src import config
from src.embeddings_index import vector_de_foto

ESQUEMA_PATH = os.path.join(config.BASE_DIR, "db", "postgres.sql")


def conectar(url: Optional[str] = None):
    """Abre una conexión en modo autocommit (cada sentencia se confirma al momento)."""
    import psycopg

    url = url or config.DATABASE_URL
    if not url:
        raise RuntimeError("Falta DATABASE_URL (p. ej. postgresql://facedetec:...@127.0.0.1:5432/facedetec).")
    try:
        return psycopg.connect(url, autocommit=True, connect_timeout=5)
    except psycopg.OperationalError as exc:
        raise RuntimeError(
            f"No se pudo conectar a PostgreSQL: {exc}\n"
            "¿Está levantado el contenedor? Ejecuta: docker compose up -d"
        ) from exc


def crear_esquema(conn) -> None:
    """Aplica db/postgres.sql (idempotente)."""
    with open(ESQUEMA_PATH, encoding="utf-8") as archivo:
        conn.execute(archivo.read())


def registrar_asistente(conn, nombre: str, profesion: str, foto: str) -> Dict[str, Any]:
    """Crea el asistente o, si la foto ya existe, actualiza sus datos y lo reactiva."""
    nombre, profesion, foto = nombre.strip(), (profesion or "").strip(), foto.strip()
    fila = conn.execute(
        """
        INSERT INTO asistentes (nombre, profesion, foto) VALUES (%s, %s, %s)
        ON CONFLICT (foto) DO UPDATE
            SET nombre = EXCLUDED.nombre, profesion = EXCLUDED.profesion, activo = TRUE
        RETURNING id
        """,
        (nombre, profesion, foto),
    ).fetchone()
    return {"id": fila[0], "nombre": nombre, "profesion": profesion, "foto": foto}


def importar_csv(conn, csv_path: str) -> int:
    """Copia data/asistentes.csv a la tabla asistentes (los ids los asigna la base)."""
    with open(csv_path, encoding="utf-8-sig", newline="") as archivo:
        filas = [f for f in csv.DictReader(archivo) if (f.get("nombre") or "").strip() and (f.get("foto") or "").strip()]
    for fila in filas:
        registrar_asistente(conn, fila["nombre"], fila.get("profesion", ""), fila["foto"])
    return len(filas)


def cargar_indice_pg(conn, fotos_dir: str) -> List[Dict[str, Any]]:
    """Mismo contrato que embeddings_index.cargar_indice, leyendo de PostgreSQL.

    Reutiliza el vector guardado si la foto no cambió (mtime) para el modelo y detector
    actuales; si no, lo calcula con vector_de_foto y lo guarda.
    """
    filas = conn.execute(
        """
        SELECT a.id, a.nombre, a.profesion, a.foto, e.foto_mtime_ns, e.vector
        FROM asistentes a
        LEFT JOIN embeddings e
               ON e.asistente_id = a.id AND e.modelo = %s AND e.detector = %s
        WHERE a.activo
        ORDER BY a.id
        """,
        (config.MODEL_NAME, config.DETECTOR_BACKEND),
    ).fetchall()

    indice: List[Dict[str, Any]] = []
    for asistente_id, nombre, profesion, foto, mtime_guardado, vector in filas:
        ruta = os.path.join(fotos_dir, foto)
        try:
            mtime = os.stat(ruta).st_mtime_ns
        except OSError:
            print(f"[AVISO] Foto faltante para {nombre}: {ruta}")
            continue
        if vector is not None and mtime_guardado == mtime:
            embedding = np.asarray(vector, dtype=np.float32)
        else:
            embedding = vector_de_foto(ruta, nombre)
            if embedding is None:
                continue
            conn.execute(
                """
                INSERT INTO embeddings (asistente_id, modelo, detector, foto_mtime_ns, vector)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (asistente_id, modelo, detector) DO UPDATE
                    SET foto_mtime_ns = EXCLUDED.foto_mtime_ns,
                        vector = EXCLUDED.vector,
                        actualizado_en = now()
                """,
                (asistente_id, config.MODEL_NAME, config.DETECTOR_BACKEND, mtime,
                 np.asarray(embedding, dtype=np.float32).tolist()),
            )
        indice.append({
            "id": asistente_id, "nombre": nombre, "profesion": profesion,
            "foto": foto, "embedding": embedding,
        })
    return indice


class RegistroPasadas:
    """Anota en `pasadas` a cada asistente reconocido, como máximo una vez cada `cooldown_s`.

    Un fallo de la base solo se avisa por consola: nunca debe detener la proyección.
    """

    def __init__(self, conn, cooldown_s: float = config.PASADA_COOLDOWN_S,
                 reloj: Callable[[], float] = time.monotonic) -> None:
        self._conn = conn
        self._cooldown_s = cooldown_s
        self._reloj = reloj
        self._ultima: Dict[int, float] = {}

    def anotar(self, detecciones: Iterable[Dict[str, Any]]) -> int:
        """Devuelve cuántas pasadas nuevas se guardaron."""
        ahora = self._reloj()
        nuevas = 0
        for det in detecciones:
            if not det.get("reconocido") or det.get("id") is None:
                continue
            asistente_id = int(det["id"])
            ultima = self._ultima.get(asistente_id)
            if ultima is not None and ahora - ultima < self._cooldown_s:
                continue
            self._ultima[asistente_id] = ahora  # también tras un error, para no saturar la consola
            score = det.get("score")
            try:
                self._conn.execute(
                    "INSERT INTO pasadas (asistente_id, score) VALUES (%s, %s)",
                    (asistente_id, None if score is None else float(score)),
                )
                nuevas += 1
            except Exception as exc:
                print(f"[AVISO] No se pudo guardar la pasada de {det.get('nombre')}: {exc}")
        return nuevas


def resumen_asistencia(conn) -> List[tuple]:
    return conn.execute(
        "SELECT nombre, profesion, pasadas, primera_vez, ultima_vez "
        "FROM resumen_asistencia ORDER BY ultima_vez DESC NULLS LAST, nombre"
    ).fetchall()


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.db_postgres", description=__doc__.splitlines()[0])
    parser.add_argument("--url", help="Por defecto, la variable DATABASE_URL")
    sub = parser.add_subparsers(dest="comando", required=True)
    sub.add_parser("init", help="crea o actualiza las tablas (db/postgres.sql)")
    imp = sub.add_parser("importar", help="importa asistentes desde un CSV")
    imp.add_argument("csv", nargs="?", default=config.CSV_PATH)
    sub.add_parser("asistentes", help="lista los asistentes")
    sub.add_parser("pasadas", help="resumen de asistencia")
    args = parser.parse_args(argv)

    with conectar(args.url) as conn:
        if args.comando == "init":
            crear_esquema(conn)
            print("[OK] Esquema aplicado.")
        elif args.comando == "importar":
            crear_esquema(conn)
            print(f"[OK] {importar_csv(conn, args.csv)} asistentes importados desde {args.csv}.")
        elif args.comando == "asistentes":
            for fila in conn.execute(
                "SELECT id, nombre, profesion, foto, activo FROM asistentes ORDER BY id"
            ).fetchall():
                print(" | ".join("" if v is None else str(v) for v in fila))
        elif args.comando == "pasadas":
            for nombre, profesion, total, primera, ultima in resumen_asistencia(conn):
                if not total:
                    print(f"{nombre} ({profesion}): sin pasadas")
                    continue
                hora = lambda t: t.astimezone().strftime("%d/%m %H:%M:%S")  # noqa: E731 (hora local)
                print(f"{nombre} ({profesion}): {total} pasadas · primera {hora(primera)} · última {hora(ultima)}")


if __name__ == "__main__":
    main()
