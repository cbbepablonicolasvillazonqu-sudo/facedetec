"""Tests de integración contra un PostgreSQL real.

Se ejecutan solo si TEST_DATABASE_URL está definida (en el CI hay un servicio postgres):
    docker compose up -d
    TEST_DATABASE_URL=postgresql://facedetec:facedetec_local@127.0.0.1:5432/facedetec pytest

Cada test trabaja en un schema temporal que se borra al terminar: no toca datos reales.
"""
import os
import uuid

import numpy as np
import pytest

URL = os.environ.get("TEST_DATABASE_URL", "")
pytestmark = pytest.mark.skipif(not URL, reason="TEST_DATABASE_URL no definida")

psycopg = pytest.importorskip("psycopg")

from src import db_postgres  # noqa: E402


@pytest.fixture
def conn():
    conexion = psycopg.connect(URL, autocommit=True)
    esquema = f"test_{uuid.uuid4().hex[:10]}"
    conexion.execute(f"CREATE SCHEMA {esquema}")
    conexion.execute(f"SET search_path TO {esquema}")
    db_postgres.crear_esquema(conexion)
    yield conexion
    conexion.execute(f"DROP SCHEMA {esquema} CASCADE")
    conexion.close()


@pytest.fixture
def fotos(tmp_path):
    carpeta = tmp_path / "fotos"
    carpeta.mkdir()
    for nombre in ("ana.jpg", "jose.jpg"):
        (carpeta / nombre).write_bytes(b"foto")
    return carpeta


@pytest.fixture
def vectores(monkeypatch):
    """Sustituye el cálculo real (DeepFace) por vectores fijos y cuenta las llamadas."""
    llamadas = []

    def falso(ruta, nombre="asistente"):
        llamadas.append(os.path.basename(ruta))
        return np.array([1.0, 0.5, 0.25], np.float32)

    monkeypatch.setattr(db_postgres, "vector_de_foto", falso)
    return llamadas


def _contar(conn, tabla):
    return conn.execute(f"SELECT count(*) FROM {tabla}").fetchone()[0]


def test_esquema_es_idempotente(conn):
    db_postgres.crear_esquema(conn)
    db_postgres.crear_esquema(conn)
    assert _contar(conn, "asistentes") == 0


def test_registrar_misma_foto_actualiza_sin_duplicar(conn):
    a = db_postgres.registrar_asistente(conn, " Ana García ", "Arquitecta", "ana.jpg")
    b = db_postgres.registrar_asistente(conn, "Ana García", "Arquitecta Senior", "ana.jpg")
    assert a["id"] == b["id"]
    assert _contar(conn, "asistentes") == 1
    assert conn.execute("SELECT nombre, profesion FROM asistentes").fetchone() == ("Ana García", "Arquitecta Senior")


def test_importar_csv(conn, tmp_path):
    csv_path = tmp_path / "asistentes.csv"
    csv_path.write_text(
        "id,nombre,profesion,foto\n1,Ana García,Arquitecta,ana.jpg\n2,José Peña,Ingeniero,jose.jpg\n3,,X,vacio.jpg\n",
        encoding="utf-8",
    )
    assert db_postgres.importar_csv(conn, str(csv_path)) == 2
    assert db_postgres.importar_csv(conn, str(csv_path)) == 2  # reimportar no duplica
    assert _contar(conn, "asistentes") == 2


def test_indice_calcula_guarda_y_reutiliza_vectores(conn, fotos, vectores):
    ana = db_postgres.registrar_asistente(conn, "Ana García", "Arquitecta", "ana.jpg")

    indice = db_postgres.cargar_indice_pg(conn, str(fotos))
    assert [(p["id"], p["nombre"], p["profesion"]) for p in indice] == [(ana["id"], "Ana García", "Arquitecta")]
    assert np.allclose(indice[0]["embedding"], [1.0, 0.5, 0.25])
    assert vectores == ["ana.jpg"]
    assert _contar(conn, "embeddings") == 1

    db_postgres.cargar_indice_pg(conn, str(fotos))
    assert vectores == ["ana.jpg"]  # segunda carga: vector leído de la base

    st = os.stat(fotos / "ana.jpg")
    os.utime(fotos / "ana.jpg", ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    db_postgres.cargar_indice_pg(conn, str(fotos))
    assert vectores == ["ana.jpg", "ana.jpg"]  # foto cambiada: se recalcula
    assert _contar(conn, "embeddings") == 1  # y se actualiza, no se duplica


def test_indice_omite_inactivos_y_fotos_faltantes(conn, fotos, vectores):
    db_postgres.registrar_asistente(conn, "Ana García", "Arquitecta", "ana.jpg")
    db_postgres.registrar_asistente(conn, "Sin Foto", "", "no_existe.jpg")
    jose = db_postgres.registrar_asistente(conn, "José Peña", "Ingeniero", "jose.jpg")
    conn.execute("UPDATE asistentes SET activo = FALSE WHERE id = %s", (jose["id"],))

    indice = db_postgres.cargar_indice_pg(conn, str(fotos))

    assert [p["nombre"] for p in indice] == ["Ana García"]


def test_pasadas_respetan_cooldown_y_alimentan_resumen(conn):
    ana = db_postgres.registrar_asistente(conn, "Ana García", "Arquitecta", "ana.jpg")
    reloj = {"t": 0.0}
    registro = db_postgres.RegistroPasadas(conn, cooldown_s=60, reloj=lambda: reloj["t"])
    reconocida = {"id": ana["id"], "nombre": "Ana García", "reconocido": True, "score": 0.91}
    invitado = {"id": None, "nombre": "Invitado / No registrado", "reconocido": False, "score": 0.6}

    assert registro.anotar([reconocida, invitado]) == 1
    reloj["t"] = 30
    assert registro.anotar([reconocida]) == 0  # dentro del cooldown
    reloj["t"] = 61
    assert registro.anotar([reconocida]) == 1

    assert _contar(conn, "pasadas") == 2
    nombre, profesion, total, primera, ultima = db_postgres.resumen_asistencia(conn)[0]
    assert (nombre, profesion, total) == ("Ana García", "Arquitecta", 2)
    assert primera <= ultima


def test_fallo_de_base_no_detiene_la_proyeccion(conn, capsys):
    registro = db_postgres.RegistroPasadas(conn, cooldown_s=60)
    # id inexistente -> viola la clave foránea; debe avisar, no lanzar.
    assert registro.anotar([{"id": 999999, "nombre": "Fantasma", "reconocido": True, "score": 0.9}]) == 0
    assert "No se pudo guardar la pasada" in capsys.readouterr().out
