"""cargar_indice: vectores sobre el recorte de la cara, caché y fotos problemáticas.

El detector y DeepFace se sustituyen por dobles: no requiere TensorFlow.
"""
import csv

import cv2
import numpy as np
import pytest

from src import embeddings_index


def _guardar_foto(ruta, alto=200, ancho=160):
    # imencode + tofile para soportar nombres con tildes en Windows.
    ok, datos = cv2.imencode(".jpg", np.full((alto, ancho, 3), 128, np.uint8))
    assert ok
    datos.tofile(str(ruta))


def _escribir_csv(ruta, filas):
    with open(ruta, "w", encoding="utf-8", newline="") as archivo:
        writer = csv.writer(archivo)
        writer.writerow(["id", "nombre", "profesion", "foto"])
        writer.writerows(filas)


@pytest.fixture
def entorno(tmp_path, monkeypatch):
    fotos = tmp_path / "fotos"
    fotos.mkdir()
    csv_path = tmp_path / "asistentes.csv"
    cache_path = tmp_path / "cache.pkl"
    recibidos = []

    def detector_falso(imagen):
        # Dos caras: una pequeña y una grande; debe elegirse la grande.
        return [
            {"box": (0, 0, 40, 40), "crop": np.zeros((40, 40, 3), np.uint8), "confidence": 0.99},
            {"box": (50, 60, 90, 110), "crop": np.ones((110, 90, 3), np.uint8), "confidence": 0.95},
        ]

    def embedding_falso(rostro):
        recibidos.append(rostro.shape)
        return np.array([1.0, 0.0, 0.0], np.float32)

    monkeypatch.setattr(embeddings_index, "detectar_rostros", detector_falso)
    monkeypatch.setattr(embeddings_index, "extraer_embedding", embedding_falso)
    return fotos, csv_path, cache_path, recibidos


def test_vector_se_calcula_sobre_la_cara_mas_grande(entorno):
    fotos, csv_path, cache_path, recibidos = entorno
    _guardar_foto(fotos / "ana.jpg")
    _escribir_csv(csv_path, [[1, "Ana García", "Arquitecta", "ana.jpg"]])

    indice = embeddings_index.cargar_indice(str(csv_path), str(fotos), str(cache_path))

    assert recibidos == [(110, 90, 3)]  # el recorte grande, no la foto completa (200x160)
    assert [p["nombre"] for p in indice] == ["Ana García"]
    assert indice[0]["profesion"] == "Arquitecta"


def test_segunda_carga_reutiliza_la_cache(entorno):
    fotos, csv_path, cache_path, recibidos = entorno
    _guardar_foto(fotos / "ana.jpg")
    _escribir_csv(csv_path, [[1, "Ana García", "Arquitecta", "ana.jpg"]])

    embeddings_index.cargar_indice(str(csv_path), str(fotos), str(cache_path))
    indice = embeddings_index.cargar_indice(str(csv_path), str(fotos), str(cache_path))

    assert len(recibidos) == 1
    assert np.array_equal(indice[0]["embedding"], np.array([1.0, 0.0, 0.0], np.float32))


def test_cache_de_version_antigua_se_descarta(entorno):
    import pickle

    fotos, csv_path, cache_path, recibidos = entorno
    _guardar_foto(fotos / "ana.jpg")
    _escribir_csv(csv_path, [[1, "Ana García", "Arquitecta", "ana.jpg"]])
    with open(cache_path, "wb") as archivo:  # caché v1 (vectores de foto completa)
        pickle.dump({"version": 1, "entries": {}}, archivo)

    embeddings_index.cargar_indice(str(csv_path), str(fotos), str(cache_path))

    with open(cache_path, "rb") as archivo:
        assert pickle.load(archivo)["version"] == embeddings_index.VERSION_CACHE
    assert len(recibidos) == 1


def test_foto_con_tildes_en_el_nombre(entorno):
    fotos, csv_path, cache_path, recibidos = entorno
    _guardar_foto(fotos / "josé_peña.jpg")
    _escribir_csv(csv_path, [[1, "José Peña", "Ingeniero", "josé_peña.jpg"]])

    indice = embeddings_index.cargar_indice(str(csv_path), str(fotos), str(cache_path))

    assert [p["nombre"] for p in indice] == ["José Peña"]


def test_omite_foto_sin_cara_y_foto_faltante(entorno, monkeypatch, capsys):
    fotos, csv_path, cache_path, recibidos = entorno
    _guardar_foto(fotos / "paisaje.jpg")
    _escribir_csv(csv_path, [
        [1, "Sin Cara", "X", "paisaje.jpg"],
        [2, "Sin Foto", "Y", "no_existe.jpg"],
    ])
    monkeypatch.setattr(embeddings_index, "detectar_rostros", lambda imagen: [])

    indice = embeddings_index.cargar_indice(str(csv_path), str(fotos), str(cache_path))

    assert indice == []
    assert recibidos == []
    salida = capsys.readouterr().out
    assert "No se detectó un rostro" in salida and "Sin Cara" in salida
    assert "Foto faltante" in salida and "Sin Foto" in salida
