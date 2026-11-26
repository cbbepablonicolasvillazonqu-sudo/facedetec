"""Prueba la orquestación de main.reconocer con dobles de los módulos de Matt y Fabian."""
import numpy as np

import main
from src import config


def test_reconocer_combina_detector_embedding_y_matcher(monkeypatch):
    rostros = [
        {"box": (10, 10, 50, 50), "crop": np.zeros((50, 50, 3), np.uint8), "confidence": 0.99},
        {"box": (100, 10, 50, 50), "crop": np.ones((50, 50, 3), np.uint8), "confidence": 0.95},
    ]
    monkeypatch.setattr(main, "detectar_rostros", lambda frame: rostros)
    monkeypatch.setattr(main, "extraer_embedding", lambda crop: np.array([float(crop.max())]))

    def matcher_falso(emb, indice, umbral):
        assert umbral == config.UMBRAL_DISTANCIA_COSENO
        if emb[0] == 0:
            return {"reconocido": True, "id": 3, "nombre": "Ana García", "profesion": "Arquitecta", "score": 0.9}
        return {"reconocido": False, "nombre": config.ETIQUETA_DESCONOCIDO, "profesion": "", "score": 0.7}

    monkeypatch.setattr(main, "buscar_coincidencia", matcher_falso)

    salida = main.reconocer(np.zeros((200, 200, 3), np.uint8), indice=[{"nombre": "Ana García"}])

    assert salida == [
        {"box": (10, 10, 50, 50), "id": 3, "nombre": "Ana García", "profesion": "Arquitecta",
         "reconocido": True, "score": 0.9},
        {"box": (100, 10, 50, 50), "id": None, "nombre": config.ETIQUETA_DESCONOCIDO, "profesion": "",
         "reconocido": False, "score": 0.7},
    ]


def test_reconocer_con_indice_vacio_marca_invitados_sin_calcular_embedding(monkeypatch):
    monkeypatch.setattr(main, "detectar_rostros", lambda f: [{"box": (0, 0, 40, 40), "crop": None, "confidence": 1.0}])

    def no_llamar(*_):
        raise AssertionError("no debe calcular embeddings si no hay asistentes")

    monkeypatch.setattr(main, "extraer_embedding", no_llamar)
    salida = main.reconocer(np.zeros((100, 100, 3), np.uint8), indice=[])
    assert salida[0]["reconocido"] is False
    assert salida[0]["nombre"] == config.ETIQUETA_DESCONOCIDO
