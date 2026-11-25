import numpy as np

from src.matcher import buscar_coincidencia


def test_reconoce_vector_identico_y_devuelve_similitud():
    indice = [{"nombre": "Ana García", "profesion": "Arquitecta", "embedding": np.array([1, 0])}]
    resultado = buscar_coincidencia(np.array([1, 0]), indice)
    assert resultado == {"reconocido": True, "nombre": "Ana García", "profesion": "Arquitecta", "score": 1.0}


def test_distancia_sobre_umbral_es_desconocida():
    indice = [{"nombre": "Ana", "profesion": "", "embedding": np.array([1, 0])}]
    resultado = buscar_coincidencia(np.array([0, 1]), indice, umbral=0.38)
    assert resultado["reconocido"] is False
    assert resultado["nombre"] == "Invitado / No registrado"
    assert resultado["score"] == 1.0


def test_indice_vacio_y_vectores_invalidos_no_reconocen():
    assert not buscar_coincidencia(np.array([1, 0]), [])["reconocido"]
    assert not buscar_coincidencia(np.array([0, 0]), [{"embedding": np.array([1, 0])}])["reconocido"]


def test_ignora_embeddings_de_dimension_incompatible():
    resultado = buscar_coincidencia(np.array([1, 0]), [{"embedding": np.array([1, 0, 0])}])
    assert resultado["reconocido"] is False
