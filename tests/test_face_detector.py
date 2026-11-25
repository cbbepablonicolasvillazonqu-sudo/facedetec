import sys
import types

import numpy as np

from src.face_detector import detectar_rostros


def test_recorta_caja_a_limites_y_filtra_baja_confianza(monkeypatch):
    fake = types.SimpleNamespace(DeepFace=types.SimpleNamespace(extract_faces=lambda **_: [
        {"facial_area": {"x": -4, "y": 5, "w": 50, "h": 60, "confidence": 0.9}},
        {"facial_area": {"x": 10, "y": 10, "w": 50, "h": 50, "confidence": 0.2}},
    ]))
    monkeypatch.setitem(sys.modules, "deepface", fake)
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    resultado = detectar_rostros(frame)
    assert len(resultado) == 1
    assert resultado[0]["box"] == (0, 5, 46, 60)
    assert resultado[0]["crop"].shape == (60, 46, 3)


def test_frame_vacio_no_invoca_modelo():
    assert detectar_rostros(np.empty((0, 0, 3), dtype=np.uint8)) == []
