import numpy as np
import pytest

from src import overlay_renderer
from src.overlay_renderer import COLOR_INVITADO, COLOR_RECONOCIDO, dibujar_overlay


def frame_negro(alto=480, ancho=640):
    return np.zeros((alto, ancho, 3), dtype=np.uint8)


def det(box, nombre="Ana García", profesion="Arquitecta", reconocido=True):
    return {"box": box, "nombre": nombre, "profesion": profesion, "reconocido": reconocido}


def test_sin_detecciones_solo_dibuja_fps():
    frame = frame_negro()
    salida = dibujar_overlay(frame, [], fps=24.5)
    assert salida is frame
    assert frame[:60, :200].any()
    assert not frame[200:, :].any()


def test_recuadro_verde_si_reconocido():
    frame = frame_negro()
    dibujar_overlay(frame, [det((200, 200, 120, 120))], fps=0)
    assert tuple(frame[260, 200]) == COLOR_RECONOCIDO  # borde izquierdo del recuadro


def test_recuadro_naranja_si_invitado():
    frame = frame_negro()
    dibujar_overlay(frame, [det((200, 200, 120, 120), "Invitado / No registrado", "", False)], fps=0)
    assert tuple(frame[260, 200]) == COLOR_INVITADO


def test_etiqueta_arriba_del_rostro():
    frame = frame_negro()
    dibujar_overlay(frame, [det((200, 250, 120, 120))], fps=0)
    assert frame[150:248, 200:320].any()  # bloque de texto encima de la caja


def test_etiqueta_debajo_si_rostro_pegado_arriba():
    frame = frame_negro()
    dibujar_overlay(frame, [det((300, 0, 120, 120))], fps=0)
    assert frame[125:200, 300:420].any()


@pytest.mark.parametrize("box", [(-20, -20, 100, 100), (600, 440, 100, 100), (0, 0, 640, 480)])
def test_cajas_en_bordes_no_fallan(box):
    frame = frame_negro()
    dibujar_overlay(frame, [det(box, nombre="Nombre Muy Largo De Prueba Con Ñandú")], fps=10)
    assert frame.shape == (480, 640, 3)


def test_fallback_sin_fuente_truetype(monkeypatch):
    monkeypatch.setattr(overlay_renderer, "_fuente", lambda tam: None)
    frame = frame_negro()
    dibujar_overlay(frame, [det((200, 250, 120, 120), nombre="José Peña")], fps=0)
    assert frame[150:248, 200:320].any()
    assert overlay_renderer._sin_acentos("José Peña") == "Jose Pena"
