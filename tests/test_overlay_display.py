import numpy as np

from src import display_output
from src.overlay_renderer import dibujar_overlay


DETECCIONES = [
    {"box": (20, 50, 80, 90), "nombre": "José Núñez", "profesion": "Médico", "reconocido": True},
    {"box": (150, 30, 60, 80), "nombre": "Invitado / No registrado", "profesion": "", "reconocido": False},
]


def test_overlay_dibuja_in_situ_las_detecciones():
    # main.py muestra el mismo frame que le pasa al overlay, así que debe dibujar in situ.
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    original = frame.copy()
    resultado = dibujar_overlay(frame, DETECCIONES, 24.5)
    assert resultado is frame
    assert resultado.shape == original.shape
    assert np.any(frame != original)


def test_overlay_muestra_estado_en_franja_superior():
    sin_estado = dibujar_overlay(np.zeros((180, 320, 3), dtype=np.uint8), [], 0.0)
    con_estado = dibujar_overlay(np.zeros((180, 320, 3), dtype=np.uint8), [], 0.0, estado="Sin conexión")
    franja = slice(0, 15), slice(150, 320)  # zona de la franja lejos del texto de FPS
    assert not sin_estado[franja].any()
    assert con_estado[franja].any()
    assert con_estado[franja][..., 2].mean() > con_estado[franja][..., 0].mean()  # predomina el rojo (BGR)


def test_salida_configura_fullscreen_y_devuelve_tecla(monkeypatch):
    llamadas = []
    monkeypatch.setattr(display_output.cv2, "namedWindow", lambda *a: llamadas.append(("window", a)))
    monkeypatch.setattr(display_output.cv2, "setWindowProperty", lambda *a: llamadas.append(("property", a)))
    monkeypatch.setattr(display_output.cv2, "imshow", lambda *a: llamadas.append(("show", a[0])))
    monkeypatch.setattr(display_output.cv2, "waitKey", lambda _: ord("f"))
    display_output.configurar_ventana("Evento", True)
    assert display_output.mostrar_frame("Evento", np.zeros((2, 2, 3), dtype=np.uint8)) == ord("f")
    assert llamadas[1][1][2] == display_output.cv2.WINDOW_FULLSCREEN
