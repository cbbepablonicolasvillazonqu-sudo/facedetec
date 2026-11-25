import numpy as np

from src import display_output
from src.overlay_renderer import dibujar_overlay


def test_overlay_no_modifica_frame_y_dibuja_detecciones():
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    original = frame.copy()
    resultado = dibujar_overlay(frame, [
        {"box": (20, 50, 80, 90), "nombre": "José Núñez", "profesion": "Médico", "reconocido": True},
        {"box": (150, 30, 60, 80), "nombre": "Invitado / No registrado", "profesion": "", "reconocido": False},
    ], 24.5, estado="Sin conexión")
    assert resultado.shape == frame.shape
    assert np.array_equal(frame, original)
    assert np.any(resultado != frame)


def test_salida_configura_fullscreen_y_devuelve_tecla(monkeypatch):
    llamadas = []
    monkeypatch.setattr(display_output.cv2, "namedWindow", lambda *a: llamadas.append(("window", a)))
    monkeypatch.setattr(display_output.cv2, "setWindowProperty", lambda *a: llamadas.append(("property", a)))
    monkeypatch.setattr(display_output.cv2, "imshow", lambda *a: llamadas.append(("show", a[0])))
    monkeypatch.setattr(display_output.cv2, "waitKey", lambda _: ord("f"))
    display_output.configurar_ventana("Evento", True)
    assert display_output.mostrar_frame("Evento", np.zeros((2, 2, 3), dtype=np.uint8)) == ord("f")
    assert llamadas[1][1][2] == display_output.cv2.WINDOW_FULLSCREEN
