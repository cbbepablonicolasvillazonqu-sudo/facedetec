"""Ventana de salida para la TV/LED (HDMI o proyección inalámbrica). Responsable: Nicolas.

Consejo para el evento: pantalla en modo "Extender", arrastrar la ventana a la TV
y luego pulsar F para pasar a pantalla completa en ese monitor.
"""
from typing import Dict

import cv2
import numpy as np

_fullscreen: Dict[str, bool] = {}


def _aplicar_fullscreen(nombre_ventana: str, fullscreen: bool) -> None:
    modo = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
    cv2.setWindowProperty(nombre_ventana, cv2.WND_PROP_FULLSCREEN, modo)
    _fullscreen[nombre_ventana] = fullscreen


def configurar_ventana(nombre_ventana: str, fullscreen: bool = True) -> None:
    cv2.namedWindow(nombre_ventana, cv2.WINDOW_NORMAL)
    _aplicar_fullscreen(nombre_ventana, fullscreen)


def alternar_fullscreen(nombre_ventana: str) -> bool:
    """Cambia entre ventana y pantalla completa. Devuelve el nuevo estado."""
    nuevo = not _fullscreen.get(nombre_ventana, False)
    _aplicar_fullscreen(nombre_ventana, nuevo)
    return nuevo


def mostrar_frame(nombre_ventana: str, frame_bgr: np.ndarray) -> int:
    """Muestra el frame y devuelve la tecla presionada (waitKey(1) & 0xFF; 255 = ninguna)."""
    cv2.imshow(nombre_ventana, frame_bgr)
    return cv2.waitKey(1) & 0xFF


def cerrar_ventanas() -> None:
    cv2.destroyAllWindows()
