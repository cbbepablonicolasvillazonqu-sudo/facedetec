"""Ventana de salida para monitor, TV o proyección."""
import cv2
import numpy as np


def configurar_ventana(nombre_ventana: str, fullscreen: bool = True) -> None:
    """Crea una ventana redimensionable y activa o desactiva pantalla completa."""
    cv2.namedWindow(nombre_ventana, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        nombre_ventana,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
    )


def mostrar_frame(nombre_ventana: str, frame_bgr: np.ndarray) -> int:
    """Muestra el frame y devuelve la tecla presionada (waitKey(1) & 0xFF)."""
    cv2.imshow(nombre_ventana, frame_bgr)
    return cv2.waitKey(1) & 0xFF
