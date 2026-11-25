"""Ventana de salida para la TV/LED (HDMI o proyección inalámbrica). Responsable: Nicolas."""
import numpy as np


def configurar_ventana(nombre_ventana: str, fullscreen: bool = True) -> None:
    raise NotImplementedError("Pendiente (Nicolas): src/display_output.py -> configurar_ventana")


def mostrar_frame(nombre_ventana: str, frame_bgr: np.ndarray) -> int:
    """Muestra el frame y devuelve la tecla presionada (waitKey(1) & 0xFF)."""
    raise NotImplementedError("Pendiente (Nicolas): src/display_output.py -> mostrar_frame")
