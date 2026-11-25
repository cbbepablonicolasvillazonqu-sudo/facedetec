"""Dibujo de recuadros, nombre, profesión y FPS sobre el frame. Responsable: Nicolas."""
from typing import Any, Dict, List

import numpy as np


def dibujar_overlay(frame_bgr: np.ndarray, detecciones: List[Dict[str, Any]], fps: float) -> np.ndarray:
    """Cada detección: {"box": (x, y, w, h), "nombre": str, "profesion": str, "reconocido": bool}."""
    raise NotImplementedError("Pendiente (Nicolas): src/overlay_renderer.py -> dibujar_overlay")
