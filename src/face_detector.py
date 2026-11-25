"""Detección de rostros en un frame. Responsable: Fabian."""
from typing import Any, Dict, List

import numpy as np


def detectar_rostros(frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
    """Detecta rostros con DeepFace.extract_faces (detector config.DETECTOR_BACKEND).

    Descarta confianza < config.CONFIANZA_MIN_DETECCION y caras menores a
    config.TAMANO_MIN_ROSTRO px. Recorta la caja dentro de los bordes del frame.

    Devuelve [{"box": (x, y, w, h), "crop": rostro_bgr, "confidence": float}, ...].
    Importar deepface dentro de la función.
    """
    raise NotImplementedError("Pendiente (Fabian): src/face_detector.py -> detectar_rostros")
