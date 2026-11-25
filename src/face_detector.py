"""Detección y recorte de rostros en un frame BGR."""
from typing import Any, Dict, List

import numpy as np


def detectar_rostros(frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
    """Extrae rostros y devuelve sus cajas (x, y, ancho, alto) y recortes BGR."""
    if not isinstance(frame_bgr, np.ndarray) or frame_bgr.ndim < 2 or frame_bgr.size == 0:
        return []

    from deepface import DeepFace
    from src import config

    alto, ancho = frame_bgr.shape[:2]
    try:
        resultados = DeepFace.extract_faces(
            img_path=frame_bgr,
            detector_backend=config.DETECTOR_BACKEND,
            enforce_detection=False,
            align=True,
        )
    except Exception as exc:
        print(f"[AVISO] No se pudieron detectar rostros: {exc}")
        return []

    detecciones: List[Dict[str, Any]] = []
    for resultado in resultados or []:
        area = resultado.get("facial_area") or {}
        try:
            x, y = int(area.get("x", 0)), int(area.get("y", 0))
            w, h = int(area.get("w", 0)), int(area.get("h", 0))
            confianza = float(area.get("confidence", resultado.get("confidence", 0.0)))
        except (TypeError, ValueError):
            continue
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(ancho, x + w), min(alto, y + h)
        w, h = x1 - x0, y1 - y0
        if confianza < config.CONFIANZA_MIN_DETECCION or w < config.TAMANO_MIN_ROSTRO or h < config.TAMANO_MIN_ROSTRO:
            continue
        crop = frame_bgr[y0:y1, x0:x1].copy()
        if crop.size:
            detecciones.append({"box": (x0, y0, w, h), "crop": crop, "confidence": confianza})
    return detecciones
