"""Captura de video desde un celular o una cámara local."""

from typing import Optional, Tuple

import cv2
import numpy as np


def abrir_stream(fuente: str) -> cv2.VideoCapture:
    """Abre una URL HTTP/RTSP o un índice de cámara como '0'.

    Si no se puede conectar, quien llama debe comprobar cap.isOpened().
    """
    fuente = str(fuente).strip()
    origen = int(fuente) if fuente.isdigit() else fuente

    try:
        cap = cv2.VideoCapture(origen)
    except cv2.error as error:
        print(f"[ERROR] No se pudo abrir la fuente: {error}")
        return cv2.VideoCapture()

    # El soporte de esta propiedad depende de la cámara y del backend.
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except cv2.error:
        pass

    if not cap.isOpened():
        print("[AVISO] No se pudo conectar con la fuente de video.")

    return cap


def leer_frame(cap: cv2.VideoCapture) -> Tuple[bool, Optional[np.ndarray]]:
    """Devuelve (True, frame BGR) o (False, None) si falla la lectura."""
    try:
        if cap is None or not cap.isOpened():
            return False, None

        correcto, frame = cap.read()

        if not correcto or frame is None:
            return False, None

        return True, frame

    except (cv2.error, AttributeError):
        return False, None
