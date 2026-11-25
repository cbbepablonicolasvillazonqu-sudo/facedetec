"""Captura de video desde el celular (IP Webcam / DroidCam) o webcam. Responsable: Matt."""
from typing import Optional, Tuple

import cv2
import numpy as np


def abrir_stream(fuente: str) -> cv2.VideoCapture:
    """Abre una URL HTTP/RTSP o un índice de cámara ("0").

    Debe fijar CAP_PROP_BUFFERSIZE = 1 para reducir la latencia en red local.
    No lanza excepción si no conecta: quien llama revisa cap.isOpened().
    """
    raise NotImplementedError("Pendiente (Matt): src/stream_capture.py -> abrir_stream")


def leer_frame(cap: cv2.VideoCapture) -> Tuple[bool, Optional[np.ndarray]]:
    """Lee un frame. Devuelve (True, frame_bgr) o (False, None) si falla; nunca lanza excepción."""
    raise NotImplementedError("Pendiente (Matt): src/stream_capture.py -> leer_frame")
