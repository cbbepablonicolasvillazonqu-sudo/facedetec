"""Dibuja detecciones y estado de conexión sobre frames BGR."""
import os
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


def _texto_unicode(frame: np.ndarray, texto: str, origen: tuple, escala: float, color: tuple, grosor: int) -> None:
    """Renderiza con Pillow para conservar tildes y caracteres Unicode."""
    from PIL import Image, ImageDraw, ImageFont

    candidatos = [
        os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "arial.ttf"),
        "DejaVuSans.ttf",
    ]
    fuente = None
    tamano = max(12, int(22 * escala))
    for ruta in candidatos:
        try:
            fuente = ImageFont.truetype(ruta, tamano)
            break
        except OSError:
            continue
    if fuente is None:
        fuente = ImageFont.load_default()
    imagen = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    dibujo = ImageDraw.Draw(imagen)
    dibujo.text(origen, texto, font=fuente, fill=(int(color[2]), int(color[1]), int(color[0])))
    frame[:] = cv2.cvtColor(np.asarray(imagen), cv2.COLOR_RGB2BGR)


def dibujar_overlay(
    frame_bgr: np.ndarray,
    detecciones: List[Dict[str, Any]],
    fps: float,
    estado: Optional[str] = None,
) -> np.ndarray:
    """Dibuja recuadros, etiquetas, FPS y opcionalmente un estado de desconexión."""
    salida = frame_bgr.copy()
    alto, ancho = salida.shape[:2]
    escala = max(0.55, min(1.2, ancho / 1000.0))
    grosor = max(2, round(2 * escala))
    desconocido = "Invitado / No registrado"

    for deteccion in detecciones or []:
        try:
            x, y, w, h = (int(v) for v in deteccion["box"])
        except (KeyError, TypeError, ValueError):
            continue
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(ancho - 1, x + w), min(alto - 1, y + h)
        if x1 <= x0 or y1 <= y0:
            continue
        reconocido = bool(deteccion.get("reconocido", False))
        color = (45, 190, 65) if reconocido else (0, 145, 255)
        cv2.rectangle(salida, (x0, y0), (x1, y1), color, grosor)
        nombre = str(deteccion.get("nombre") or desconocido)
        profesion = str(deteccion.get("profesion") or "")
        etiqueta = nombre if not profesion else f"{nombre} · {profesion}"
        max_chars = max(18, int(ancho / (13 * escala)))
        if len(etiqueta) > max_chars:
            etiqueta = etiqueta[: max_chars - 1] + "…"
        texto_y = max(0, y0 - int(34 * escala))
        texto_alto = int(32 * escala)
        (texto_ancho, _), _ = cv2.getTextSize("Ag", cv2.FONT_HERSHEY_SIMPLEX, 0.65 * escala, 1)
        # El bloque cubre el ancho estimado y queda dentro del frame.
        bloque_x1 = min(ancho - 1, x0 + max(texto_ancho * 7, int(len(etiqueta) * 13 * escala)))
        cv2.rectangle(salida, (x0, texto_y), (bloque_x1, min(alto - 1, texto_y + texto_alto)), (25, 25, 25), -1)
        try:
            _texto_unicode(salida, etiqueta, (x0 + 5, texto_y + 3), escala, (255, 255, 255), 1)
        except ImportError:
            cv2.putText(salida, etiqueta.encode("ascii", "replace").decode(), (x0 + 5, texto_y + 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55 * escala, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(salida, f"FPS: {max(0.0, float(fps)):.1f}", (12, max(24, int(30 * escala))),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7 * escala, (255, 255, 255), 2, cv2.LINE_AA)
    if estado:
        try:
            _texto_unicode(salida, estado, (12, max(42, int(60 * escala))), escala,
                           (0, 0, 255), 1)
        except ImportError:
            cv2.putText(salida, estado.encode("ascii", "replace").decode(), (12, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65 * escala, (0, 0, 255), 2, cv2.LINE_AA)
    return salida
