"""Dibujo de recuadros, nombre, profesión y FPS sobre el frame. Responsable: Nicolas.

cv2.putText no soporta tildes ni ñ, así que los textos se dibujan con Pillow y una
fuente TrueType. Si no hay fuente disponible se usa cv2 quitando los acentos.
"""
import unicodedata
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover
    Image = ImageDraw = ImageFont = None

# Colores en BGR
COLOR_RECONOCIDO = (80, 200, 0)
COLOR_INVITADO = (0, 165, 255)
COLOR_FONDO = (20, 20, 20)
COLOR_NOMBRE = (255, 255, 255)
COLOR_PROFESION = (255, 235, 130)
OPACIDAD_FONDO = 0.75

FUENTES = (
    "arialbd.ttf",
    "arial.ttf",
    "segoeui.ttf",
    "DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
)

Texto = Tuple[str, Tuple[int, int], int, Tuple[int, int, int]]  # (texto, (x, y_sup), tam_px, color_bgr)


@lru_cache(maxsize=32)
def _fuente(tam_px: int) -> Optional[Any]:
    if ImageFont is None:
        return None
    for nombre in FUENTES:
        try:
            return ImageFont.truetype(nombre, tam_px)
        except OSError:
            continue
    return None


def _sin_acentos(texto: str) -> str:
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")


def _escala_cv2(tam_px: int) -> float:
    return tam_px / 30.0


def _medir(texto: str, tam_px: int) -> int:
    """Ancho en píxeles del texto."""
    fuente = _fuente(tam_px)
    if fuente is not None:
        izq, _, der, _ = fuente.getbbox(texto)
        return int(der - izq)
    escala = _escala_cv2(tam_px)
    (ancho, _), _ = cv2.getTextSize(_sin_acentos(texto), cv2.FONT_HERSHEY_SIMPLEX, escala, 2)
    return int(ancho)


def _escribir(frame: np.ndarray, textos: Sequence[Texto]) -> None:
    """Dibuja todos los textos en una sola pasada (una conversión a PIL por frame)."""
    if not textos:
        return
    if _fuente(textos[0][2]) is not None:
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        dibujo = ImageDraw.Draw(img)
        for texto, (x, y), tam, (b, g, r) in textos:
            dibujo.text((x, y), texto, font=_fuente(tam), fill=(r, g, b))
        frame[:] = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        return
    for texto, (x, y), tam, color in textos:
        escala = _escala_cv2(tam)
        cv2.putText(
            frame, _sin_acentos(texto), (x, y + int(tam * 0.8)),
            cv2.FONT_HERSHEY_SIMPLEX, escala, color, 2, cv2.LINE_AA,
        )


def _fondo_semitransparente(frame: np.ndarray, izq: int, sup: int, der: int, inf: int) -> None:
    roi = frame[sup:inf, izq:der]
    if roi.size == 0:
        return
    fondo = np.full_like(roi, COLOR_FONDO)
    frame[sup:inf, izq:der] = cv2.addWeighted(fondo, OPACIDAD_FONDO, roi, 1 - OPACIDAD_FONDO, 0)


def dibujar_overlay(frame_bgr: np.ndarray, detecciones: List[Dict[str, Any]], fps: float) -> np.ndarray:
    """Dibuja sobre `frame_bgr` (in situ) y lo devuelve.

    Cada detección: {"box": (x, y, w, h), "nombre": str, "profesion": str, "reconocido": bool}.
    Recuadro verde = reconocido, naranja = invitado. La etiqueta va encima del rostro,
    o debajo si no cabe arriba. El tamaño del texto escala con la resolución del frame.
    """
    alto_img, ancho_img = frame_bgr.shape[:2]
    k = max(0.6, alto_img / 720.0)
    tam_nombre = int(round(26 * k))
    tam_profesion = int(round(21 * k))
    pad = int(round(10 * k))
    grosor = max(2, int(round(2 * k)))
    textos: List[Texto] = []

    for det in detecciones:
        x, y, w, h = (int(v) for v in det["box"])
        reconocido = bool(det.get("reconocido", False))
        nombre = str(det.get("nombre", "") or "")
        profesion = str(det.get("profesion", "") or "")
        color = COLOR_RECONOCIDO if reconocido else COLOR_INVITADO

        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, grosor)

        lineas = [(nombre, tam_nombre, COLOR_NOMBRE)]
        if profesion:
            lineas.append((profesion, tam_profesion, COLOR_PROFESION))
        alto_linea = [int(t * 1.3) for _, t, _ in lineas]
        ancho_bloque = max(w, max(_medir(txt, t) for txt, t, _ in lineas) + 2 * pad)
        alto_bloque = sum(alto_linea) + pad

        izq = min(max(0, x), max(0, ancho_img - ancho_bloque))
        sup = y - alto_bloque - grosor
        if sup < 0:  # no cabe arriba: debajo del rostro
            sup = min(y + h + grosor, alto_img - alto_bloque)
        sup = max(0, sup)
        der = min(ancho_img, izq + ancho_bloque)
        inf = min(alto_img, sup + alto_bloque)

        _fondo_semitransparente(frame_bgr, izq, sup, der, inf)
        cv2.rectangle(frame_bgr, (izq, sup), (izq + max(3, grosor * 2), inf), color, -1)

        ty = sup + pad // 2
        for (txt, tam, col), alto in zip(lineas, alto_linea):
            textos.append((txt, (izq + pad, ty), tam, col))
            ty += alto

    _escribir(frame_bgr, textos)

    cv2.putText(
        frame_bgr, f"FPS: {fps:.1f}", (int(15 * k), int(30 * k)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7 * k, (220, 220, 220), grosor, cv2.LINE_AA,
    )
    return frame_bgr
