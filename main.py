"""Punto de entrada: captura -> detección -> reconocimiento -> overlay -> pantalla. Responsable: Nicolas.

Uso:
    STREAM_URL=http://192.168.43.1:8080/video python main.py
    STREAM_URL=0 python main.py          # webcam de la laptop

Teclas: Q/ESC salir · R recargar CSV y fotos · F alternar pantalla completa.
"""
import os
import time
from typing import Any, Dict, List

import cv2
import numpy as np

from src import config
from src.display_output import alternar_fullscreen, cerrar_ventanas, configurar_ventana, mostrar_frame
from src.embeddings_index import cargar_indice, extraer_embedding
from src.face_detector import detectar_rostros
from src.matcher import buscar_coincidencia
from src.overlay_renderer import dibujar_overlay
from src.performance import ControladorRitmo
from src.stream_capture import abrir_stream, leer_frame

FALLOS_PARA_AVISO = 10  # frames perdidos seguidos antes de mostrar "Sin conexión"
FALLOS_ANTES_DE_RECONECTAR = 30  # ~1 s de frames perdidos
TECLAS_SALIR = (ord("q"), ord("Q"), 27)
ESTADO_SIN_CONEXION = "Sin conexión — reconectando…"


def cargar_base() -> List[Dict[str, Any]]:
    if not os.path.isfile(config.CSV_PATH):
        print(
            f"[AVISO] No existe {config.CSV_PATH}. Copia data/asistentes.example.csv a "
            "data/asistentes.csv y agrega fotos en fotos_registrados/. Todos saldrán como invitados."
        )
        return []
    indice = cargar_indice(config.CSV_PATH, config.FOTOS_DIR, config.CACHE_PATH)
    print(f"[LISTO] {len(indice)} asistentes cargados en memoria.")
    return indice


def reconocer(frame: np.ndarray, indice: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Detecta rostros y los identifica; devuelve detecciones listas para el overlay."""
    detecciones: List[Dict[str, Any]] = []
    for rostro in detectar_rostros(frame):
        embedding = extraer_embedding(rostro["crop"]) if indice else None
        if embedding is None:
            resultado = {"reconocido": False, "nombre": config.ETIQUETA_DESCONOCIDO, "profesion": ""}
        else:
            resultado = buscar_coincidencia(embedding, indice, config.UMBRAL_DISTANCIA_COSENO)
        reconocido = bool(resultado.get("reconocido", False))
        detecciones.append(
            {
                "box": rostro["box"],
                "nombre": resultado.get("nombre") or config.ETIQUETA_DESCONOCIDO,
                "profesion": resultado.get("profesion", "") if reconocido else "",
                "reconocido": reconocido,
            }
        )
    return detecciones


def conectar(fuente: str) -> cv2.VideoCapture:
    cap = abrir_stream(fuente)
    if not cap.isOpened():
        raise RuntimeError(
            f"No se pudo conectar al stream '{fuente}'. Verifica que el celular y la laptop estén "
            "en la misma red/hotspot y que la URL abra en el navegador."
        )
    return cap


def ejecutar() -> None:
    indice = cargar_base()
    cap = conectar(config.STREAM_URL)
    configurar_ventana(config.VENTANA_NOMBRE, config.FULLSCREEN)
    ritmo = ControladorRitmo(config.DETECTAR_CADA_N_FRAMES)
    detecciones: List[Dict[str, Any]] = []
    ultimo_frame = None  # copia limpia del último frame bueno, para el aviso de desconexión
    fallos = 0
    print("Sistema iniciado. [Q]/[ESC] salir · [R] recargar base · [F] pantalla completa.")

    try:
        while True:
            ok, frame = leer_frame(cap)
            if not ok or frame is None:
                fallos += 1
                if fallos >= FALLOS_PARA_AVISO and ultimo_frame is not None:
                    aviso = dibujar_overlay(ultimo_frame.copy(), [], ritmo.fps, estado=ESTADO_SIN_CONEXION)
                    tecla = mostrar_frame(config.VENTANA_NOMBRE, aviso)
                else:
                    tecla = cv2.waitKey(1) & 0xFF
                if tecla in TECLAS_SALIR:
                    break
                if fallos >= FALLOS_ANTES_DE_RECONECTAR:
                    print("[AVISO] Stream perdido, reconectando...")
                    cap.release()
                    time.sleep(1.0)
                    cap = abrir_stream(config.STREAM_URL)
                    fallos = 0
                else:
                    time.sleep(0.03)
                continue
            fallos = 0
            ultimo_frame = frame.copy()

            if ritmo.debe_procesar():
                try:
                    detecciones = reconocer(frame, indice)
                except NotImplementedError:
                    raise
                except Exception as exc:  # un frame malo no debe tumbar la proyección
                    print(f"[ERROR] reconocimiento: {exc}")

            dibujar_overlay(frame, detecciones, ritmo.tick())
            tecla = mostrar_frame(config.VENTANA_NOMBRE, frame)
            if tecla in TECLAS_SALIR:
                break
            if tecla in (ord("r"), ord("R")):
                indice = cargar_base()
            elif tecla in (ord("f"), ord("F")):
                alternar_fullscreen(config.VENTANA_NOMBRE)
    finally:
        cap.release()
        cerrar_ventanas()


if __name__ == "__main__":
    ejecutar()
