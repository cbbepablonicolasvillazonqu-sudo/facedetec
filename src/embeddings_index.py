"""Extracción de embeddings Facenet512 e índice persistente de asistentes."""
import os
import pickle
import tempfile
from typing import Any, Dict, List, Optional

import numpy as np


def extraer_embedding(rostro_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Calcula un embedding float32 para un recorte de rostro; devuelve None si falla."""
    if not isinstance(rostro_bgr, np.ndarray) or rostro_bgr.size == 0:
        return None
    from deepface import DeepFace
    from src import config

    try:
        resultados = DeepFace.represent(
            img_path=rostro_bgr,
            model_name=config.MODEL_NAME,
            detector_backend="skip",
            enforce_detection=False,
        )
        if not resultados:
            return None
        vector = np.asarray(resultados[0].get("embedding"), dtype=np.float32)
        return vector if vector.ndim == 1 and vector.size and np.all(np.isfinite(vector)) else None
    except Exception as exc:
        print(f"[AVISO] No se pudo calcular un embedding: {exc}")
        return None


def _guardar_cache(cache_path: str, cache: Dict[str, Any]) -> None:
    carpeta = os.path.dirname(os.path.abspath(cache_path))
    os.makedirs(carpeta, exist_ok=True)
    temporal = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=carpeta, delete=False) as archivo:
            temporal = archivo.name
            pickle.dump(cache, archivo, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporal, cache_path)
    finally:
        if temporal and os.path.exists(temporal):
            os.remove(temporal)


def cargar_indice(csv_path: str, fotos_dir: str, cache_path: str) -> List[Dict[str, Any]]:
    """Lee asistentes; reutiliza vectores cuya foto, modelo y detector no cambiaron."""
    import csv
    from src import config

    cache: Dict[str, Any] = {"version": 1, "entries": {}}
    try:
        with open(cache_path, "rb") as archivo:
            cargada = pickle.load(archivo)
        if (isinstance(cargada, dict) and cargada.get("version") == 1
                and isinstance(cargada.get("entries"), dict)):
            cache = cargada
    except (OSError, EOFError, pickle.PickleError, AttributeError, ValueError):
        pass

    try:
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as archivo:
            filas = list(csv.DictReader(archivo))
    except OSError as exc:
        print(f"[AVISO] No se pudo leer el CSV de asistentes '{csv_path}': {exc}")
        return []

    entrada_cache = cache["entries"]
    indice: List[Dict[str, Any]] = []
    modificada = False
    for fila in filas:
        nombre_foto = (fila.get("foto") or "").strip()
        ruta = os.path.join(fotos_dir, nombre_foto)
        try:
            mtime = os.stat(ruta).st_mtime_ns
        except OSError:
            print(f"[AVISO] Foto faltante para {fila.get('nombre', 'asistente')}: {ruta}")
            continue
        clave = os.path.normcase(os.path.abspath(ruta))
        existente = entrada_cache.get(clave, {})
        embedding = None
        if (existente.get("mtime_ns") == mtime
                and existente.get("model") == config.MODEL_NAME
                and existente.get("detector") == config.DETECTOR_BACKEND):
            try:
                embedding = np.asarray(existente.get("embedding"), dtype=np.float32)
                if embedding.ndim != 1 or not embedding.size or not np.all(np.isfinite(embedding)):
                    embedding = None
            except (TypeError, ValueError):
                embedding = None
        if embedding is None:
            try:
                import cv2
                imagen = cv2.imread(ruta)
            except Exception:
                imagen = None
            if imagen is None:
                print(f"[AVISO] No se pudo abrir la foto: {ruta}")
                continue
            embedding = extraer_embedding(imagen)
            if embedding is None:
                continue
            entrada_cache[clave] = {
                "mtime_ns": mtime,
                "model": config.MODEL_NAME,
                "detector": config.DETECTOR_BACKEND,
                "embedding": embedding,
            }
            modificada = True
        indice.append({
            "id": fila.get("id", ""), "nombre": fila.get("nombre", ""),
            "profesion": fila.get("profesion", ""), "foto": nombre_foto,
            "embedding": embedding,
        })
    if modificada:
        try:
            _guardar_cache(cache_path, cache)
        except OSError as exc:
            print(f"[AVISO] No se pudo guardar la caché de embeddings: {exc}")
    return indice
