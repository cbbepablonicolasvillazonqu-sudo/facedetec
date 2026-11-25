"""Índice de vectores faciales de los asistentes registrados. Responsable: Fabian."""
from typing import Any, Dict, List, Optional

import numpy as np


def extraer_embedding(rostro_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Vector facial (float32) de un recorte ya detectado.

    DeepFace.represent(model_name=config.MODEL_NAME, detector_backend="skip",
    enforce_detection=False). Devuelve None si falla. Importar deepface dentro de la función.
    """
    raise NotImplementedError("Pendiente (Fabian): src/embeddings_index.py -> extraer_embedding")


def cargar_indice(csv_path: str, fotos_dir: str, cache_path: str) -> List[Dict[str, Any]]:
    """Lee el CSV y calcula (o reutiliza de la caché pickle) el embedding de cada foto.

    La caché se invalida por foto si cambia su mtime, el modelo o el detector.
    Avisa por consola de fotos faltantes y las omite.

    Devuelve [{"id", "nombre", "profesion", "foto", "embedding": np.ndarray}, ...].
    """
    raise NotImplementedError("Pendiente (Fabian): src/embeddings_index.py -> cargar_indice")
