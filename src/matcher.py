"""Comparación de un rostro contra el índice de asistentes. Responsable: Fabian."""
from typing import Any, Dict, List

import numpy as np


def buscar_coincidencia(
    embedding: np.ndarray, indice: List[Dict[str, Any]], umbral: float = 0.38
) -> Dict[str, Any]:
    """Busca la persona con menor distancia coseno.

    Si la mejor distancia <= umbral:
        {"reconocido": True, "nombre": ..., "profesion": ..., "score": 1 - distancia}
    Si no (o índice vacío):
        {"reconocido": False, "nombre": config.ETIQUETA_DESCONOCIDO, "profesion": "", "score": distancia}
    """
    raise NotImplementedError("Pendiente (Fabian): src/matcher.py -> buscar_coincidencia")
