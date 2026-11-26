"""Comparación coseno de un rostro contra el índice de asistentes."""
from typing import Any, Dict, List

import numpy as np


def buscar_coincidencia(
    embedding: np.ndarray, indice: List[Dict[str, Any]], umbral: float = 0.38
) -> Dict[str, Any]:
    """Devuelve la coincidencia válida más cercana o la etiqueta de desconocido."""
    from src import config

    desconocido = config.ETIQUETA_DESCONOCIDO
    try:
        consulta = np.asarray(embedding, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        consulta = np.empty(0)
    if not consulta.size or not np.all(np.isfinite(consulta)):
        return {"reconocido": False, "nombre": desconocido, "profesion": "", "score": 0.0}

    norma_consulta = float(np.linalg.norm(consulta))
    if norma_consulta == 0:
        return {"reconocido": False, "nombre": desconocido, "profesion": "", "score": 1.0}

    mejor_distancia = float("inf")
    mejor_persona = None
    for persona in indice or []:
        try:
            candidato = np.asarray(persona.get("embedding"), dtype=np.float64).reshape(-1)
            if candidato.shape != consulta.shape or not np.all(np.isfinite(candidato)):
                continue
            norma_candidato = float(np.linalg.norm(candidato))
            if norma_candidato == 0:
                continue
            distancia = 1.0 - float(np.dot(consulta, candidato) / (norma_consulta * norma_candidato))
            distancia = max(0.0, min(2.0, distancia))
        except (TypeError, ValueError, FloatingPointError):
            continue
        if distancia < mejor_distancia:
            mejor_distancia, mejor_persona = distancia, persona

    if mejor_persona is None:
        score = 2.0 if not indice else 1.0
        return {"reconocido": False, "nombre": desconocido, "profesion": "", "score": score}
    if mejor_distancia <= umbral:
        return {
            "reconocido": True, "nombre": mejor_persona.get("nombre", ""),
            "profesion": mejor_persona.get("profesion", ""), "score": 1.0 - mejor_distancia,
        }
    return {"reconocido": False, "nombre": desconocido, "profesion": "", "score": mejor_distancia}
