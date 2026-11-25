"""Control de ritmo de procesamiento y FPS. Responsable: Nicolas."""


class ControladorRitmo:
    """Decide en qué frames correr detección/reconocimiento y calcula FPS suavizados."""

    def __init__(self, cada_n_frames: int = 3) -> None:
        raise NotImplementedError("Pendiente (Nicolas): src/performance.py -> ControladorRitmo")

    def debe_procesar(self) -> bool:
        raise NotImplementedError

    def tick(self) -> float:
        raise NotImplementedError
