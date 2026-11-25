"""Control de ritmo de procesamiento y FPS. Responsable: Nicolas."""
import time
from typing import Callable


class ControladorRitmo:
    """Decide en qué frames correr detección/reconocimiento y calcula FPS suavizados.

    El reconocimiento es lo caro; el video se sigue mostrando en todos los frames
    con las últimas detecciones, así la imagen en la TV se ve fluida.
    """

    def __init__(
        self,
        cada_n_frames: int = 3,
        suavizado: float = 0.9,
        reloj: Callable[[], float] = time.perf_counter,
    ) -> None:
        if cada_n_frames < 1:
            raise ValueError("cada_n_frames debe ser >= 1")
        if not 0.0 <= suavizado < 1.0:
            raise ValueError("suavizado debe estar en [0, 1)")
        self.cada_n_frames = cada_n_frames
        self._suavizado = suavizado
        self._reloj = reloj
        self._frame = 0
        self._t_anterior = None
        self._fps = 0.0

    def debe_procesar(self) -> bool:
        """True en el primer frame y luego cada `cada_n_frames` frames."""
        procesar = self._frame % self.cada_n_frames == 0
        self._frame += 1
        return procesar

    def tick(self) -> float:
        """Registra un frame mostrado y devuelve el FPS suavizado (media exponencial)."""
        ahora = self._reloj()
        if self._t_anterior is not None:
            instantaneo = 1.0 / max(ahora - self._t_anterior, 1e-6)
            if self._fps == 0.0:
                self._fps = instantaneo
            else:
                self._fps = self._suavizado * self._fps + (1 - self._suavizado) * instantaneo
        self._t_anterior = ahora
        return self._fps

    @property
    def fps(self) -> float:
        return self._fps
