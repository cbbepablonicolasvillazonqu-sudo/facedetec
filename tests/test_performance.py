import pytest

from src.performance import ControladorRitmo


class RelojFalso:
    def __init__(self, paso: float) -> None:
        self.t = 0.0
        self.paso = paso

    def __call__(self) -> float:
        actual = self.t
        self.t += self.paso
        return actual


def test_procesa_primer_frame_y_luego_cada_n():
    ritmo = ControladorRitmo(cada_n_frames=3)
    assert [ritmo.debe_procesar() for _ in range(7)] == [True, False, False, True, False, False, True]


def test_cada_frame_si_n_es_1():
    ritmo = ControladorRitmo(cada_n_frames=1)
    assert all(ritmo.debe_procesar() for _ in range(5))


def test_n_invalido():
    with pytest.raises(ValueError):
        ControladorRitmo(cada_n_frames=0)


def test_fps_estable_converge():
    ritmo = ControladorRitmo(reloj=RelojFalso(paso=0.04))  # 25 FPS
    assert ritmo.tick() == 0.0  # primer tick no tiene referencia
    for _ in range(50):
        fps = ritmo.tick()
    assert fps == pytest.approx(25.0, rel=1e-6)
    assert ritmo.fps == fps


def test_fps_suavizado_no_salta_de_golpe():
    reloj = RelojFalso(paso=0.1)  # 10 FPS
    ritmo = ControladorRitmo(suavizado=0.9, reloj=reloj)
    for _ in range(20):
        ritmo.tick()
    reloj.paso = 0.02  # pasa a 50 FPS (el reloj aplica el nuevo paso desde la siguiente lectura)
    ritmo.tick()
    fps = ritmo.tick()
    assert fps == pytest.approx(0.9 * 10 + 0.1 * 50)  # 14 FPS: sube gradualmente, no salta a 50
