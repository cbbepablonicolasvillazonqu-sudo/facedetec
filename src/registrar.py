"""Registro de asistentes en data/asistentes.csv + fotos_registrados/. Responsable: Matt.

Uso previsto:
    python -m src.registrar "Nombre Apellido" "Profesión" --img ruta/foto.jpg
    python -m src.registrar "Nombre Apellido" "Profesión" --stream http://192.168.43.1:8080/video
"""
from typing import Any, Dict


def registrar_asistente(
    nombre: str, profesion: str, ruta_foto: str, csv_path: str, fotos_dir: str
) -> Dict[str, Any]:
    """Copia la foto a fotos_dir con nombre limpio y agrega la fila al CSV.

    Crea el CSV con encabezado `id,nombre,profesion,foto` (UTF-8) si no existe.
    Devuelve {"id": int, "nombre": str, "profesion": str, "foto": str}.
    """
    raise NotImplementedError("Pendiente (Matt): src/registrar.py -> registrar_asistente")


def main() -> None:
    """CLI con modos --img y --stream (captura con ESPACIO, cancela con Q)."""
    raise NotImplementedError("Pendiente (Matt): src/registrar.py -> main (CLI)")


if __name__ == "__main__":
    main()
