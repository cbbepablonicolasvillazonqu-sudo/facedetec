"""Configuración central del sistema. Responsable: Nicolas.

Los valores que cambian entre la laptop de pruebas y el evento se pueden
sobrescribir con variables de entorno, sin tocar el código.
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "asistentes.csv")
FOTOS_DIR = os.path.join(BASE_DIR, "fotos_registrados")
CACHE_PATH = os.path.join(BASE_DIR, "embeddings_cache.pkl")

# Base de datos (opcional). Vacío = modo CSV (data/asistentes.csv + embeddings_cache.pkl).
# Con PostgreSQL (docker compose up -d):
#   DATABASE_URL=postgresql://facedetec:facedetec_local@127.0.0.1:5432/facedetec
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
PASADA_COOLDOWN_S = int(os.environ.get("PASADA_COOLDOWN_S", "60"))  # 1 pasada por persona cada N s

# Fuente de video:
# - IP Webcam (Android): "http://<IP_CELULAR>:8080/video"
# - RTSP: "rtsp://<IP_CELULAR>:8554/h264"
# - Webcam local de prueba: "0"
STREAM_URL = os.environ.get("STREAM_URL", "http://192.168.43.1:8080/video")

# Reconocimiento
MODEL_NAME = "Facenet512"
DETECTOR_BACKEND = os.environ.get("DETECTOR_BACKEND", "opencv")  # "retinaface" = más preciso, más lento
UMBRAL_DISTANCIA_COSENO = float(os.environ.get("UMBRAL", "0.38"))  # menor = más estricto
CONFIANZA_MIN_DETECCION = 0.5
TAMANO_MIN_ROSTRO = 35  # px
ETIQUETA_DESCONOCIDO = "Invitado / No registrado"

# Rendimiento y pantalla
DETECTAR_CADA_N_FRAMES = int(os.environ.get("CADA_N_FRAMES", "3"))
FULLSCREEN = os.environ.get("FULLSCREEN", "1") != "0"
VENTANA_NOMBRE = "Evento - Identificacion en Vivo"
