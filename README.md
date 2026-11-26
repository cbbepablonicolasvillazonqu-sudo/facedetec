# facedetec — Reconocimiento facial en vivo para eventos

Captura de video desde un celular en trípode vía red local/hotspot, reconocimiento facial en tiempo
real en una laptop contra `data/asistentes.csv` + `fotos_registrados/`, y proyección en pantalla
completa (TV/LED por HDMI) del video con recuadro, **nombre** y **profesión** de cada asistente.

```
[Celular + IP Webcam/DroidCam] --RTSP/HTTP--> [Hotspot / Router] --> [Laptop: main.py]
                                                                        |-- data/asistentes.csv
                                                                        |-- fotos_registrados/
                                                                        v
                                                      [Ventana fullscreen] --HDMI--> [TV / LED]
```

Equipo: **Nicolas** (repo, integración, proyección) · **Matt** (captura y registro) ·
**Fabian** (motor de reconocimiento). El reparto detallado está en [PLAN.md](PLAN.md).

## 1. Instalación

Se requiere **Python 3.12** (TensorFlow, que usa DeepFace, no soporta aún 3.13/3.14).

```bash
# Windows (con el Python install manager: py install 3.12)
py -3.12 -m venv .venv
.venv\Scripts\activate
# Linux / macOS
python3.12 -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt -r requirements-dev.txt
```

La primera ejecución descarga los pesos de `Facenet512` (unos 90 MB).

## 2. Datos de asistentes

```bash
cp data/asistentes.example.csv data/asistentes.csv   # Windows: copy
```

- `data/asistentes.csv` (UTF-8): columnas `id,nombre,profesion,foto`.
- `fotos_registrados/`: una foto frontal y bien iluminada por asistente; el nombre del archivo debe
  coincidir con la columna `foto`.
- **Ni `data/asistentes.csv` ni las fotos se suben a GitHub** (son datos personales/biométricos; están
  en `.gitignore`). Cada quien usa sus propias fotos de prueba en local.

Registrar a alguien (módulo de Matt):

```bash
python -m src.registrar "Ana García" "Arquitecta" --img ruta/foto.jpg
python -m src.registrar "Ana García" "Arquitecta" --stream http://192.168.43.1:8080/video
```

## 3. Ejecutar

```bash
# Celular con IP Webcam en la misma red / hotspot
set STREAM_URL=http://192.168.43.1:8080/video      # PowerShell: $env:STREAM_URL="..."
python main.py

# Prueba con la webcam de la laptop
set STREAM_URL=0
python main.py
```

Teclas: `Q`/`ESC` salir · `R` recargar CSV y fotos · `F` alternar pantalla completa.

## 3.1 Base de datos PostgreSQL (opcional)

Por defecto el sistema usa `data/asistentes.csv` y `embeddings_cache.pkl`. Si defines `DATABASE_URL`
usa PostgreSQL ([db/postgres.sql](db/postgres.sql)), que además guarda un **registro de asistencia**:
quién pasó frente a la cámara y a qué hora (una pasada por persona cada `PASADA_COOLDOWN_S` = 60 s).

| Tabla | Contenido |
|---|---|
| `asistentes` | nombre, profesión, archivo de foto, `activo` (desactivar sin borrar historial) |
| `embeddings` | vector facial por asistente, modelo y detector (reemplaza `embeddings_cache.pkl`) |
| `pasadas` | asistente, similitud y hora de cada pasada |
| `resumen_asistencia` (vista) | pasadas, primera y última vez por asistente |

```bash
docker compose up -d                       # PostgreSQL 16 en 127.0.0.1:5432 (crea las tablas)
# PowerShell: $env:DATABASE_URL="postgresql://facedetec:facedetec_local@127.0.0.1:5432/facedetec"
export DATABASE_URL=postgresql://facedetec:facedetec_local@127.0.0.1:5432/facedetec

python -m src.db_postgres importar data/asistentes.csv   # pasar los asistentes del CSV a la base
python main.py                                            # reconoce y anota pasadas
python -m src.db_postgres pasadas                         # resumen de asistencia
python -m src.db_postgres asistentes                      # lista de asistentes
```

- Las fotos siguen en `fotos_registrados/`; la base guarda solo el nombre del archivo.
- La base solo escucha en `127.0.0.1`: los vectores faciales son datos biométricos y no deben salir
  de la laptop. La contraseña por defecto es para uso local; cámbiala con `POSTGRES_PASSWORD`.
- Si la base falla durante el evento, la proyección sigue: solo se avisa por consola.
- Borrar todo (incluido el historial): `docker compose down -v`.

## 4. Flujo de trabajo en GitHub

1. Actualiza `main` antes de empezar:
   ```bash
   git checkout main && git pull origin main
   ```
2. Crea tu rama desde `main`: `feat/<modulo>`, `fix/<problema>` o `test/<modulo>`
   (ej. `feat/stream-capture`, `feat/matcher`).
3. Commits pequeños y claros. **Nunca hagas push directo a `main`.**
4. Abre un Pull Request hacia `main` y pide revisión a los otros dos integrantes. Los tests de
   GitHub Actions deben pasar antes de hacer merge.

## 5. Contratos entre módulos (`src/`)

Respeta estas firmas para que todo encaje sin conflictos. Importa `deepface` **dentro** de las funciones
(no al inicio del archivo) para que los tests corran rápido sin TensorFlow.

| Módulo | Responsable | Firma |
|---|---|---|
| `src/config.py` | Nicolas | Constantes: rutas, `STREAM_URL`, `MODEL_NAME`, `DETECTOR_BACKEND`, `UMBRAL_DISTANCIA_COSENO`, `DETECTAR_CADA_N_FRAMES`, `FULLSCREEN`, `VENTANA_NOMBRE` |
| `src/stream_capture.py` | Matt | `abrir_stream(fuente: str) -> cv2.VideoCapture` · `leer_frame(cap) -> tuple[bool, np.ndarray \| None]` |
| `src/registrar.py` | Matt | `registrar_asistente(nombre, profesion, ruta_foto, csv_path, fotos_dir) -> dict` + CLI `--img` / `--stream` |
| `src/face_detector.py` | Fabian | `detectar_rostros(frame_bgr) -> list[dict]` con `{"box": (x, y, w, h), "crop": np.ndarray, "confidence": float}` |
| `src/embeddings_index.py` | Fabian | `cargar_indice(csv_path, fotos_dir, cache_path) -> list[dict]` con `{"id","nombre","profesion","foto","embedding"}` · `extraer_embedding(rostro_bgr) -> np.ndarray \| None` |
| `src/matcher.py` | Fabian | `buscar_coincidencia(embedding, indice, umbral=0.38) -> {"reconocido": bool, "id", "nombre": str, "profesion": str, "score": float}` (`id` solo si reconocido) |
| `src/overlay_renderer.py` | Nicolas | `dibujar_overlay(frame_bgr, detecciones, fps, estado=None) -> np.ndarray`; cada detección: `{"box", "id", "nombre", "profesion", "reconocido", "score"}` |
| `src/db_postgres.py` | Nicolas | `conectar(url)` · `crear_esquema(conn)` · `cargar_indice_pg(conn, fotos_dir)` (mismo formato que `cargar_indice`) · `registrar_asistente(conn, nombre, profesion, foto)` · `importar_csv(conn, csv_path)` · `RegistroPasadas(conn).anotar(detecciones)` |
| `src/display_output.py` | Nicolas | `configurar_ventana(nombre_ventana, fullscreen=True) -> None` · `mostrar_frame(nombre_ventana, frame_bgr) -> int` |
| `src/performance.py` | Nicolas | `ControladorRitmo(cada_n_frames=3)` con `.debe_procesar() -> bool` y `.tick() -> float` (FPS suavizado) |
| `main.py` | Nicolas | Orquesta el ciclo: captura → detección → embedding → matching → overlay → pantalla |

## 6. Tests

```bash
pytest
# Incluir los tests de PostgreSQL (usan schemas temporales, no tocan tus datos):
docker compose up -d
TEST_DATABASE_URL=postgresql://facedetec:facedetec_local@127.0.0.1:5432/facedetec pytest
```
