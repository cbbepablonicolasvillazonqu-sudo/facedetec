# Plan de implementación — equipo de 3

Objetivo: el día del evento, una cámara (celular en trípode) en el punto de paso envía video a la
laptop, que reconoce a cada asistente registrado y proyecta en la TV/LED el video con su recuadro,
nombre y profesión.

## Reparto

| Persona | Rol | Entregables |
|---|---|---|
| **Nicolas** | Líder de repo · integración · proyección | Estructura del repo, README, CI, `src/config.py`, `src/overlay_renderer.py`, `src/display_output.py`, `src/performance.py`, `main.py`, `tests/test_overlay.py`, `tests/test_performance.py` |
| **Matt** | Captura móvil · red · registro | `src/stream_capture.py`, `src/registrar.py` (CLI `--img` / `--stream`), `docs/checklist_evento.md`, `tests/test_registrar.py` |
| **Fabian** | Motor de reconocimiento | `src/face_detector.py`, `src/embeddings_index.py` (caché `embeddings_cache.pkl` invalidada por `mtime` de cada foto), `src/matcher.py` (distancia coseno), `tests/test_matcher.py` |

Las firmas exactas de cada función están en la sección 5 del [README](README.md). Cada módulo ya
existe como esqueleto con `NotImplementedError`: solo hay que reemplazar el cuerpo.

## Tareas por persona

### Nicolas — rama `feat/integracion-proyeccion`
- [x] Crear repo, README, PLAN, `.gitignore`, `requirements*.txt`, plantilla de PR y GitHub Actions.
- [x] Esqueletos de todos los módulos con sus contratos.
- [ ] `config.py` con todos los parámetros (y `STREAM_URL` desde variable de entorno).
- [ ] `overlay_renderer.py`: recuadro verde (reconocido) / naranja (invitado), bloque semitransparente con nombre + profesión, FPS.
- [ ] `display_output.py`: ventana fullscreen para HDMI / proyección; alternar con `F`.
- [ ] `performance.py`: procesar 1 de cada N frames y FPS suavizado.
- [ ] `main.py`: bucle completo, teclas `Q`/`ESC`/`R`/`F`, reconexión si se corta el stream.
- [ ] Revisar y hacer merge de los PR de Matt y Fabian.

### Matt — ramas `feat/stream-capture` y `feat/registrar`
- [ ] `abrir_stream`: URL HTTP/RTSP o índice de cámara (`"0"`), `CAP_PROP_BUFFERSIZE = 1` para bajar latencia.
- [ ] `leer_frame`: devuelve `(False, None)` si falla, sin lanzar excepción.
- [ ] `registrar_asistente`: copia la foto a `fotos_registrados/` con nombre limpio (`ana_garcia_3.jpg`), agrega fila al CSV con id siguiente, crea el CSV con encabezado si no existe.
- [ ] CLI `python -m src.registrar "Nombre" "Profesión" --img foto.jpg | --stream URL` (captura con ESPACIO, cancela con Q).
- [ ] `docs/checklist_evento.md`: configurar hotspot, IP Webcam/DroidCam, probar URL en VLC/navegador, resolución recomendada (720p).
- [ ] Tests con carpetas temporales (`tmp_path`).

### Fabian — ramas `feat/face-detector` y `feat/matcher`
- [ ] `detectar_rostros`: `DeepFace.extract_faces(..., enforce_detection=False)`, descartar confianza < 0.5 y caras < 35 px, recortar dentro de los bordes.
- [ ] `extraer_embedding`: `DeepFace.represent(..., model_name="Facenet512", detector_backend="skip")`.
- [ ] `cargar_indice`: leer CSV, reutilizar embeddings de la caché si el `mtime` no cambió, avisar fotos faltantes.
- [ ] `buscar_coincidencia`: distancia coseno contra todo el índice, umbral 0.38; si no hay match → `"Invitado / No registrado"`.
- [ ] Tests del matcher con vectores sintéticos (sin DeepFace).

## Hitos

1. **Día 1** — Nicolas sube el esqueleto; cada uno clona, crea su venv 3.12 y su rama.
2. **Semana 1** — PRs de módulos individuales con tests verdes.
3. **Integración** — `python main.py` con webcam local (`STREAM_URL=0`) reconociendo a los 3 del equipo.
4. **Prueba de red** — mismo flujo con el celular por hotspot.
5. **Ensayo general (30 min, los 3)** — en el lugar real con TV/LED por HDMI: calibrar distancia de
   cámara, iluminación, `UMBRAL_DISTANCIA_COSENO` y `DETECTAR_CADA_N_FRAMES`.

## Checklist del ensayo general
- [ ] Laptop conectada al hotspot del celular o a un router propio (no al Wi‑Fi del recinto).
- [ ] URL del stream abre en el navegador/VLC de la laptop.
- [ ] Pantalla en modo **Extender**; ventana arrastrada a la TV antes de pulsar `F`.
- [ ] Fotos y CSV de los asistentes cargados (`[LISTO] N asistentes cargados`).
- [ ] FPS ≥ 10 en pantalla; si no, subir `DETECTAR_CADA_N_FRAMES` o bajar resolución del celular.
- [ ] Probar con alguien no registrado: debe salir "Invitado / No registrado" en naranja.
