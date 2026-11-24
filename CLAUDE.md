# Sistema Comedor UNPRG: reconocimiento facial

Kiosco de asistencia para el comedor universitario. El comensal se para frente a una **webcam fija**, el sistema verifica que sea una persona real (liveness), lo identifica y registra su asistencia. Objetivo del usuario: **rápido (≈2 s), sin pedirle nada a la persona y lo más cerca posible del 100%**. Solo se usará cámara web, no IR ni 3D.

## Reglas de trabajo

- **Responder siempre en español.** El usuario lo pidió varias veces.
- Dar conclusiones precisas con porcentajes medidos y explicar la estrategia, no solo los números.
- **Nunca subir `.env`** (credenciales de la BD y `SECRET_KEY`). La plantilla es `.env.example`.
- **Nunca subir `data/`**: contiene fotos de rostros de la calibración y el log de liveness.
- No hacer commit ni push sin que el usuario lo pida.
- `origin/2026` es una rama paralela de otra línea de trabajo (GPU, MiniFASNet y reto de giro de cabeza). No está integrada en `master` y no se debe mezclar sin preguntar.

## Stack

| Parte | Tecnología |
|---|---|
| Servidor | FastAPI + Uvicorn, Python **3.10** |
| BD | **PostgreSQL 17 + pgvector 0.8.0**, SQLAlchemy async (asyncpg) |
| Vistas | Jinja2 (`templates/`), cámara desde el navegador |
| IA | onnxruntime 1.16.3 en **CPU**, InsightFace 0.7.3, OpenCV 4.10, numpy 1.24.3 |
| Logs | loguru |

## Puesta en marcha (Windows)

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env        # y completar la contraseña de postgres y SECRET_KEY

# pgvector: compilarlo con Visual Studio C++ e instalarlo en PostgreSQL 17 (pide administrador)
powershell -ExecutionPolicy Bypass -File scripts\instalar_pgvector.ps1

# Crear la BD comedor_db (CREATE DATABASE) y luego el esquema y los datos iniciales:
psql -U postgres -d comedor_db -f scripts\crear_bd.sql

.\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
```

- `crear_bd.sql` crea 7 tablas: usuarios, comensales, embeddings `vector(512)`, asistencias, marcaciones, intentos_acceso y configuracion. También carga los horarios de comida y el usuario **admin / admin123**, con hash SHA-256 (ver `generar_hash.py`).
- `venv/` es un entorno viejo de otra PC y no sirve. El bueno es `.venv/`.
- Los modelos `buffalo_l` de InsightFace se descargan solos a `~/.insightface`. Los de anti-spoofing están en `models/antispoof/`, y el servicio verifica su SHA-256 al cargarlos.

## Modelos

| Modelo | Archivo | Uso |
|---|---|---|
| SCRFD-10G | `det_10g.onnx` | Detección de la cara |
| 106 puntos 2D | `2d106det.onnx` | Puntos faciales para la capa de profundidad |
| ArcFace ResNet-50 (WebFace600K) | `w600k_r50.onnx` | Huella de 512 dimensiones. Distancia coseno < 0.45 para match y < 0.50 para duplicado |
| MiniFASNetV2 (escala 2.7) + MiniFASNetV1SE (escala 4.0) | `models/antispoof/*.onnx` | Liveness por textura. Silent-Face, Apache-2.0 |

- MiniFASNet recibe un recorte BGR de **80×80 con píxeles 0–255 sin normalizar**. Si se divide entre 255, la salida es basura.
- Tiene 3 clases; la clase 1 es "real". Se aplica softmax en cada modelo y se promedian los dos.
- El recorte con contexto se desplaza en los bordes de la imagen, en vez de achicarse, igual que en `CropImage` de Silent-Face.
- `FaceAnalysis` carga solo `detection`, `recognition` y `landmark_2d_106`. Quitar los otros módulos ahorra ~35% de tiempo. `DET_SIZE=480`.

## Flujo del kiosco (`/` = `templates/index.html` → `app/routes/rafaga.py`)

El kiosco es la página principal. `/reconocimiento` solo redirige a `/`. El flujo viejo de una foto (`/api/reconocer` con votación) se eliminó: todo pasa por la ráfaga.


0. **Espera.** Cada 0.7 s el kiosco consulta `/api/validar_rostro` con `kiosco: true`. Si no hay cara, `referencia_fondo.actualizar()` aprende el fondo vacío.
1. **Ráfaga.** Toma 5 fotos cada 180 ms con la exposición y el balance de blancos bloqueados y las envía a `POST /api/reconocer_rafaga`. En modo luz, antes pide una secuencia de colores de un solo uso a `POST /api/liveness/desafio`.
2. **Calidad.** SCRFD y los 106 puntos. Se necesitan ≥3 fotos válidas y brillo ≥60. Si falla: `NO_FACE`, `TOO_FAR`, `LOW_QUALITY` o `LOW_LIGHT`.
3. **MiniFASNet.** Pasa con nota ≥0.50. Si falla: `LIVENESS_FAILED`.
4. **Capas físicas** (`app/services/multiframe_liveness.py`):
   - **Profundidad.** Calcula el residuo de la homografía de los 106 puntos dividido por el movimiento. Es 3D si la relación es ≥0.35 y el movimiento ≥12. Una foto quieta tiene un ruido de ≤7.7.
   - **Fondo de referencia.** Compara por textura (ZNCC en bloques de 16 px) el anillo alrededor de la cabeza con el fondo memorizado. Es real si coincide ≥30% de los bloques. Los ataques dan ≤0.25 y la gente real ≥0.39.
   - **Fondo quieto (lámina).** Flujo óptico LK con RANSAC: detecta si lo que rodea la cara se mueve en bloque. **Solo registra** (`FONDO_LAMINA_DECIDE=False`), porque con 2 personas en la fila da falsas alarmas.
   - **Luz.** Correlación entre la cromaticidad de la piel y los colores mostrados, ≥0.5. Los colores cambian a menos de 3 por segundo (WCAG).
   - **Modo sombra.** Con `LIVENESS_CAPAS_ACTIVAS=False` las capas solo escriben en `data/liveness_log.jsonl` y no rechazan. Hay que activarlas después de calibrar.
5. **Identidad.** ArcFace contra el caché en memoria (`cache_service`). La misma persona tiene que salir en ≥60% de las fotos. Si no: `NO_MATCH`.
6. **Cooldown.** `_cooldowns` vive en `app/routes/reconocimiento.py` y lo consulta la ráfaga. Si ya marcó: `COOLDOWN`. Solo se activa en `_confirmar_asistencia` cuando la asistencia se guarda; un rechazo (fuera de horario, comida ya consumida) no bloquea.
7. **Asistencia.** `_confirmar_asistencia()` en `reconocimiento.py` responde `MATCH`, con mensaje y voz.

- **Decisión de liveness:** `fas_ok and (capas_ok or not LIVENESS_CAPAS_ACTIVAS)`, con `capas_ok = prof_ok and fondo_ok and ref_ok and luz_ok`.
- **Calibración:** la página `/calibracion` solo funciona desde localhost, porque guarda rostros en `data/calibracion/`. Tiene un botón "Memorizar fondo", etiquetas real/foto/celular/video y graba con la barra espaciadora.
- **Diagrama visual:** `docs/flujo-kiosco.html`.

Todos los umbrales están en `app/core/config.py` y comentados.

## Resultados medidos (noviembre 2026)

Las pruebas usaron datos públicos y ataques simulados. **Todavía falta confirmarlos con la cámara real** en `/calibracion`.

| Prueba | Resultado |
|---|---|
| CelebA-Spoof (1,779 imágenes), liveness anterior LBP+FFT+color | 100% de ataques pasan, AUC 0.377 (peor que el azar) |
| CelebA-Spoof, MiniFASNet | 51.8% de ataques pasan, 8.5% de reales rechazados, AUC 0.821 |
| YOLOX para detectar el celular o la pantalla | 51.2% de ataques pasan: casi sin mejora, descartado |
| Profundidad (CREMA-D, 91 personas) | 98.9% de reales detectados como 3D. **Fotos: pasa el 2.2%.** Video en pantalla: pasa el 82.4% |
| Fondo de referencia (91 personas) | Real 100%, con gente en la fila 97.8–100%. **Videos y fotos en celular o tablet: 0% pasan** |
| De punta a punta, otra pared (25 personas) | Real 96%. Video en la mano bloqueado al 96% |
| Adversario con el mismo fondo (40 personas) | Fondo parecido o mismo fondo en un celular en la mano: 0%. **Pantalla que tapa toda la cámara, alineada a ≤4 px: pasa el 100%** (límite conocido; lo cubriría la capa de luz) |
| Velocidad | 334 → 145–177 ms por foto. La decisión bajó de ~4.5 s a **~2 s** (~2.8 s en modo luz) |
| Luz del día a la noche | Con 8% de luz ArcFace ya no reconoce (distancia 0.469). Por eso se agregó el aviso `LOW_LIGHT` |

### Prueba de exactitud con 94 registrados y 56 desconocidos (24 de noviembre de 2026)

El detalle completo está en **`docs/pruebas-exactitud.md`**. Los scripts están en `scripts/pruebas/`.

| Métrica | Resultado |
|---|---|
| Reconocimiento correcto en el kiosco, con ráfaga y otro video del mismo actor | **94.5%**. Todo el 5.5% restante falló por calidad (`det_score < 0.80`) |
| Reconocimiento correcto en fotos de otros años (LFW) | 94.8% |
| Confusiones con otra persona y desconocidos aceptados | **0%** (121 intentos de desconocidos; 172,396 pares de personas distintas, el más parecido a 0.695) |
| Galería de 94 a 1,928 personas | Sin cambios |
| Anti-duplicado | 15 de 15 bloqueados |
| MiniFASNet en CelebA-Spoof | AUC 0.83; con umbral 0.50 pasa el 53.7% de los ataques |
| Foto en la mano | **Pasa el 100% en modo sombra** y el 0% con las capas activas (las personas reales pasan el 98.2%) |

- **Recomendaciones:** activar las capas, probar `DETECTION_CONFIDENCE=0.70` (recuperaba todas las fallidas) y mantener `MATCH_THRESHOLD=0.45`.
- **Es optimista:** no mide lentes, personas parecidas, gemelos ni el paso de semanas. Los datos son de otra población y no se usó la cámara real.
- **La BD de prueba** quedó con 94 comensales (apellido `PRUEBA-*`, DNI `9000xxxx`). Hay que borrarlos antes de producción.

## Convenciones del código

- El trabajo pesado de CPU (detección, liveness) se ejecuta con `run_in_threadpool` para no bloquear el servidor.
- `app/routes/reconocimiento.py` ya no tiene endpoints: solo `_confirmar_asistencia`, los horarios de comida y el cooldown.
- Los endpoints de `app/routes/reportes.py` exigen el token del login (`get_current_user`).
- Para leer el caché de huellas se usan `cache.info()` y el valor que devuelve `enrich_embedding()`, no `cache._data`.

## Limitaciones y pendientes

- La capa de **luz no está medida** en hardware real. Hay que calibrar en el kiosco con los dos modos y luego decidir el modo y activar las capas.
- El fondo de referencia necesita **un fondo con textura** detrás del usuario, por ejemplo un póster. Con una pared lisa devuelve `sin_referencia` y no bloquea.
- Un video grabado con la misma cámara y encuadre del kiosco, en una pantalla que tape todo el lente, pasa el fondo de referencia.
- Cambiar la contraseña de `admin` después de instalar.
- Faltan pruebas con lentes (MeGlass), con las parejas más parecidas de LFW y un piloto con alumnos reales. Ver `docs/pruebas-exactitud.md`.
- Borrar los comensales `PRUEBA-*` de la BD antes de producción.
- `.env` estuvo versionado en commits anteriores a este cambio. Si el repositorio es público, conviene rotar la contraseña de postgres y `SECRET_KEY`.
