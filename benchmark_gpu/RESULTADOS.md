# Reconocimiento facial en GPU + anti-suplantación (2026-10-04)

Equipo: i7-9700 (8 núcleos), RTX 5060 Ti 16 GB, Windows 11, onnxruntime-gpu 1.23.2 (CUDA 12.9, cuDNN 9 por pip).
Modelo: InsightFace buffalo_l (SCRFD det_10g + ArcFace w600k_r50), det_size 640.

## 1. Velocidad por cuadro (lo que hace `/api/reconocer`: detectar + embedding)

| Configuración | Webcam 640×480 (1 cara) | Foto de grupo (6 caras) | VRAM |
|---|---|---|---|
| CPU, 5 modelos (versión original) | 502 ms (2.0 FPS) | 2,123 ms | — |
| CPU, 2 modelos (det + rec) | 282 ms (3.6 FPS) | 526 ms | — |
| GPU, 5 modelos | 26 ms (38 FPS) | 97 ms | 876 MB |
| **GPU, 2 modelos (versión nueva)** | **17 ms (58 FPS)** | **42 ms** | **582 MB** |

Corrida limpia (GPU libre, 100 iteraciones, 2026-10-05): **29× más rápido** por cuadro de webcam, 50× en foto de grupo.

- Los embeddings GPU y CPU coinciden (similitud coseno 0.99999): mismo resultado.
- `FaceAnalysis` cargaba 5 modelos, pero el sistema solo usa 2 (no usa landmarks 2D/3D ni género/edad).
  Quitarlos ya acelera ~1.8× incluso en CPU.
- Repetir: `F:\Tesis\venv_gpu\Scripts\python.exe benchmark_gpu\bench_gpu.py --iter 200`

## 2. Anti-suplantación: heurística actual vs MiniFASNet

Datos: 70 videos de webcam (UniqueData/web-camera-face-liveness-detection, CC BY-NC-ND 4.0, uso académico):
10 reales + 60 ataques (monitor, foto A4, foto recortada, máscara 2D, silueta, máscara de silicona); 12 cuadros
por video, reducidos a 640 px. Script: `eval_antispoof.py`. Métricas ISO/IEC 30107-3
(APCER = ataques aceptados; BPCER = reales rechazados).

| Método | APCER | BPCER | AUC |
|---|---|---|---|
| Heurística actual (LBP+color+nitidez, umbral 0.37) | 85.5 % | 4.2 % | 0.57 |
| MiniFASNet V2+V1SE (umbral 0.5) | 85.4 % | 4.2 % | 0.69 |
| Ambos (deben pasar los dos) | 76.6 % | 7.6 % | — |

Conclusión honesta: con **un solo cuadro de webcam de baja resolución**, ningún método pasivo frena estos ataques.
El modelo separa algo mejor (AUC 0.69 vs 0.57) y no requiere calibración manual, pero no basta.
Limitaciones del conjunto: videos muy comprimidos, el monitor llena todo el cuadro (sin bordes visibles) y la
heurística se calibró con otra cámara. Falta la prueba decisiva: **tu cámara + tu celular**.

## 3. Prueba con la cámara real (Razer Kiyo V2 X, 640×480, celular del autor) — 2026-10-05

Sesión guiada `sesion_camara.py` (1,927 cuadros, cuarto con poca luz). `analizar_sesion.py`:

| Fase | Heurística acepta | MiniFASNet acepta | Giro (p5…p95) |
|---|---|---|---|
| Real, de frente | 66 % | **100 %** | −0.16…0.08 |
| Real, girando la cabeza | 6 % | **100 %** | **−1.28…0.49** |
| Real, lejos/cerca | 75 % | **100 %** | −0.16…0.11 |
| Foto en celular | 70 % | 71 % | −0.04…0.01 |
| Foto en celular girado | **95 %** | 16 % | −0.03…0.09 |
| Video en celular | 0 % | **3 %** | −0.45…0.63 |

- La heurística rechaza a la persona real (sobre todo al girar) y acepta el celular girado → queda solo en el log.
- El modelo nunca rechazó a la persona real (percentil 1 = 0.63), pero una selfie de cerca lo engaña.
- **Giro** = (x nariz − x centro de ojos) / distancia entre ojos, con los 5 puntos de SCRFD. Una superficie plana no
  pasa de 0.09 aunque se gire el celular; una cabeza real llega a ±0.4–1.3 (umbral 0.25, margen 3×).
- Se complementan: la foto engaña al modelo pero no puede girar; el video puede girar pero el modelo lo rechaza.
- Al girar, la confianza del detector baja de 0.80 (43 % de cuadros a giro medio, 4 % a giro fuerte): por eso el reto
  avanza ANTES del filtro de calidad, pero solo con cuadros que pasaron el modelo.

**Decisión implementada:** modelo pasivo (cada cuadro) + reto activo aleatorio (izquierda/derecha y volver al frente)
antes de confirmar la votación 3/5. `simular_reto.py` reproduce la sesión por la lógica real del endpoint
(cada 0.4 s, 4 desfases × 2 lados): **real que gira: 8/8 confirmados en 1.8–3.8 s; ataques: 0/24 aceptados**;
real que no gira cuando se le pide: no confirma (correcto).

Limitaciones: una persona, una sesión, una iluminación; no se probaron máscaras 3D ni foto impresa. Un video que
muestre el giro correcto Y engañe al modelo pasaría (aquí el modelo rechazó el 97 % de los cuadros de video).
La identidad en la simulación se aproximó (no se guardaron embeddings). Falta prueba de punta a punta en la web:
la base local no tiene pgvector ni la tabla `embeddings`.


## Cambios en el sistema
- `app/services/gpu.py` (nuevo): CUDA si hay `onnxruntime-gpu`, si no CPU; tope de VRAM `GPU_MEM_LIMIT_MB`.
- `app/services/face_service.py`: usa `gpu.providers()` y solo det + rec.
- `app/services/antispoof_service.py` (nuevo) + `models_ml/antispoof/*.onnx` (Apache-2.0, yakhyo/face-anti-spoofing).
- `app/services/reto_service.py` (nuevo): reto de giro aleatorio.
- `app/routes/reconocimiento.py`: modelo decide el liveness pasivo (heurística al log), reto antes de confirmar
  (`_add_vote(..., permitir_confirmar)`), la instrucción viaja como `message` del estado `SCANNING`.
- `templates/index.html` y `reconocimiento.html`: intervalo 1500 → 400 ms.
- `app/core/config.py`: `ANTISPOOF_*`, `RETO_*`, `USE_GPU`, `GPU_MEM_LIMIT_MB`.
- `requirements-gpu.txt` (nuevo).

## Entorno en Windows (`F:\Tesis\venv_gpu`)
El `venv` del repo es de Linux (Python 3.10). En Windows, insightface 0.7.3 no trae wheel y su extensión
`face3d` (solo para `mask_renderer`, que el sistema no usa) necesita MSVC: se instaló desde el código fuente
con `ext_modules=[]` y esos imports en try/except. cuDNN carga sub-DLL por PATH: `gpu.py` agrega
`site-packages\nvidia\*\bin`.
