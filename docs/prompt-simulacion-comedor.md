# Prompt: simulación de 2 semanas del comedor con 100 alumnos

Este es el prompt para pedirle a Claude Code, en otra máquina con este repositorio, la simulación con métricas ISO para la clase. Primero hay que hacer la **preparación de la máquina**, y después copiar el prompt completo.

## Preparación de la máquina (una sola vez)

1. **Sistema:** seguir "Puesta en marcha" en `CLAUDE.md`:
   - Python 3.10 en `.venv`, con `requirements.txt`.
   - PostgreSQL 17 con pgvector y `scripts/crear_bd.sql`.
   - `.env` creado a partir de `.env.example`.
2. **Dependencias de las pruebas:**
   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements-pruebas.txt
   ```
3. **Datos públicos** (~1.5 GB, van a `datos_pruebas/`, que no se sube a git):
   ```powershell
   .\.venv\Scripts\python.exe scripts\pruebas\descargar_datos.py
   ```
4. **Huellas de población** (~25 min en CPU):
   ```powershell
   .\.venv\Scripts\python.exe scripts\pruebas\embeddings_poblacion.py
   ```
5. Los modelos de InsightFace (`buffalo_l`, ~280 MB) se descargan solos la primera vez, así que hace falta internet. Los de anti-spoofing ya están en `models/antispoof/`.

## Prompt

````markdown
# Tarea: simulación de 2 semanas del comedor UNPRG con 100 alumnos (datos de prueba)

Responde siempre en español. Trabaja en este repositorio. Lee primero `CLAUDE.md`,
`docs/pruebas-exactitud.md` y `docs/prompt-simulacion-comedor.md` (preparación).
No hagas commit ni push, y no cambies la configuración del sistema
(`app/core/config.py`) ni su código.

Parte de los scripts que ya existen y verifica antes que la preparación esté hecha
(`datos_pruebas/` con LFW, FairFace, CelebA-Spoof y `emb_*.npz`; BD y `.env`):
- `scripts/pruebas/simulacion_comedor.py` (simulación) y
  `scripts/pruebas/metricas_simulacion.py` (métricas). Mejóralos y complétalos.

## Objetivo
Simular la operación del comedor durante 10 días hábiles con 100 alumnos registrados, en
desayuno, almuerzo y cena, para mostrar en una clase las métricas de desempeño
(ISO/IEC 19795-1 e ISO/IEC 30107-3). TODO es SIMULADO: márcalo así en la BD, en el reporte
y en cada gráfico ("Datos simulados – no es una medición en campo").

## 1. Muestra (documenta exactamente qué se usó)
- **Registrados (100 exactos):** personas de LFW con ≥10 fotos, priorizando rasgos latinos
  con el selector entrenado en FairFace. Regístralas por el endpoint real
  `/api/registrar_multiframe` (15 fotogramas a partir de sus 3 fotos más nítidas). Si una
  falla, reemplázala por la siguiente y cuéntala en el FTE. Las fotos de registro NO se
  reutilizan en las consultas.
- **Impostores (30 personas × 5 intentos = 150):** NO los elijas al azar. Toma los 30 MÁS
  PARECIDOS a algún registrado (menor distancia ArcFace): 15 de LFW con ≥5 fotos y 15
  latinos de FairFace. Es el peor caso realista (personas con facciones parecidas). Reparte
  sus 5 intentos en días y turnos al azar. Si alguno resulta ser la misma persona con otro
  nombre (pasa en LFW), repórtalo aparte.
- **Ataques (40):** 20 suplantaciones de registrados (10 con foto impresa y 10 con celular,
  sostenidos en la mano con movimiento plano) y 20 ataques reales de CelebA-Spoof. Evalúa
  cada ataque en dos modos: `LIVENESS_CAPAS_ACTIVAS=False` (actual) y `True`.

## 2. Calendario y condiciones (reloj simulado)
- 10 días hábiles (2 semanas). Ventanas: desayuno 06:30–08:50, almuerzo 11:45–14:45,
  cena 17:45–20:15. Asistencia: desayuno 75%, almuerzo 95% y cena 65%.
- Haz que el sistema crea que es la fecha y hora del evento: reemplaza `datetime` en
  `app/routes/reconocimiento.py` y `app/services/validacion_service.py`, y fija
  `fecha_hora` de marcaciones e intentos con un `before_insert` de SQLAlchemy. Limpia
  `_cooldowns` al cambiar de turno.
- Cámara por turno sobre una escena de 640×480, con ráfagas de 5 fotogramas con
  micro-movimiento y ruido de sensor:
  - mañana: luz natural fría y tenue;
  - tarde: buena luz, y 30% de las veces contraluz;
  - noche: fluorescente, poca luz, tinte, más ruido y 30% de las veces desenfoque.
- Reintentos realistas: si el sistema no reconoce al alumno, este vuelve a intentar con
  otra foto suya (hasta 3 intentos por visita). Registra cada intento por separado.
- Las personas simuladas son fotos, así que MiniFASNet las rechazaría. Solo para los
  intentos de personas "reales" (registrados e impostores) reemplaza la nota de MiniFASNet,
  pero mantén el control de poca luz (`LOW_LIGHT`). Para la BPCER usa el valor medido con
  videos reales (CREMA-D: 0% con el modo actual y 1.8% con capas activas, ver
  `docs/pruebas-exactitud.md`) y dilo. Los ataques pasan por el MiniFASNet real.

## 3. Calibración a ~95% (sin trampas)
- Meta didáctica: tasa de identificación correcta de los registrados de ~95% (93–97%).
- Primero corre un piloto de 2 días. Si la tasa queda fuera de 93–97%, ajusta SOLO la
  severidad de las condiciones del ambiente (luz, distancia y ruido de cada turno), nunca
  el umbral, el código del sistema ni los resultados. Documenta cada ajuste en el reporte,
  en la sección "Calibración".
- Usa semillas fijas para que la corrida se pueda repetir. Reporta el valor que salga,
  aunque no sea 95%.

## 4. Reglas de objetividad
- Nunca reportes "0%" a secas. Si no hubo errores, da el límite superior al 95%
  (Clopper-Pearson; con 0 errores en n intentos es ≈3/n) y explica qué significa.
- Da todas las tasas con su intervalo de confianza al 95% (Wilson).
- Los grupos no se mezclan: legítimos (A), impostores (B), ataques (C).
- **Una sola accuracy:** Tasa de identificación correcta = OK / N_leg. No calcules
  (OK + impostores rechazados) / total: sale inflada porque mezcla grupos.
- Reporta todos los fallos con su motivo: NO_FACE, LOW_QUALITY, LOW_LIGHT, NO_MATCH,
  confusiones y rechazos por horario o por comida ya consumida.
- Si un impostor entra y "come" como un registrado, y luego el registrado es rechazado
  porque esa comida ya figura consumida, cuéntalo como FA del impostor (no como fallo del
  legítimo) y muéstralo como un caso.
- Agrega una sección de limitaciones: qué no mide esta simulación (gemelos, cámara real,
  población de Lambayeque, ataques físicos reales).

## 5. Base de datos (que quede auditado)
- Borra antes los comensales con apellido `PRUEBA-%` y sus marcaciones, asistencias,
  intentos y embeddings.
- Los 100 quedan como comensales con apellido `PRUEBA-SIM` y DNI `93xxxxxx`. Sus
  asistencias, marcaciones e intentos quedan con las fechas simuladas.
- Crea la tabla `auditoria_pruebas` con una fila por intento:
  - `corrida` (`SIM-AAAAMMDD-HHMM`), `fecha_hora`, `dia`, `comida`;
  - `tipo` (`registrado` | `impostor` | `ataque`), `persona`, `comensal_id_real`, `intento`,
    `condicion`;
  - `status_sistema`, `mensaje`, `comensal_id_detectado`, `distancia`;
  - `clase` (`OK` | `NR` | `CF` | `FA` | `TN` | `ATAQUE_PASA` | `ATAQUE_BLOQUEADO`),
    `correcto`, `capas_activas`.
- Al final entrega las consultas SQL listas para mostrar en clase: aciertos por turno,
  impostores que entraron, ataques por tipo y resumen por día.

## 6. Métricas (todas, EXCEPTO las de tiempo)
Abreviaturas: N_leg (intentos legítimos), OK, NR, CF, N_imp, FA. Debe cumplirse
OK + NR + CF = N_leg. Calcula cada métrica por intento y por visita (hasta 3 intentos).

**1. Registro y captura**
- FTE.
- FTA, con sus motivos.
- Calidad media del registro (fotogramas válidos de 15 y det_score medio).

**2. Legítimos (grupo A)**
- ⭐ Tasa de identificación correcta = accuracy (OK/N_leg).
- ⭐ FNIR = (NR+CF)/N_leg.
- FRR = NR/N_leg.
- Tasa de confusión = CF/N_leg, con su límite superior.
- Recall.
- Precision = OK/(OK+CF+FA).
- F1.
- Tasa correcta total = OK/(N_leg + reintentos por vida), con la BPCER medida.
- Rank-1 (R1): la persona más cercana es la correcta, sin aplicar umbral.

**3. Impostores (grupo B)**
- ⭐ FPIR por intento (FA/N_imp).
- ⭐ FPIR por persona (impostores que entraron al menos una vez / 30).
- Ambos con su límite superior al 95%.

**4. Umbral y distancias**
- Distancia de la misma persona (media y DE).
- Distancia entre personas distintas (media y DE).
- FMR y FNMR en el umbral 0.45, con barrido de 0.30 a 0.60.
- EER.
- Curva DET y ROC (PNG).
- Confianza = (1 − d/2) × 100.
- Recomienda un umbral y justifícalo.

**5. Detección de vida (grupo C)**
- ⭐ APCER por tipo de ataque, en los dos modos: se decide por el peor tipo.
- APCER global.
- ⭐ BPCER (tomada de la medición con videos reales e indicándolo).

**7. Equidad y condiciones**
- ⭐ FNIR por turno: desayuno, almuerzo y cena, con la Q de Cochran (bloques
  alumno-día que asistieron a los 3 turnos).
- ⭐ FNIR por sexo (sexo según los atributos de LFW), con prueba exacta de Fisher y
  odds ratio.
- FNIR por día (¿mejora con el auto-enriquecimiento?) y por número de intento.
- Matriz de confusión (OK/NR/CF para legítimos y FA/TN para impostores) y tabla por turno.

## 7. Entregables
- `docs/simulacion-comedor.md`, con estas secciones:
  - resumen para la clase (accuracy, FNIR, FPIR, APCER y BPCER en una tabla);
  - qué muestra se usó;
  - calibración;
  - todas las métricas con su intervalo;
  - limitaciones.
- `docs/simulacion-comedor.json` con los números.
- Figuras PNG en `docs/img/`: curva DET, histogramas de distancias, aciertos por turno y
  por día, y la matriz de confusión.
- Los scripts actualizados en `scripts/pruebas/`, para repetir la corrida.
- No guardes fotos de rostros en el repositorio (`data/` y `datos_pruebas/` quedan fuera).
````
