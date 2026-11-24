# Pruebas de exactitud con datos públicos (24 de noviembre de 2026)

Se probó el sistema completo (registro, kiosco por ráfaga, identidad y liveness) con datasets públicos. Se usaron la configuración actual de `app/core/config.py` y la BD real.

- **Scripts:** `scripts/pruebas/eval_exactitud.py` (prueba), `scripts/pruebas/analizar_exactitud.py` (métricas) y `scripts/pruebas/confianza_deteccion.py` (análisis de calidad).
- **Duración:** unos 34 minutos en CPU.

> **Importante:** estos números son **optimistas**. En la sección "Qué NO mide esta prueba" están los lentes, las personas parecidas, los gemelos y el paso del tiempo. La única medición válida para producción es un piloto con alumnos reales y la webcam del kiosco.

## Datos usados

| Rol | Datos | Cantidad |
|---|---|---|
| **Registrados** | CREMA-D: 15 fotogramas de un video por actor (`IEO_NEU`), por `/api/registrar_multiframe` | 55 de 61 intentos |
| | LFW: 3 fotos por persona, cada una en 5 variantes de movimiento y luz | 39 de 45 intentos |
| | **Total en la BD** | **94 comensales** |
| **Consultas de registrados** | CREMA-D: ráfagas de 5 fotogramas de **otros dos videos** del mismo actor (`DFA_NEU` neutral e `ITS_HAP` sonriendo) | 110 ráfagas |
| | LFW: fotos **no usadas en el registro**, en general de otros años | 173 fotos |
| **Desconocidos** | 30 actores de CREMA-D no registrados + 6 que no lograron registrarse | 72 ráfagas |
| | 20 personas de LFW | 49 fotos |
| | **Total** | **56 personas, 121 intentos** |
| **Escala** | Rostros distractores de LFW agregados a la galería | 1,834, para una galería de 1,928 |
| | Pares de personas distintas comparados | 172,396 |
| **Liveness** | CelebA-Spoof: 434 vivos y 1,349 ataques | 1,783 imágenes |
| | Ráfagas reales de CREMA-D | 168 |
| | Foto en la mano (simulada) | 83 |

## Resultados

### Identidad (umbral `MATCH_THRESHOLD=0.45`)

| Prueba | Reconoce bien | Confunde con otro | No reconocido | Falla de calidad | Desconocido aceptado |
|---|---|---|---|---|---|
| Kiosco, ráfaga (110) | **94.5%** | **0%** | 0% | 5.5% | **0%** (de 72) |
| · video neutral | 96.4% | 0% | 0% | 3.6% | 0% |
| · video sonriendo | 92.7% | 0% | 0% | 7.3% | 0% |
| LFW, fotos de otros años (173) | **94.8%** | **0%** | 1.2% | 4.0% | **0%** (de 49) |

**Barrido del umbral de identidad:**

| Umbral | Kiosco | LFW | Desconocido aceptado |
|---|---|---|---|
| 0.30 | 94.5% | 76.3% | 0% |
| 0.35 | 94.5% | 90.2% | 0% |
| 0.40 | 94.5% | 93.6% | 0% |
| **0.45** | **94.5%** | **94.8%** | **0%** |
| 0.50–0.60 | 94.5% | 95.4% | 0% |

**Distancias coseno:**

| Comparación | Media | p5 | p95 | Máximo |
|---|---|---|---|---|
| Misma persona, kiosco | 0.118 | 0.042 | 0.211 | 0.293 |
| Misma persona, LFW | 0.244 | 0.153 | 0.362 | 0.934 (caso aislado) |
| Impostor más cercano, kiosco | 0.831 | 0.774 | 0.876 | mínimo 0.670 |

- En los 172,396 pares de personas distintas, el par más parecido quedó en **0.695**.
- Entre los 94 registrados, el par más parecido quedó en 0.68.
- **Conclusión:** 0.45 tiene mucho margen hacia los dos lados y no conviene moverlo.

### Escala: cuántos comensales aguanta

| Galería | Reconoce bien (kiosco) | Reconoce bien (LFW) | Desconocido aceptado |
|---|---|---|---|
| 94 | 94.5% | 94.8% | 0% |
| 494 | 94.5% | 94.8% | 0% |
| 994 | 94.5% | 94.8% | 0% |
| 1,928 | 94.5% | 94.8% | 0% |

- Con 0 de 172,396 pares bajo 0.60, la probabilidad por par es como máximo 1.7×10⁻⁵ (regla del 3, con 95% de confianza).
- Con 1,000 comensales, el peor caso de aceptar a un desconocido en **una sola foto** es ~1.7%. La ráfaga exige que la misma persona salga en ≥3 de 5 fotos, así que el riesgo real es bastante menor.
- **Entre personas que no se parecen, la identidad no es el cuello de botella.**
- **Cuántos desconocidos probar:** con 0 aceptados de N intentos, el límite superior al 95% es 3/N. Con 50 personas se demuestra menos del ~5%. Con 300 intentos, menos del 1%.

### Liveness: MiniFASNet (AUC 0.83 en CelebA-Spoof)

| Umbral | Ataques que pasan (APCER) | Vivos rechazados, CelebA (BPCER) | Ráfagas reales rechazadas, kiosco |
|---|---|---|---|
| 0.30 | 62.5% | 4.8% | 0% |
| 0.40 | 58.5% | 5.5% | 0% |
| **0.50 (actual)** | **53.7%** | **8.3%** | **0%** |
| 0.60 | 48.3% | 11.5% | 0% |
| 0.70 | 42.3% | 13.1% | 0% |
| 0.80 | 36.1% | 18.2% | 0% |
| 0.90 | 28.0% | 22.8% | 0.6% |

- MiniFASNet **solo** no alcanza: con cualquier umbral pasa al menos el 28% de los ataques.
- Las ráfagas reales del kiosco sacan en promedio 0.99.

### Liveness: capa de profundidad

| | Modo sombra (actual, `LIVENESS_CAPAS_ACTIVAS=False`) | Capas activas |
|---|---|---|
| Personas reales que pasan (168) | 100% | **98.2%** (3 detectadas como "plano") |
| Foto en la mano que pasa (83) | **100%** ⚠️ | **0%** |

- **Con la configuración actual una foto movida en la mano pasa**, porque la profundidad solo se registra.
- Al activarla, la foto queda bloqueada al 100% y se rechaza a ~2% de las personas reales.
- El fondo quieto (lámina) dio `inconcluso` siempre: el fondo de CREMA-D es liso.
- El fondo de referencia no se probó aquí, porque CREMA-D no tiene fotogramas del fondo vacío. Ya estaba medido antes (ver `CLAUDE.md`).

### Calidad: la principal causa de fallos

- Todo el 5.5% que falló en el kiosco y los 6 registros fallidos de CREMA-D (~10%) se deben a la confianza de detección: `det_score < DETECTION_CONFIDENCE=0.80`.

| Ráfagas | det_score medio | Mínimo | ≥0.80 | ≥0.70 | Distancia máxima a su registro |
|---|---|---|---|---|---|
| Fallidas por calidad | 0.791 | 0.76 | 26% | **100%** | 0.287 |
| Buenas | 0.862 | 0.79 | 99% | 100% | 0.183 |

- Con `DETECTION_CONFIDENCE=0.70` se recuperarían **todas** las fallidas, y seguirían bajo 0.45. Eso subiría el reconocimiento del kiosco de ~94.5% a ~99%.
- CREMA-D tiene baja resolución (480×360). Con la webcam real hay que confirmarlo antes de cambiar el valor.

### Otros

- **Anti-duplicado:** 15 de 15 personas que intentaron registrarse otra vez con otro DNI y otro video fueron **bloqueadas**.
- **Velocidad:** el análisis de la ráfaga tarda ~1.07 s (mediana) y la respuesta del endpoint 0.93 s.
- **De punta a punta por HTTP:**
  - 54 de 61 registrados recibieron `REJECTED` "Fuera de horario de atención". Fueron identificados, pero la prueba corrió fuera del horario de comidas, así que no se guardaron asistencias.
  - 3 registrados recibieron `NO_MATCH`: son actores que no lograron registrarse.
  - Los 30 desconocidos dieron `NO_MATCH` o `LOW_QUALITY`: **no entró ninguno**.

## Qué NO mide esta prueba (por qué los números son optimistas)

1. **Registro y consulta del mismo día.** En CREMA-D los videos son de la misma sesión, cámara, luz, ropa y peinado. Por eso la distancia es tan baja (0.12). En el comedor pasan semanas entre el registro y el uso. LFW sí tiene fotos de otros años, y ahí la distancia sube a 0.24.
2. **Lentes.** No se comparó a la misma persona con y sin lentes. Los lentes de medida normales afectan poco; las monturas gruesas y los reflejos, más; **los lentes oscuros, mucho**, porque tapan los ojos.
3. **Personas parecidas.** Los impostores fueron personas al azar. Hermanos, primos y **gemelos idénticos** son el peor caso de estos modelos: con solo una webcam no hay forma confiable de distinguir gemelos.
4. **Población.** LFW tiene sobre todo famosos de EE. UU. y Europa, y CREMA-D actores de EE. UU. Los alumnos de la UNPRG son otra población, y está documentado que la exactitud cambia entre grupos étnicos.
5. **Ataques simulados.** La "foto en la mano" fue una imagen digital desplazada y girada. MiniFASNet la vio como imagen limpia y no como papel impreso, así que en la realidad también ayudaría.
6. **Cámara y luz real.** No se usó la webcam ni la iluminación del comedor.

## Recomendaciones

1. **Activar `LIVENESS_CAPAS_ACTIVAS=True`** después de calibrar en `/calibracion` con la cámara real. Es la mejora de seguridad más grande.
2. **Probar `DETECTION_CONFIDENCE=0.70`** con la webcam real.
3. **Mantener `MATCH_THRESHOLD=0.45`.**
4. **Quien use lentes debería registrarse con lentes y sin lentes** (dos plantillas). Hoy solo se compara contra el promedio.
5. Para gemelos o hermanos muy parecidos, **pedir el DNI como segundo factor**.

## Pruebas siguientes propuestas

| Prueba | Datos | Estado |
|---|---|---|
| Con y sin lentes | MeGlass (1,710 personas con y sin lentes) | Pendiente: hay que ver si se puede descargar |
| Los más parecidos | Las parejas de personas distintas más cercanas entre las 13,233 fotos de LFW | Pendiente, ya está descargado |
| Edad y tiempo | CALFW | Pendiente |
| **Piloto real** | 30–50 alumnos con la webcam del kiosco, registrados un día y probados otro, con y sin lentes, más hermanos | **La única medición válida para producción** |

## Datos de prueba en la BD

- La prueba dejó **94 comensales de prueba** en `comedor_db`, con apellido `PRUEBA-CREMA` o `PRUEBA-LFW` y DNI `9000xxxx`, junto con sus embeddings e intentos de acceso.
- Al volver a correr `eval_exactitud.py`, primero se borran los de la corrida anterior.
- **Hay que borrarlos antes de usar el sistema en producción.**
