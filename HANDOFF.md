# Traspaso — TPI IA 2026, Comprensión visual de escenas

> Preparado el 2026-10-23 tras las primeras 4 tareas del plan. Léase junto con [`REPORTE.md`](REPORTE.md) (reporte técnico de la cátedra, se actualiza incrementalmente) y el PDF del enunciado en la raíz del repo.

## 1. Objetivo y restricciones

Competencia privada de Kaggle `tpi-ia-2026-primavera-comprension-visual-de-escenas` (Cátedra IA, UTN). Pipeline a construir: **imagen → detección open-vocabulary → segmentación (SAM2) → `SceneJSON` estructurado → preguntas/descripción (determinista + LLM liviano opcional)**.

Niveles de logro (acumulativos): aprobado (F1 detección ≥0.40 + pipeline completa + `questions` determinista), promoción (F1 ≥0.50 + atributos/relaciones + `questions_llm` + LLM liviano), sobresaliente (extensión propia justificada).

**Restricciones duras:**
- La notebook final debe correr **100% en Kaggle**, sin archivos externos propios (solo el dataset oficial + lo que la propia notebook descargue públicamente).
- Prohibido usar las anotaciones para asistir la inferencia, o identificar/etiquetar manualmente las imágenes de test.
- El LLM (modo promoción) solo puede recibir el `SceneJSON` predicho, **nunca la imagen original**. Conteos/filtros/relaciones se resuelven por código, no por el LLM.
- Entrega: grupo de 3-5 personas, **1 de noviembre de 2026 23:59**. Nombre de archivo requerido: `TPI_IA_2026P_[nro_de_grupo]`.
- **Regla de esta sesión, no escrita en ningún archivo del repo:** el asistente (yo) tiene prohibido interactuar con Kaggle para unirse a la competencia, subir datasets/notebooks o hacer submissions — eso lo hace únicamente el usuario (Jerónimo) manualmente. Solo lectura vía navegador (con su sesión de Chrome) está permitida. **Cualquier asistente que continúe este proyecto debería mantener esta misma restricción salvo que el usuario diga explícitamente lo contrario.**

**Sin resolver, revisar con la cátedra:** el PDF del enunciado dice que el archivo de entrega se llama `submission.csv`; la página de Kaggle dice `solution.csv`. No bloquea el desarrollo pero hay que confirmarlo antes de la entrega final.

## 2. Decisiones tomadas y por qué

| Decisión | Motivo |
|---|---|
| Detector: **OWLv2** (`google/owlv2-base-patch16-ensemble`, HF `transformers`) | Pip-instalable sin compilar operadores CUDA custom (a diferencia de Grounding DINO) → más confiable en el entorno de Kaggle. Soporta vocabulario de texto libre. |
| Vocabulario: las **120 categorías completas** de `annotations.json`, en un solo batch de texto por imagen | No sabemos de antemano qué categorías aparecen en las imágenes de test; el EDA mostró cola larga, así que no tiene sentido recortar el vocabulario. |
| Umbral de confianza baseline: **0.1** (permisivo) | El EDA mostró que la mediana de `area_ratio` es ≈0.003 (objetos muy chicos) → un umbral alto mataría el recall. |
| NMS **por categoría** (no global), IoU>0.5 | Un NMS cruzado entre categorías borraría objetos distintos que se superponen legítimamente (ej. persona + mochila). El EDA de oclusión (13% de instancias con IoU>0.3 con su vecino) anticipó que iba a hacer falta. |
| Resolución de entrada: **960×960 fija**, la que usa OWLv2 por defecto | Es la resolución nativa del checkpoint (embeddings posicionales calibrados para ese tamaño); además el resize es no-uniforme (sin letterbox) y el dataset tiene relación de aspecto muy variada (0.45–1.78), así que cambiar la resolución sin re-calibrar el modelo arriesga más de lo que soluciona. Justificación completa en `REPORTE.md` sección 3.1. La cátedra aclaró que este punto solo exige justificación, no experimentación — el experimento controlado (sección 4) evalúa el umbral de confianza en su lugar. |
| Métrica `detection_f1` con política **LVIS-aware** (positiva / `neg_category_ids` / `not_exhaustive_category_ids` / no verificada) | Tomada literalmente del `README.md` del dataset de desarrollo (no es una invención nuestra) — ver el docstring de `src/metrics.detection_f1` para la lógica exacta. |
| `src/geometry.py` con `bbox_iou` compartido | Lo necesitan `eda.py` (oclusión), `detection.py` (NMS) y `metrics.py` (F1) — evita triplicar la misma función. |
| Segmentador: **SAM2.1 tiny** (`facebook/sam2.1-hiera-tiny`, HF `transformers`, no el repo oficial de Meta) | Misma lógica que con el detector: evita clonar/compilar `facebookresearch/sam2`. Checkpoint `tiny` (el más chico de la familia) por volumen de cajas a segmentar (7293 tras NMS) y presupuesto de cómputo. |
| `mask_to_rle`/`rle_to_mask` implementados a mano (no `pycocotools`) | El schema exige RLE no comprimido en orden column-major con `counts` como lista de enteros plana — no es el formato comprimido que da `pycocotools.mask.encode` por defecto, así que se implementó directo para controlar el formato exacto. Verificado con round-trip sobre máscaras reales. |
| Repo git consolidado en `/home/jeronimo-osuna/IA/TPI/.git` (normal) | La carpeta original estaba en un disco NTFS que no soporta `chmod` (rompe git); se usó `--separate-git-dir` como workaround temporal y se consolidó al mover el proyecto a un filesystem nativo. Detalle completo en el historial de chat, no en el repo. |
| `REPORTE.md` se completa **incrementalmente**, no al final | Para no perder el razonamiento de cada decisión; pedido explícito del usuario a mitad de proyecto. |

## 3. Estado actual — qué está implementado vs. solo propuesto

**Implementado y verificado con datos reales o tests sintéticos:**
- `src/eda.py` — completo. Corrido sobre las 120 imágenes reales, resultados en `reports/eda/` (CSVs + 4 PNGs).
- `src/geometry.py` — completo (`bbox_iou`).
- `src/detection.py` — `detect_objects` y `suppress_duplicates` completos. Verificados con inferencia real sobre imágenes del dataset y un caso sintético de NMS.
- `src/metrics.py::detection_f1` — completo, validado con un caso sintético calculado a mano antes de usarse sobre datos reales.
- `src/segmentation.py` — `segment_with_sam2` (SAM2.1 tiny, vía `transformers`), `mask_to_rle`/`rle_to_mask` (RLE no comprimido column-major). Corrido sobre **las 120 imágenes completas / 7293 cajas** (`scripts/run_segmentation.py`, resultados en `reports/segmentation/`): 100% de las máscaras no vacías, 541.6s totales (~9 min), `mask_area/bbox_area` media 0.63 (SAM2 recorta la forma real, no rellena la caja). RLE verificado (`sum(counts)==h×w`) sobre una muestra de los objetos reales.
- `src/scene_builder.py::build_scene_json` — ensambla `image`/`objects`/`relations` a partir de los objetos detectados+segmentados, con `id` único por objeto e `image.id` = nombre de archivo sin extensión (coincide con `image_id` de `sample_submission.csv`, verificado). Corrido sobre las 120 imágenes completas (`scripts/build_scenes.py`): 120 escenas, 7293 objetos, 0 escenas vacías.
- `src/properties.py` — `compute_centroid`, `compute_area_ratio`, `compute_position`, `compute_size` (cortes = terciles empíricos reales 0.00126/0.00765, no números inventados), `compute_dominant_color` (cuantización + bin más frecuente, no promedio), `compute_relations` (relaciones limitadas a los **5 vecinos más cercanos** por objeto, no todos los pares — la primera versión sin ese límite generaba hasta 12499 relaciones en la imagen más poblada, mayormente ruido). Corrido sobre las 120 escenas completas (`scripts/enrich_scenes.py`): **120/120 escenas validan contra `scene_schema.json` real** (usando la librería `jsonschema`, no una verificación manual), 22740 relaciones totales (antes 152906 sin el límite de vecinos), tamaños balanceados (small=2671, medium=2613, large=2009).
- `src/tracing.py::Tracer` — `log_step` (registro explícito) + `step` (context manager que mide tiempo y captura+relanza errores, registrándolos con `status="error"`) + `load_trace` (helper de lectura). Formato JSON Lines en `reports/trace/pipeline.jsonl`, un `run_id` por ejecución. Probado de forma aislada (incluyendo el camino de error, que de verdad relanza la excepción) **y** integrado en `scripts/run_eda.py`, `scripts/build_scenes.py` y `scripts/enrich_scenes.py`, re-corridos con tracing activo: 361 registros reales, resultados idénticos a las corridas previas sin tracing. **`run_detection.py` y `run_segmentation.py` tienen el código instrumentado pero no se volvieron a correr** (ya habían corrido antes de agregar tracing; repetirlos solo para generar trazas hubiera costado ~15 min sin aportar nada nuevo) — sus módulos (`detection`, `nms`, `segmentation`) todavía no aparecen en `pipeline.jsonl`.
- `src/qa_engine.py` — `answer_question` (count/existence/position/size/relation/description) + `describe_scene` (plantilla determinista, top-5 categorías). Decisiones consultadas con el usuario antes de implementar: predicado de `relation` por matching de frases fijas en español (verificado contra las 10 preguntas reales), ambigüedad (>1 instancia donde se presupone una sola) → `unsupported`, `confidence` = mínimo de las detecciones involucradas. Corrido sobre las **60 preguntas reales** de `questions_dev.jsonl` (`scripts/run_qa.py`, resultados en `reports/qa/`): **`count` exact-match 3/10 (30%), `existence` accuracy 9/10 (90%)** contra el ground truth real — el `count` bajo es la métrica de detección (F1=0.593) propagándose a QA, no un bug del motor de preguntas (detalle completo en `REPORTE.md` 4.6/sección 5). 11/20 preguntas de `position`/`size`/`relation` dieron `unsupported`, 9 de esas 11 por ambigüedad (el detector encuentra más de una instancia donde la pregunta presupone una sola) — consecuencia medible del umbral de confianza permisivo (0.1) elegido en detección.
- `scripts/run_eda.py`, `scripts/run_detection.py`, `scripts/run_segmentation.py`, `scripts/build_scenes.py`, `scripts/enrich_scenes.py`, `scripts/run_qa.py` — funcionales, ya corridos de punta a punta sobre el dataset completo.
- `REPORTE.md` — secciones 1 (Portada, con integrantes/nro. de grupo **pendientes de completar**), 2 (Análisis del dataset, completa), 3.1-3.7 (Metodología de detección, métrica, segmentación, construcción de SceneJSON, propiedades/relaciones, trazabilidad y QA determinista, completas), 4.1-4.6 (Resultados de detección, segmentación, construcción de SceneJSON, propiedades/validación de schema, trazabilidad y QA, completos). Sección 6 (Conclusiones) pendiente.
- Repo git: working tree limpio, con remoto privado en GitHub (ver sección 6) — ver `git log` para el número exacto de commits al momento de leer esto.

**Solo esqueleto (`raise NotImplementedError`), sin probar — todo lo que queda del plan:**
- `src/qa_llm.py` (`interpret_query`, `generate_description_llm`)
- `src/metrics.py::mask_f1`, `scene_json_consistency`, `qa_accuracy`

## 4. Experimentos realizados

1. **EDA completo** sobre las 120 imágenes de desarrollo — ver `reports/eda/` y sección 2 de `REPORTE.md`.
2. **Detección baseline** sobre las 120 imágenes (`confidence_threshold=0.1`, `nms_iou_threshold=0.5`):
   - **F1=0.593, precision=0.512, recall=0.705** (ya supera el piso de promoción, ≥0.50).
   - TP=1262, FP=1203, FN=528, predicciones ignoradas por categoría no verificada=4828 (66% del total — esperado, porque se consultan las 120 categorías en cada imagen).
   - Tiempo: 366.7s total / 3.1s por imagen en **CPU local, sin batching** (en GPU de Kaggle debería ser bastante más rápido).
   - Resultados crudos en `reports/detection/predictions.json` y `reports/detection/detection_f1_summary.json`.
3. **Segmentación SAM2.1 tiny** sobre las 7293 cajas detectadas (todas las 120 imágenes):
   - 7293/7293 objetos segmentados, ninguna máscara vacía.
   - Tiempo: 541.6s total (~9 min) — SAM2 batchea todas las cajas de una imagen en un solo forward, por eso no escala linealmente con la cantidad de cajas.
   - `mask_area/bbox_area` media 0.634, mediana 0.646 (SAM2 recorta la forma real del objeto, no rellena la caja completa).
   - Resultados crudos en `reports/segmentation/objects_with_masks.json` (7MB) y `reports/segmentation/segmentation_summary.json`.
4. **Construcción de `SceneJSON`** sobre las 120 imágenes (`scripts/build_scenes.py`): 120 escenas, 7293 objetos, 0 escenas vacías. Salida en `reports/scene_json/scenes.json` (7MB, ahora un artefacto intermedio superado por el siguiente punto).
5. **Enriquecimiento con propiedades + relaciones + validación de schema** sobre las 120 escenas (`scripts/enrich_scenes.py`):
   - **120/120 escenas válidas contra `scene_schema.json`** (validación real con la librería `jsonschema`, no manual).
   - Primer intento de `compute_relations` (todos los pares por imagen): 152906 relaciones totales, hasta 12499 en una sola imagen (257 objetos) — demasiado ruido, mayormente pares sin relación semántica real.
   - Versión final (5 vecinos más cercanos por objeto): 22740 relaciones totales, máximo 819 por imagen, promedio 189.5.
   - Tamaños: small=2671, medium=2613, large=2009 (balanceado, usando los terciles empíricos de `area_ratio`: 0.00126/0.00765).
   - Salida final en `reports/scene_json/scenes_enriched.json` — **este es el `SceneJSON` completo y válido, no `scenes.json`**.
6. **Trazabilidad** — `run_eda.py`, `build_scenes.py` y `enrich_scenes.py` re-corridos con `Tracer` activo: 361 registros en `reports/trace/pipeline.jsonl`, resultados idénticos a las corridas sin tracing (reproducibilidad confirmada). Camino de error verificado por separado con un test sintético (la excepción se relanza, no se traga).
7. **QA determinista** sobre las 60 preguntas reales de `questions_dev.jsonl` (`scripts/run_qa.py`):
   - `count` exact-match: 3/10 (30%). `existence` accuracy: 9/10 (90%). Comparado contra el ground truth real de `annotations.json`.
   - 11/20 preguntas `position`/`size`/`relation` → `unsupported` (9 por ambigüedad, 2 por falta de detección real).
   - **Hallazgo clave:** el bajo exact-match de `count` y la alta tasa de `unsupported` por ambigüedad no son bugs — son el umbral de confianza permisivo (0.1, elegido en detección) propagándose hacia la capa de QA. Más objetos detectados de más = conteos inflados y más ambigüedad en preguntas de instancia única.
   - Resultados crudos en `reports/qa/answers_dev.json` y `reports/qa/qa_summary.json`.

**No se corrió ninguna comparación controlada todavía** (ej. umbral de confianza alternativo, otra resolución de entrada, otro detector) — eso es la Tarea 12 del plan original y sigue pendiente.

## 5. Problemas conocidos / huecos detectados

1. ~~Resolución de entrada de OWLv2 sin justificar.~~ **Resuelto** (ver `REPORTE.md` sección 3.1): es un resize no-uniforme a 960×960 fijo, sin letterbox; se justificó mantenerla nativa en vez de experimentar con ella, por indicación explícita de la cátedra de que solo exige justificación, no comparación.
2. Discrepancia `submission.csv` (PDF) vs `solution.csv` (Kaggle) sin resolver — confirmar con la cátedra.
3. La carpeta original del proyecto estaba en un disco NTFS (`/mnt/datos/...`) que no soporta `chmod`; si en algún momento se vuelve a trabajar desde ese disco, git va a volver a fallar. Ya no aplica mientras el proyecto siga en `/home/jeronimo-osuna/IA/TPI` (filesystem nativo).
4. `~/.gitconfig` tiene entradas `safe.directory` de otro proyecto del usuario (`BitLogic/backoffice`) — no tocar, no están relacionadas con este repo.
5. Integrantes del grupo y número de grupo **sin completar** en `REPORTE.md` (necesario también para el nombre de archivo final `TPI_IA_2026P_[nro_de_grupo]`).

## 6. Entorno de desarrollo local (importante, no está en el repo)

- Ya existe un venv persistente en `~/IA/TPI/.venv` (creado por el usuario, gitignored) con `torch`, `transformers`, `scipy`, `pandas`, `matplotlib`, etc. ya instalados desde `requirements.txt` — **no hace falta reinstalar desde cero**, solo activar (`source .venv/bin/activate`) o usar `.venv/bin/python`.
- El `pip install` directo con `python3 -m pip` falla en este sistema (Debian/Ubuntu "externally-managed-environment") — siempre usar el venv.
- `data/development/` (120 imágenes + `annotations.json`), `data/tpi_ia_2026_competition_public/` (60 imágenes de test) y `data/tpi_2026_qa_public/` ya están descargados y organizados localmente, pero **gitignored** (pesan demasiado). `data/scene_schema.json` y `data/sample_submission.csv` sí están trackeados en git — son la referencia autoritativa del contrato, **no hace falta volver a scrapear Kaggle para leerlos**, ya están en el repo tal cual los bajó Kaggle.
- Internet sí está disponible desde este entorno (se pudo descargar el modelo OWLv2 de HuggingFace sin problema).
- El repo tiene remoto en GitHub, **privado**: [`github.com/jeronimoosuna/tpi-ia-2026-comprension-visual-escenas`](https://github.com/jeronimoosuna/tpi-ia-2026-comprension-visual-escenas). `origin/master` rastreado, un `git push` normal alcanza para subir commits nuevos.

## 7. Tareas pendientes (en el orden del plan original)

1. Completar métricas: `mask_f1`, `scene_json_consistency`, `qa_accuracy` (`detection_f1` ya está hecho; `scene_json_consistency` en parte ya se demostró indirectamente con la validación `jsonschema` de `scripts/enrich_scenes.py`, pero falta la función propia en `src/metrics.py`; `qa_accuracy` ya se calculó ad-hoc dentro de `scripts/run_qa.py` para `count`/`existence`, pero falta la función reutilizable en `metrics.py`).
2. Generar y validar `submission.csv`/`solution.csv` final (80 filas) — ya hay 120 escenas completas y válidas en `reports/scene_json/scenes_enriched.json`, falta filtrar las 20 de desarrollo + las 60 de test (que todavía no se corrieron por la pipeline) y armar el CSV final.
3. LLM liviano para `questions_llm` y descripción (`src/qa_llm.py`) — nivel promoción. Importante: dado el hallazgo de la sección 4 (umbral de confianza afecta mucho la tasa de `unsupported`), vale la pena correr primero el punto 5 (experimento de umbral) antes de evaluar el LLM, para no confundir "el LLM interpreta mal" con "el detector generó ambigüedad".
4. Experimento controlado vs. línea base sobre el **umbral de confianza** (ej. 0.1 vs 0.3) — Tarea 12 del plan. Ahora con un argumento más fuerte para priorizarlo: no es solo F1 de detección, también mejoraría directamente la tasa de respuestas resueltas en QA (ver hallazgo en sección 4, punto 7).
5. Reporte técnico final + consolidación de todo en una única notebook de Kaggle (el usuario la sube manualmente).
6. (Opcional, barato) Re-correr `run_detection.py`/`run_segmentation.py` con tracing ya activo para que `reports/trace/pipeline.jsonl` cubra la pipeline completa, no solo eda/scene_builder/properties/qa_engine — no es urgente porque los resultados ya existen y son válidos, pero suma si hay tiempo.

**Pendiente transversal, no es una tarea en sí:** correr toda la pipeline (detección → segmentación → scene_builder → properties → qa) también sobre las **60 imágenes de test** (`data/tpi_ia_2026_competition_public/test_images/`), no solo sobre las 120 de desarrollo — hasta ahora todo se corrió únicamente sobre desarrollo. Sin eso no se puede armar el `submission.csv` final.

**Instrucción del usuario para toda sesión futura:** consultar antes de tomar decisiones de diseño (no asumir y avisar después); cualquier error/limitación identificado debe quedar escrito en `REPORTE.md`, no solo mencionado en el chat.

## 8. Próximo paso recomendado

**Experimento controlado de umbral de confianza** (Tarea 12 del plan, punto 4 de la lista de arriba) — ahora tiene más prioridad que antes: el hallazgo de esta sesión (el umbral 0.1 genera ambigüedad que tira abajo la tasa de respuestas resueltas en QA, no solo afecta el F1 de detección) es un argumento concreto para correrlo pronto, y además sirve como la comparación controlada que pide la Etapa 9 del enunciado (evaluación cuantitativa). Alternativa razonable: seguir con `src/qa_llm.py` si se prefiere completar todas las etapas de implementación antes de experimentar — consultar con el usuario cuál prefiere antes de arrancar.

## 9. Mapeo con las etapas oficiales del enunciado

El PDF (págs. 5-6, sección "Desarrollo del trabajo") numera el trabajo en **9 etapas explícitas**, distintas de la numeración de tareas que usamos en el plan interno. Para ubicarse rápido:

1. Análisis exploratorio del dataset → `src/eda.py` ✅
2. Detección open-vocabulary → `src/detection.py` ✅
3. Segmentación de instancias con SAM2 **+ derivar propiedades de las máscaras** (centroide, área relativa, posición, tamaño, color dominante, superposición/proximidad, relaciones espaciales — todo bajo el mismo punto 3, no es una etapa aparte) → `src/segmentation.py` ✅ + `src/properties.py` ✅
4. Representación estructurada de la escena (JSON con id únicos, clase/confianza/bbox, máscara, propiedades, atributos/relaciones) → `src/scene_builder.py` ✅
5. Trazabilidad de la pipeline → `src/tracing.py` ✅ (instrumentado en los 5 scripts, corrido de verdad en 3; `run_detection.py`/`run_segmentation.py` instrumentados pero no re-ejecutados, ver sección 7 punto 6)
6. Responder preguntas en modo determinista → `src/qa_engine.py` ✅ (corrido sobre las 60 preguntas reales; `count` 30% exact-match / `existence` 90% accuracy, ver hallazgo de propagación de errores en sección 4 punto 7)
7. LLM liviano (promoción): descripción factual + segundo modo de QA interpretado → `src/qa_llm.py` ❌ **siguiente paso (a confirmar con el usuario, ver sección 8)**
8. Evaluación cuantitativa y análisis de casos → `src/metrics.py` (parcial: `detection_f1` ✅, resto ❌) + secciones 4.6/5 de `REPORTE.md` (parcial, con el hallazgo de propagación de errores ya documentado)

Si en algún momento hay que decidir qué falta para "aprobado" vs. "promoción", conviene revisar esta lista contra la sección "Criterios de evaluación y niveles de logro" del PDF (págs. 8-9) en vez de improvisar.
