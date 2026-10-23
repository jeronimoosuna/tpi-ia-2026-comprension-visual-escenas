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
| Resolución de entrada: **960×960 fija**, la que usa OWLv2 por defecto | **No es una decisión justificada, es un hueco detectado y todavía no resuelto** — ver sección 5. El enunciado pide explícitamente justificar este punto. |
| Métrica `detection_f1` con política **LVIS-aware** (positiva / `neg_category_ids` / `not_exhaustive_category_ids` / no verificada) | Tomada literalmente del `README.md` del dataset de desarrollo (no es una invención nuestra) — ver el docstring de `src/metrics.detection_f1` para la lógica exacta. |
| `src/geometry.py` con `bbox_iou` compartido | Lo necesitan `eda.py` (oclusión), `detection.py` (NMS) y `metrics.py` (F1) — evita triplicar la misma función. |
| Repo git consolidado en `/home/jeronimo-osuna/IA/TPI/.git` (normal) | La carpeta original estaba en un disco NTFS que no soporta `chmod` (rompe git); se usó `--separate-git-dir` como workaround temporal y se consolidó al mover el proyecto a un filesystem nativo. Detalle completo en el historial de chat, no en el repo. |
| `REPORTE.md` se completa **incrementalmente**, no al final | Para no perder el razonamiento de cada decisión; pedido explícito del usuario a mitad de proyecto. |

## 3. Estado actual — qué está implementado vs. solo propuesto

**Implementado y verificado con datos reales o tests sintéticos:**
- `src/eda.py` — completo. Corrido sobre las 120 imágenes reales, resultados en `reports/eda/` (CSVs + 4 PNGs).
- `src/geometry.py` — completo (`bbox_iou`).
- `src/detection.py` — `detect_objects` y `suppress_duplicates` completos. Verificados con inferencia real sobre imágenes del dataset y un caso sintético de NMS.
- `src/metrics.py::detection_f1` — completo, validado con un caso sintético calculado a mano antes de usarse sobre datos reales.
- `scripts/run_eda.py`, `scripts/run_detection.py` — funcionales, ya corridos de punta a punta.
- `REPORTE.md` — secciones 1 (Portada, con integrantes/nro. de grupo **pendientes de completar**), 2 (Análisis del dataset, completa), 3.1-3.2 (Metodología de detección y de la métrica, completa salvo el punto de resolución de entrada — ver sección 5), 4.1 (Resultados de detección, completa). Secciones 5 y 6 parciales/pendientes.
- Repo git: 4 commits, working tree limpio.

**Solo esqueleto (`raise NotImplementedError`), sin probar — todo lo que queda del plan:**
- `src/segmentation.py` (`segment_with_sam2`, `mask_to_rle`)
- `src/scene_builder.py` (`build_scene_json`)
- `src/properties.py` (`compute_centroid`, `compute_area_ratio`, `compute_position`, `compute_size`, `compute_dominant_color`, `compute_relations`) — las constantes `POSITIONS`/`SIZES`/`PREDICATES` ya están cargadas con los valores exactos del `scene_schema.json` real.
- `src/tracing.py` (`Tracer`)
- `src/qa_engine.py` (`answer_question`, `describe_scene`) — constante `QUESTION_TYPES` ya cargada.
- `src/qa_llm.py` (`interpret_query`, `generate_description_llm`)
- `src/metrics.py::mask_f1`, `scene_json_consistency`, `qa_accuracy`

## 4. Experimentos realizados

1. **EDA completo** sobre las 120 imágenes de desarrollo — ver `reports/eda/` y sección 2 de `REPORTE.md`.
2. **Detección baseline** sobre las 120 imágenes (`confidence_threshold=0.1`, `nms_iou_threshold=0.5`):
   - **F1=0.593, precision=0.512, recall=0.705** (ya supera el piso de promoción, ≥0.50).
   - TP=1262, FP=1203, FN=528, predicciones ignoradas por categoría no verificada=4828 (66% del total — esperado, porque se consultan las 120 categorías en cada imagen).
   - Tiempo: 366.7s total / 3.1s por imagen en **CPU local, sin batching** (en GPU de Kaggle debería ser bastante más rápido).
   - Resultados crudos en `reports/detection/predictions.json` y `reports/detection/detection_f1_summary.json`.

**No se corrió ninguna comparación controlada todavía** (ej. umbral de confianza alternativo, otra resolución de entrada, otro detector) — eso es la Tarea 12 del plan original y sigue pendiente.

## 5. Problemas conocidos / huecos detectados

1. **Resolución de entrada de OWLv2 sin justificar.** El enunciado exige justificar explícitamente este punto para el módulo de detección. Verificamos que OWLv2 usa 960×960 fijo (resize+padding, sin importar el tamaño original) por defecto del `processor`, pero no se comparó contra ninguna alternativa ni se escribió la justificación en el reporte. Es el hueco más concreto pendiente ahora mismo.
2. Discrepancia `submission.csv` (PDF) vs `solution.csv` (Kaggle) sin resolver — confirmar con la cátedra.
3. La carpeta original del proyecto estaba en un disco NTFS (`/mnt/datos/...`) que no soporta `chmod`; si en algún momento se vuelve a trabajar desde ese disco, git va a volver a fallar. Ya no aplica mientras el proyecto siga en `/home/jeronimo-osuna/IA/TPI` (filesystem nativo).
4. `~/.gitconfig` tiene entradas `safe.directory` de otro proyecto del usuario (`BitLogic/backoffice`) — no tocar, no están relacionadas con este repo.
5. Integrantes del grupo y número de grupo **sin completar** en `REPORTE.md` (necesario también para el nombre de archivo final `TPI_IA_2026P_[nro_de_grupo]`).

## 6. Entorno de desarrollo local (importante, no está en el repo)

- Ya existe un venv persistente en `~/IA/TPI/.venv` (creado por el usuario, gitignored) con `torch`, `transformers`, `scipy`, `pandas`, `matplotlib`, etc. ya instalados desde `requirements.txt` — **no hace falta reinstalar desde cero**, solo activar (`source .venv/bin/activate`) o usar `.venv/bin/python`.
- El `pip install` directo con `python3 -m pip` falla en este sistema (Debian/Ubuntu "externally-managed-environment") — siempre usar el venv.
- `data/development/` (120 imágenes + `annotations.json`), `data/tpi_ia_2026_competition_public/` (60 imágenes de test) y `data/tpi_2026_qa_public/` ya están descargados y organizados localmente, pero **gitignored** (pesan demasiado). `data/scene_schema.json` y `data/sample_submission.csv` sí están trackeados en git — son la referencia autoritativa del contrato, **no hace falta volver a scrapear Kaggle para leerlos**, ya están en el repo tal cual los bajó Kaggle.
- Internet sí está disponible desde este entorno (se pudo descargar el modelo OWLv2 de HuggingFace sin problema).

## 7. Tareas pendientes (en el orden del plan original)

1. Cerrar el hueco de resolución de entrada en `REPORTE.md` (sección 5 de este documento).
2. **Segmentación SAM2** (`src/segmentation.py`) — usar como prompts las bboxes ya detectadas en `reports/detection/predictions.json`.
3. Construcción de `SceneJSON` (`src/scene_builder.py`).
4. Propiedades derivadas de máscaras (`src/properties.py`): centroide, `area_ratio`, `position` (grilla 3×3), `size`, color dominante, relaciones espaciales.
5. Trazabilidad/logging (`src/tracing.py`).
6. Motor de QA determinista (`src/qa_engine.py`) — mínimo conteo y existencia.
7. Completar métricas: `mask_f1`, `scene_json_consistency`, `qa_accuracy` (`detection_f1` ya está hecho).
8. Generar y validar `submission.csv`/`solution.csv` final (80 filas).
9. LLM liviano para `questions_llm` y descripción (`src/qa_llm.py`) — nivel promoción.
10. Experimento controlado vs. línea base (ej. umbral de confianza alternativo) — Tarea 12 del plan, incluye resolver el punto 1 de esta lista con datos en vez de solo texto.
11. Reporte técnico final + consolidación de todo en una única notebook de Kaggle (el usuario la sube manualmente).

## 8. Próximo paso recomendado

**Segmentación con SAM2** (ítem 2 de la lista de arriba) — es el siguiente eslabón natural de la pipeline, ya hay bboxes reales en `reports/detection/predictions.json` para usar como prompts, y desbloquea tanto `scene_builder.py` como `properties.py`.

Antes de arrancar, vale la pena resolver rápido el punto 1 (resolución de entrada) porque es barato y cierra un requisito explícito del enunciado que quedó suelto.
