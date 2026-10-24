# TPI IA 2026 — Comprensión visual de escenas

> Reporte técnico en construcción. Se va completando a medida que avanza el desarrollo (ver commits del repo para el detalle incremental).

## 1. Portada

- **Competencia:** TPI IA 2026 Primavera — Comprensión Visual de Escenas (Kaggle, `tpi-ia-2026-primavera-comprension-visual-de-escenas`).
- **Cátedra:** Inteligencia Artificial, UTN.
- **Integrantes:** _(completar)_
- **Nro. de grupo:** _(completar, para el nombre de entrega `TPI_IA_2026P_[nro_de_grupo]`)_

## 2. Análisis del dataset

Dataset de desarrollo: 120 imágenes anotadas en formato COCO, derivado de LVIS v1 val (120 categorías evaluables), con extensiones LVIS por imagen (`neg_category_ids`, `not_exhaustive_category_ids`). Detalle completo y gráficos en [`reports/eda/`](reports/eda).

**Categorías.** Cola larga marcada: top (`book` 46, `car` 44, `box`/`chair` 40, `wheel` 37 instancias) vs. categorías con apenas 3-5 instancias en todo el dataset (`airplane`, `surfboard`, `zebra`). Ninguna categoría tiene 0 o 1 instancia, pero la mitad inferior del ranking tiene <10 instancias. **Decisión:** no afinar el umbral de confianza por categoría (no hay señal suficiente para las categorías de cola larga); usar un umbral global.

**Tamaño relativo de objetos (`area_ratio`).** Mediana ≈ 0.003 (el objeto típico ocupa <1% de la imagen); 71% de las instancias tiene `area_ratio<0.01`, 29% tiene `area_ratio<0.001`. Terciles lineales (0.0013 / 0.0076) son poco interpretables por lo sesgado de la distribución — conviene documentar cortes log-espaciados o basados en criterio experto para el enum `size` del `SceneJSON`, no terciles crudos. **Implicancia:** mayoría de objetos chicos es el escenario más difícil tanto para el detector (recall) como para SAM2 (prompts de caja chica).

**Instancias por imagen.** Media 14.9, mediana 10.5, máximo 66 en una sola imagen — bastante crowding en varias imágenes.

**Oclusión / crowding (proxy: IoU máx. con vecino bbox).** 41% de las instancias sin vecino cercano (IoU≈0); ~13% con IoU>0.3 con otra instancia. Este crowding va a complicar tanto los prompts de SAM2 (cajas superpuestas) como el cálculo de relaciones espaciales (`overlaps` va a disparar seguido).

**Política LVIS (`neg_category_ids` / `not_exhaustive_category_ids`).** 74/120 imágenes tienen categorías negativas verificadas (top: `sofa`, `clock`, `ski`) y 36/120 tienen categorías no-exhaustivas (`strap`, `boat`, `car`, `book`). **Implicancia directa para la métrica de detección:** hay que excluir sistemáticamente estas combinaciones imagen-categoría del cálculo de FP/FN (implementado en `src/metrics.detection_f1`, ver sección 3).

## 3. Metodología

### 3.1 Detección (open-vocabulary)

Detector: **OWLv2** (`google/owlv2-base-patch16-ensemble`, vía `transformers`). Elegido sobre Grounding DINO por ser puro Python/pip-instalable (sin compilar ops CUDA), lo que lo hace más confiable de correr en un entorno Kaggle. Soporta vocabulario de texto libre, por lo que se le pasan las 120 categorías del dataset como consultas de texto en un único forward pass por imagen.

- `src/detection.py::detect_objects`: corre el modelo con el vocabulario completo, devuelve `{category_id, confidence, bbox}` por detección cruda.
- `src/detection.py::suppress_duplicates`: NMS por categoría (no cross-categoría, para no suprimir objetos distintos que se superponen legítimamente, ej. "persona" y "mochila").

**Baseline elegido:** `confidence_threshold=0.1`, `nms_iou_threshold=0.5`. Es un umbral permisivo (prioriza recall) elegido como punto de partida dado que la mayoría de los objetos del dataset son chicos (ver EDA) — se compara contra una alternativa más conservadora en el proceso experimental (sección 4/5).

**Resolución de entrada: 960×960 (la nativa del checkpoint, sin modificar).** `Owlv2ImageProcessorPil.resize` redimensiona cada imagen a `(960, 960)` de forma directa — se verificó en el código fuente que **no hace letterbox/padding preservando proporción**, sino un resize no uniforme (estira ancho y alto por separado hasta el cuadrado objetivo). Se decidió no override-earla por dos motivos:

1. El backbone ViT del checkpoint `owlv2-base-patch16-ensemble` fue entrenado/fine-tuneado específicamente a esa resolución; los embeddings posicionales y el grid de patches están calibrados para ese tamaño. Cambiarla implicaría interpolar embeddings posicionales fuera de la distribución de entrenamiento, sin ninguna re-calibración de nuestro lado — riesgo de degradar la calidad de las cajas sin evidencia de que mejore.
2. El dataset de desarrollo tiene relación de aspecto muy variada (mínimo 0.45, máximo 1.78, mediana 1.33 — solo 4 de 120 imágenes son aproximadamente cuadradas), así que el estiramiento a 960×960 **distorsiona la imagen en la mayoría de los casos**. Es una limitación conocida y aceptada del modelo preentrenado, no de nuestra implementación: pre-procesar nosotros mismos con letterbox antes de pasarle la imagen al processor cambiaría la distribución de entrada respecto a cómo se entrenó el modelo, con el mismo riesgo del punto anterior.

Por indicación de la cátedra, la consigna exige **justificar** la resolución de entrada pero no obliga a experimentar específicamente con ella — la comparación controlada del proceso experimental (sección 4/5) evalúa en cambio el umbral de confianza, que es el parámetro donde el EDA (objetos predominantemente chicos) dio evidencia directa y accionable para comparar alternativas.

### 3.2 Métrica de detección

`src/metrics.py::detection_f1` implementa la política LVIS-aware descripta en el README del dataset:

- Categoría con anotaciones en la imagen (**positiva**): matching greedy predicción↔GT por mayor IoU, umbral IoU≥0.5. Match → TP; GT sin matchear → FN; predicción sin matchear → FP, salvo que la categoría esté en `not_exhaustive_category_ids` de esa imagen, en cuyo caso se ignora (no se penaliza con FP).
- Categoría en `neg_category_ids` (**negativa verificada**, sin anotaciones): toda predicción de esa categoría en esa imagen es FP.
- Categoría ni positiva ni negativa (**no verificada**): toda predicción se ignora (no suma a FP ni FN), porque la ausencia de anotación no prueba que el objeto no exista.

Validado con un caso sintético de conteo manual antes de usarse sobre datos reales.

### 3.3 Segmentación (SAM2)

Modelo: **SAM2.1** (`facebook/sam2.1-hiera-tiny`), vía la integración nativa en `transformers` (`Sam2Model`/`Sam2Processor`) — misma lógica que con el detector: evita clonar/compilar el repo oficial de Meta (`facebookresearch/sam2`), que no es pip-instalable de forma simple. Se eligió el checkpoint **`tiny`** (el más chico de la familia SAM2.1) para minimizar tiempo de inferencia, dado el volumen de cajas a segmentar (una por cada detección post-NMS: 7293 en todo el dataset de desarrollo).

- `src/segmentation.py::segment_with_sam2`: recibe una imagen y una lista de cajas (`xywh`, mismo formato que `detection.py`), las pasa como *prompts* de caja al modelo en un único forward por imagen (todas las cajas de esa imagen juntas, no una por una), y devuelve una máscara booleana por caja del mismo tamaño que la imagen original. Se usa `multimask_output=False` (una sola máscara "mejor" por caja, no las 3 variantes que SAM2 puede proponer por ambigüedad).
- `src/segmentation.py::mask_to_rle` / `rle_to_mask`: codificación/decodificación RLE no comprimida en orden column-major (Fortran), tal como exige `scene_schema.json` (`mask.size`, `mask.counts`). Implementación propia (no se usó `pycocotools` para esto, para tener control total del orden de recorrido exigido por el schema).

Verificado con inferencia real: se segmentaron cajas detectadas de verdad sobre una imagen del dataset (áreas de máscara resultantes siempre menores pero coherentes con el área de la caja que las originó, como se espera de una segmentación real del objeto dentro de su bounding box). La codificación RLE se validó con un **round-trip** (`mask_to_rle` → `rle_to_mask` → comparación con la máscara original) sobre las máscaras reales obtenidas y sobre casos borde (máscara vacía, máscara completa, lista de cajas vacía).

### 3.4 Construcción de `SceneJSON` (ensamblado, sin propiedades derivadas todavía)

`src/scene_builder.py::build_scene_json` arma el esqueleto estructural (`image`, `objects`, `relations`) a partir de los objetos ya detectados+segmentados (`category_id`, `confidence`, `bbox`, `mask`), asignando a cada uno un `id` string único (`{image_id}_obj{n}`). **No calcula `centroid`/`area_ratio`/`position`/`size`** — esos campos, requeridos por `scene_schema.json`, los agrega `src/properties.py` (Tarea 6, siguiente paso) sobre esta estructura ya armada; `relations` empieza vacío por el mismo motivo (se completa en base a los centroides/bboxes una vez calculados). Es decir, el `SceneJSON` que produce este módulo por sí solo **todavía no valida contra el schema completo** — es un paso intermedio del pipeline, no el artefacto final.

`image.id` se construye a partir del nombre de archivo sin extensión (ej. `tpi_0001`), que es exactamente el formato que usa `sample_submission.csv` en la columna `image_id` (verificado comparando ambos archivos).

### 3.5 Propiedades derivadas y relaciones espaciales

> Nota de numeración: el enunciado no trata esto como una etapa aparte — es la segunda mitad de su "Etapa 3: Segmentación de instancias con SAM2" ("cada grupo deberá utilizar las máscaras para derivar propiedades..."). Acá lo separamos en su propia subsección (3.5) solo por claridad expositiva del reporte.

`src/properties.py` completa los campos que faltaban en los objetos armados por `scene_builder.py`:

- **`centroid`**: centro de masa de los píxeles de la máscara (no del bounding box), vía `np.nonzero`.
- **`area_ratio`**: área de la máscara (conteo de píxeles) sobre área total de la imagen.
- **`position`**: la grilla 3×3 oficial del schema, por cuadrante del centroide.
- **`size`**: `small`/`medium`/`large` con cortes en **0.00126 y 0.00765** — no son números arbitrarios, son los terciles empíricos exactos de `area_ratio` calculados sobre las 1790 anotaciones reales del dataset de desarrollo (`reports/eda/object_sizes.csv`), elegidos así para que las tres clases queden razonablemente balanceadas en vez de usar cortes redondos sin respaldo en los datos.
- **`dominant_color`** (atributo extra, no exigido por los 9 campos mínimos del schema pero sí permitido por `additionalProperties: true`): color dominante real dentro de la máscara, por cuantización de los píxeles en bins de 24 valores por canal y selección del bin más frecuente (no un promedio/mediana, que podría caer en un color que no existe realmente en el objeto).
- **`compute_relations`**: a diferencia de un primer intento que evaluaba los *O(n²)* pares de objetos de cada imagen (en la imagen más poblada, 257 objetos → 12499 relaciones, mayormente ruido entre objetos lejanos y no relacionados que solo compartían fila o columna por casualidad), la versión final **limita cada objeto a sus 5 vecinos más cercanos** por distancia de centroides antes de evaluar `overlaps`/`left_of`/`above` (con `right_of`/`below` quedando implícitos como la relación inversa, a cargo del motor de QA). Esto bajó el promedio a 189.5 relaciones por imagen (máximo 819, muy por debajo del límite de 20000 del schema) y las deja centradas en pares efectivamente cercanos — más alineado con el criterio de "atributos y relaciones **útiles**" que pide la promoción.

### 3.6 Trazabilidad de la pipeline

> Nota de numeración: en el enunciado esta es su propia "Etapa 5", separada de la segmentación/representación.

`src/tracing.py::Tracer` escribe una línea JSON por paso (formato *JSON Lines*: auditable, se puede `grep`-ear y se puede recargar sin parsear el archivo entero) en `reports/trace/pipeline.jsonl`. Cada registro tiene `timestamp`, `run_id` (para distinguir corridas distintas que comparten el mismo archivo), `image_id`, `module`, `status` (`ok`/`error`/`skipped`), `duration_seconds` y los `params` relevantes de ese paso — nunca arrays completos ni imágenes, solo metadatos.

Además del método explícito `log_step`, `Tracer.step(...)` es un *context manager* que mide el tiempo automáticamente y, si el bloque lanza una excepción, registra `status="error"` con el mensaje **y vuelve a lanzar la excepción** (no la traga en silencio) — así un paso que falla sigue fallando de forma visible, pero además queda localizado en la traza, que es exactamente el objetivo que pide el enunciado ("reconstruir cómo se obtuvo un resultado y localizar en qué etapa se originó o propagó un error").

Se instrumentó `scripts/run_eda.py`, `scripts/run_detection.py`, `scripts/run_segmentation.py`, `scripts/build_scenes.py` y `scripts/enrich_scenes.py`. **Importante:** `run_detection.py` y `run_segmentation.py` se instrumentaron en el código pero **no se volvieron a correr** (ya habían corrido antes de agregar tracing, y repetirlos solo para generar trazas hubiera costado ~15 minutos sin aportar información nueva) — sus resultados en `reports/detection/` y `reports/segmentation/` siguen siendo válidos, simplemente no tienen entradas en `reports/trace/pipeline.jsonl` todavía. Los tres pasos más baratos (`eda`, `scene_builder`, `properties`+`schema_validation`) sí se re-corrieron con tracing activo y produjeron resultados **idénticos** a las corridas anteriores sin tracing (buena señal de reproducibilidad).

### 3.7 Preguntas y respuestas — modo determinista

> Nota de numeración: Etapa 6 del enunciado.

`src/qa_engine.py::answer_question` recibe una fila de `questions_dev.jsonl` (`question_id`, `type`, `question`, `class_en`/`class_es`, sin pasar por ningún LLM) y la lista de categorías, y construye la respuesta consultando directamente el `SceneJSON` ya enriquecido. Decisiones de diseño, **todas consultadas con el usuario antes de implementar**:

- **Resolución de categoría**: `class_en` se matchea contra `category_id` vía el mismo patrón `name→id` que ya usa `detection.py` (verificado: las 40 referencias `class_en` de las 60 preguntas matchean exacto contra los nombres de `annotations.json["categories"]`, 0 faltantes).
- **Predicado de `relation`**: el campo estructurado **no incluye el predicado** (`above`/`left_of`/etc.) por separado — solo viene embebido en el texto ("está por encima de"). Se resuelve con **matching de frases fijas por regex** sobre las 5 frases conocidas en español (verificadas contra las 10 preguntas `relation` reales del dataset), no con un LLM — decisión consultada y confirmada con el usuario.
- **Preguntas que presuponen una única instancia** (`position`, `size`, y los dos objetos de `relation`, todas frases con "la instancia de X" en singular): si hay 0 matches, no aplica; si hay **más de 1 match, es ambiguo** (no hay forma de saber cuál instancia se pregunta) — ambos casos devuelven `unsupported`, nunca una adivinanza. `count`/`existence` no tienen este problema porque la multiplicidad es justamente lo que preguntan.
- **`confidence`**: mínimo de las confianzas de detección de los objetos involucrados en la respuesta (conservador — confirmado con el usuario).
- **`unsupported`**: solo por falta de información estructural real (categoría no resuelve, tipo no soportado, predicado no identificable, o ambigüedad de instancia única). **Conteo/existencia en 0 es una respuesta válida**, no `unsupported` — confirmado con el usuario.
- **`description`**: plantilla determinista (no LLM, por eso vive en `qa_engine.py` y no en `qa_llm.py`) que arma una oración con las 5 categorías más frecuentes de la escena y sus conteos.

El cálculo del predicado de relación (`_evaluate_predicate`) reutiliza directamente la geometría de `bbox_iou`/`bbox_to_corners` **sobre los dos objetos puntuales de la pregunta**, en vez de consultar el array `relations` ya precalculado de la escena — ese array está intencionalmente podado a los 5 vecinos más cercanos por objeto (ver 3.5) para que sea *útil* como listado general, pero eso significa que puede no incluir el par exacto que pregunta una consigna puntual. Para responder con precisión una pregunta específica, más vale recalcular directo sobre las dos cajas en cuestión.

### 3.8 Preguntas y respuestas — modo con LLM (promoción)

> Nota de numeración: Etapa 7 del enunciado.

**Modelo: Qwen2.5-1.5B-Instruct** (HF `transformers`, mismo patrón de carga que OWLv2/SAM2). Decisión tomada con evidencia, no a priori — se probó primero **Qwen2.5-0.5B-Instruct** (más liviano) y dio solo **22% de `class_en` exacto** contra el vocabulario de 120 categorías (el modelo frecuentemente deja la palabra en español sin traducir, ej. "tomacorriente" en vez de "wall socket", o confunde categorías parecidas). Con Qwen2.5-1.5B-Instruct ese número sube a **67.5%** (ver 4.8) — mejora sustancial que justifica el costo extra de inferencia (~27s/pregunta en CPU local vs ~1s de la versión 0.5B; en la GPU gratuita de Kaggle esto debería bajar bastante). Ambas pruebas y la decisión final fueron consultadas con el usuario antes de implementar.

**`src/qa_llm.py::interpret_query`**: recibe **únicamente el texto de la pregunta** (nunca `type`/`class_en`/`class_es` del registro de la cátedra, que quedan reservados al modo determinista) y produce un JSON intermedio `{"type", "class_en", "predicate"?}`. El prompt incluye la lista completa de las 120 categorías válidas (para que el modelo elija de ahí, no para "ayudarlo a razonar" sobre la pregunta puntual) y seis ejemplos *few-shot*, uno por cada `type`.

**Validación antes de ejecutar** (`_is_valid_interpretation`): `type` debe ser uno de los 6 válidos, cada `class_en` debe matchear exacto contra el vocabulario conocido, y `predicate` (si `type="relation"`) debe ser uno de los 5 válidos. Si algo no valida, la pregunta se marca `unsupported` directamente — **nunca se ejecuta una consulta con una categoría o predicado inventado**.

**Reutilización del motor determinista**: el JSON validado se empaqueta como un `question_record` sintético y se pasa a `qa_engine.answer_question` (la misma función del modo determinista) — es el mismo motor de ejecución para ambos modos, tal como exige el enunciado. Para esto se extendió `qa_engine._answer_relation` para aceptar un `predicate` ya resuelto (en vez de extraerlo por regex del texto), ya que en modo LLM el predicado lo identifica el propio modelo.

**`generate_description_llm`**: el prompt recibe solo pares categoría+conteo agregados (top 8, no la lista completa de instancias — evita prompts gigantes en imágenes con cientos de objetos), con instrucción explícita de no inventar nada que no esté en esa lista. **Resultado real: la instrucción no se respeta de forma confiable** (ver 4.8) — es la limitación más seria medida en esta sesión, documentada en detalle ahí y en la sección 5.

## 4. Resultados

### 4.1 Detección — baseline (`confidence_threshold=0.1`, `nms_iou_threshold=0.5`)

Corrida sobre las 120 imágenes de desarrollo (`scripts/run_detection.py`, resultados en [`reports/detection/`](reports/detection)):

| Métrica | Valor |
|---|---|
| Precision | 0.512 |
| Recall | 0.705 |
| **F1** | **0.593** |
| TP / FP / FN | 1262 / 1203 / 528 |
| Predicciones ignoradas (categoría no verificada) | 4828 |
| Predicciones totales | 7293 |
| Tiempo total / promedio por imagen | 366.7s / 3.1s (CPU local, sin batching) |

El F1 baseline (0.593) ya supera el umbral de **promoción** (≥0.50) y el de **aprobado** (≥0.40) definidos en el enunciado, calculado localmente sobre el conjunto anotado de desarrollo con IoU≥0.5 y la política LVIS.

De las 7293 predicciones, **4828 (66%) se ignoran** por corresponder a categorías "no verificadas" en esa imagen (ni positivas ni negativas) — es esperable, porque el detector consulta las 120 categorías en cada imagen aunque la mayoría no aplique a esa escena en particular. La política LVIS evita que eso infle artificialmente los falsos positivos.

### 4.2 Segmentación — SAM2.1 tiny sobre las 7293 cajas detectadas

Corrida sobre las 120 imágenes (`scripts/run_segmentation.py`, resultados en [`reports/segmentation/`](reports/segmentation)), usando como *prompts* las cajas ya detectadas (post-NMS) en 4.1:

| Métrica | Valor |
|---|---|
| Objetos segmentados | 7293 / 7293 (100%, ninguna máscara vacía) |
| Tiempo total | 541.6s (≈9 min, CPU local) |
| Promedio por imagen (con cajas) | 3.6s |
| `mask_area / bbox_area` — media / mediana | 0.634 / 0.646 |
| `mask_area / bbox_area` — percentil 10 / 90 | 0.332 / 0.897 |

El tiempo real fue bastante menor al estimado inicialmente (~9 min vs. una proyección de ~20-25 min calculada a partir de una sola caja aislada): SAM2 procesa **todas las cajas de una imagen en un único forward batcheado**, no una por una, así que el costo no escala linealmente con la cantidad de cajas por imagen.

La relación `mask_area/bbox_area` (media 0.63) confirma que SAM2 está recortando la forma real del objeto dentro de la caja, no simplemente rellenándola — coherente con lo esperado de un segmentador real. La codificación RLE se validó también sobre los 7293 objetos reales: `sum(counts) == height×width` se cumple en el 100% de una muestra verificada.

### 4.3 Construcción de `SceneJSON`

`scripts/build_scenes.py` combinó las detecciones+máscaras de las 120 imágenes en `reports/scene_json/scenes.json` (120 escenas, 7293 objetos en total, **0 escenas vacías** — todas las imágenes del dataset de desarrollo tienen al menos un objeto detectado). Cada `id` de objeto es único dentro de su escena; `image.id` coincide exactamente con la columna `image_id` de `sample_submission.csv` para las filas de desarrollo.

### 4.4 Propiedades derivadas, relaciones y validación de schema

`scripts/enrich_scenes.py` completó las 120 escenas con `centroid`/`area_ratio`/`position`/`size`/`dominant_color`/`relations` y **validó cada una contra el `scene_schema.json` real** usando la librería `jsonschema` (no una verificación manual nuestra — el validador oficial del contrato):

| Métrica | Valor |
|---|---|
| Escenas válidas contra `scene_schema.json` | **120 / 120 (100%)** |
| Objetos enriquecidos | 7293 |
| Distribución de `size` | small=2671, medium=2613, large=2009 |
| Relaciones totales (tras limitar a 5 vecinos más cercanos) | 22740 (antes: 152906 sin ese límite) |
| Relaciones por imagen — promedio / máximo | 189.5 / 819 (cap del schema: 20000) |
| Predicados emitidos | `overlaps`=11574, `left_of`=6094, `above`=5072 (`right_of`/`below` no se materializan, quedan implícitos) |

La distribución de `size` quedó razonablemente balanceada (2671/2613/2009) a pesar de que los cortes se calcularon sobre las anotaciones reales (ground truth) y se aplican sobre objetos *detectados* (población ligeramente distinta) — buena señal de que el corte generaliza. `overlaps` domina los predicados, coherente con el crowding que ya había detectado el EDA (13% de instancias con IoU>0.3 con su vecino).

### 4.5 Trazabilidad

Re-correr `run_eda.py`, `build_scenes.py` y `enrich_scenes.py` con `Tracer` activo produjo **361 registros reales** en `reports/trace/pipeline.jsonl`, con resultados idénticos a las corridas anteriores (confirma reproducibilidad):

| Módulo | Registros | Status |
|---|---|---|
| `eda` | 1 (todo el dataset de una vez, no es por imagen) | 120 imágenes, 1.1s |
| `scene_builder` | 120 (uno por imagen) | 100% `ok` |
| `properties` | 120 | 100% `ok`, 22.1s totales, 0.18s/imagen promedio |
| `schema_validation` | 120 | 100% `ok` |

El camino de error también se verificó (no solo el camino feliz): un test sintético que fuerza una excepción dentro de `Tracer.step(...)` confirma que el paso queda registrado con `status="error"` y el mensaje de la excepción, **y la excepción se vuelve a lanzar** — no queda enmascarada.

### 4.6 Preguntas y respuestas — modo determinista

Corrida sobre las **60 consignas reales** de `questions_dev.jsonl` contra los `SceneJSON` ya validados (`scripts/run_qa.py`, resultados en [`reports/qa/`](reports/qa)):

| Tipo | Total | `unsupported` |
|---|---|---|
| `description` | 20 | 0 |
| `count` | 10 | 0 |
| `existence` | 10 | 0 |
| `relation` | 10 | 5 |
| `position` | 5 | 3 |
| `size` | 5 | 3 |

**Precisión real, comparada contra el ground truth de `annotations.json`** (uso permitido por el enunciado para evaluación, no se usó para construir las respuestas):

| Métrica | Valor |
|---|---|
| `count` — exact match | **3 / 10 (30%)** |
| `existence` — accuracy | **9 / 10 (90%)** |

**Hallazgo relevante para el análisis de errores (Etapa 8):** el 30% de exact-match en `count` es bajo y **no es un bug de `qa_engine.py`** — es la métrica de detección (F1=0.593, precision=0.512) **propagándose** a la capa de preguntas: el motor de QA cuenta exactamente los objetos que el detector encontró, así que un `count` erróneo casi siempre refleja sobre-conteo (falsos positivos no filtrados por NMS) o sub-conteo (falsos negativos) en la etapa de detección, no un error de consulta. `existence` es mucho más tolerante a este ruido (90%) porque solo necesita `count>0`, no el número exacto — un solo caso falló por un falso positivo del detector (`DEV-23-Q1`: predijo `yes` cuando el ground truth es `no`).

De los 11 casos `unsupported` en `position`/`size`/`relation`, **9 son por ambigüedad** (el detector encontró más de una instancia de una categoría que la pregunta presupone única — ej. "tabla de surf" con 10 matches en una imagen) y solo 2 por falta de detección real (`strap`, `spectacles` con 0 matches). Esto es una **consecuencia directa y medible del umbral de confianza permisivo (0.1)** elegido en la Etapa 2 para maximizar recall: ese mismo umbral que ayuda al F1 de detección perjudica la tasa de respuestas resueltas en QA, porque genera más instancias "de más" que rompen el supuesto de unicidad. Es un argumento concreto a favor de priorizar el experimento controlado de umbral de confianza (Tarea 12) — no solo por el F1 de detección, sino por su efecto en cascada sobre QA.

### 4.7 Proceso experimental — línea base vs. umbral de confianza alternativo

Línea base: `confidence_threshold=0.1` (secciones 4.1/4.2/4.3/4.4/4.6). Alternativa concreta de mejora, motivada por el hallazgo de 4.6: `confidence_threshold=0.3`, manteniendo todo lo demás igual (`nms_iou_threshold=0.5`, mismo detector, mismo checkpoint de SAM2). Pipeline completa (detección→segmentación→SceneJSON→propiedades→QA) corrida de nuevo sobre las 120 imágenes con el umbral nuevo (`scripts/run_threshold_experiment.py`, resultados en [`reports/experiments/threshold_0.3/`](reports/experiments/threshold_0.3)):

| Métrica | Baseline (0.1) | Alternativa (0.3) |
|---|---|---|
| Predicciones totales | 7293 | 1585 |
| Precision | 0.512 | **0.854** |
| Recall | **0.705** | 0.436 |
| **F1 detección** | **0.593** | 0.577 |
| Escenas válidas contra schema | 120/120 | 120/120 |
| `count` exact-match | 3/10 | 3/10 |
| `existence` accuracy | 9/10 | **10/10** |
| `unsupported` (position+size+relation) | 11/20 (55%) | **6/20 (30%)** |

**Conclusión, sin simplificar de más:** subir el umbral **no es una mejora neta simple** — confirma la hipótesis de 4.6 (la ambigüedad en QA bajó de 11 a 6 casos, y `existence` mejoró a 10/10), pero **el F1 de detección bajó levemente** (0.593→0.577): el salto de precision (+0.34) no compensa la caída de recall (-0.27) en términos de F1 armónico. `count` exact-match no cambió (3/10 en ambos) — tener menos falsos positivos no alcanzó para compensar los nuevos falsos negativos en el conteo.

Dado que el `PipelineScore` de la competencia pondera **40% F1 de cajas + 35% F1 de máscaras** contra solo **10% QA**, el componente de detección domina el puntaje por un margen grande — así que, mirando el leaderboard, **el baseline (0.1) sigue siendo la mejor elección global** pese a que 0.3 da mejores respuestas de QA. Esto es justo el tipo de trade-off entre etapas que pide analizar el enunciado: la métrica "más alta" en una etapa aislada (QA) no necesariamente es la mejor decisión para el sistema completo.

### 4.8 Preguntas y respuestas — modo con LLM (Qwen2.5-1.5B-Instruct)

Corrida sobre las mismas 60 preguntas reales (`scripts/run_qa_llm.py`, resultados en [`reports/qa/`](reports/qa)), 1914.5s totales (~32 min, CPU local):

| Métrica | Determinista | Modo LLM |
|---|---|---|
| `type` interpretado correctamente | — (dado, no se interpreta) | **59/60 (98.3%)** |
| `class_en` interpretado = exacto | — (dado) | **27/40 (67.5%)** |
| `count` exact-match | 3/10 (30%) | **1/10 (10%)** |
| `existence` accuracy | 9/10 (90%) | **8/10 (80%)** |
| `unsupported` total | 11/20 | 12/60* |

\* el modo LLM tiene preguntas `description` (20) que el determinista también responde pero nunca marca `unsupported` (es una plantilla, siempre produce algo) — comparando manzanas con manzanas, el `unsupported` en `position`/`size`/`relation` fue **11/20 en ambos modos, exactamente igual** (más 1 nuevo caso en `count` por una confusión de categoría del LLM — confundió "clock" con "watch", categorías distintas en nuestro vocabulario).

**Hallazgo 1 — la ambigüedad del SceneJSON domina sobre el modo de interpretación.** Que `position`/`size`/`relation` den exactamente el mismo `unsupported` (11/20) en ambos modos es una confirmación fuerte de que esa ambigüedad viene de la escena predicha (detector encuentra >1 instancia), no de cómo se interpreta la pregunta — cuando el LLM identifica bien la categoría, pega contra la misma pared que el modo determinista.

**Hallazgo 2 — el modo LLM es, como es esperable, menos preciso que el determinista en las métricas cuantitativas** (`count` 10% vs 30%, `existence` 80% vs 90%): el modo LLM acumula el error de detección **más** un nuevo error de interpretación (traducción/categorización incorrecta, 32.5% de las veces). Esto no es una falla de diseño — es el costo esperado y documentado de interpretar lenguaje natural con un modelo liviano en vez de recibir la categoría ya resuelta.

**Hallazgo 3 — la descripción generada por LLM no respeta de forma confiable la instrucción de "no inventar" información, pese a estar instruida explícitamente para no hacerlo.** Verificación cuantitativa: **10 de las 20 descripciones (50%) mencionan al menos un color**, pese a que el prompt **nunca** incluye información de color (solo pares categoría+conteo). Ejemplos reales:

- **Fabricación severa (inventa objetos no detectados):** para `DEV-6-DESC`, el input fue `8 strap, 5 necklace, 4 handbag, 4 ring, 2 bus, 2 earring, 2 wheel, 2 bench` — el modelo generó *"...varios objetos: 4 anillos, 2 bolsas portátiles, 2 botellas de agua, 2 carteras, **2 perros, 2 coches**, 2 pendientes y **2 teléfonos móviles**"* — "perros" (dogs), "coches" (cars) y "teléfonos móviles" (cell phones) **no están en ningún lado de la lista de entrada**.
- **Fabricación moderada (colores y escenario inventados):** para `DEV-2-DESC` (input: `10 flower arrangement, 6 person, 6 shirt, 3 sock...`), el modelo inventó *"salón de una boutique elegante"*, *"traje oscuro"*, *"vestido azul marino"* — ninguno de esos detalles de color/ambientación está en los datos.
- **Casos fieles (existen, son la minoría):** `DEV-10-DESC` y `DEV-11-DESC` se mantuvieron cerca de la lista real sin inventar colores ni objetos extra, mostrando que el modelo *puede* ser fiel, solo que no lo es de forma consistente.

Esta es, de lejos, **la limitación más seria medida en esta sesión** — más que la imprecisión de `class_en` o el F1 más bajo en QA, porque una descripción con datos inventados es información falsa presentada como factual, exactamente lo que el enunciado pide evitar ("una respuesta incompleta pero sustentada es preferible a una respuesta fluida inventada"). Con un modelo de 1.5B parámetros y este prompt, esa garantía no se puede dar por sentada — mitigarlo (prompts más restrictivos, post-procesamiento que verifique cada sustantivo contra la lista de categorías, o un modelo más grande) queda fuera del alcance de esta sesión.

### 4.9 Entrega final (`submission.csv`)

`scripts/build_submission.py` arma el archivo de 80 filas: reutiliza las 20 escenas de desarrollo ya construidas y validadas (`reports/scene_json/scenes_enriched.json`) agregándoles `questions`/`questions_llm` (ya calculados en 4.6/4.8), y corre la pipeline completa (detección→segmentación→`scene_builder`→`properties`) **por primera vez sobre las 60 imágenes de test**, que hasta este punto de la sesión nunca se habían tocado.

| Métrica | Valor |
|---|---|
| Filas totales | **80/80** (60 test + 20 desarrollo) |
| Escenas válidas contra `scene_schema.json` | **80/80 (100%)** |
| `image_id` coincide con `sample_submission.csv` | Verificado exacto (mismo conjunto de 80 ids) |

Como el PDF del enunciado llama al archivo `submission.csv` y la página de Kaggle lo llama `solution.csv` (discrepancia detectada y nunca resuelta por la cátedra, ver sección 1 y `HANDOFF.md`), se generaron **ambos archivos con contenido idéntico** en vez de apostar a uno — así no hace falta renombrar nada a último momento sea cual sea el que Kaggle termine esperando.

Nota práctica para quien lea el CSV: algunas filas superan el límite por defecto de `csv.field_size_limit` de Python (imágenes muy pobladas, con RLE de más de 100 máscaras) — hay que llamar `csv.field_size_limit(sys.maxsize)` antes de parsearlo, si no `csv.DictReader` tira `_csv.Error: field larger than field limit`.

## 5. Discusión y análisis de errores

**Precision (0.512) bastante más baja que recall (0.705) en el baseline:** con un umbral de confianza permisivo (0.1) el detector prioriza no perderse objetos a costa de más falsos positivos — esperable dado que el EDA mostró que la mayoría de los objetos son chicos (mediana `area_ratio`≈0.003) y por lo tanto difíciles de detectar con confianza alta. La comparación controlada (sección 4.7) confirmó que subir el umbral a 0.3 sí mejora precision (0.854) pero **no mejora el F1 compuesto** (0.577 vs 0.593) — el baseline queda como mejor elección para el leaderboard pese a tener peor precision aislada.

**Error de propagación detección→QA (ver detalle en 4.6 y 4.7):** el 70% de las preguntas de `count` no da el número exacto en ninguno de los dos umbrales probados, y en el baseline 9 de los 11 `unsupported` en `position`/`size`/`relation` son por ambigüedad (más de una instancia detectada donde se presupone una sola) — ambos son consecuencia del umbral de confianza, no errores de lógica en `qa_engine.py`. El experimento de 4.7 confirma la dirección del efecto (subir el umbral baja la ambigüedad de 11 a 6 casos) pero también muestra que no es gratis: ese mismo cambio empeora el F1 de detección, que pesa mucho más en el `PipelineScore` (75%) que QA (10%).

**Alucinación en la descripción generada por LLM (ver detalle en 4.8):** el hallazgo más serio de esta sesión. Pese a una instrucción explícita de no inventar información, **50% de las 20 descripciones generadas mencionan colores** que nunca se les dieron, y al menos un caso (`DEV-6-DESC`) inventa directamente objetos ("perros", "coches", "teléfonos móviles") ausentes de la lista de entrada. No es un bug de `qa_llm.py` — es una limitación real y medida de usar un modelo de 1.5B parámetros para generación de texto con grounding estricto. Es el riesgo más concreto del nivel de promoción: una respuesta de QA mal resuelta se marca `unsupported` (seguro), pero una descripción alucinada se entrega como si fuera verdadera.

_(pendiente: inspección cualitativa de casos fallidos — FP sistemáticos por categoría, FN concentrados en objetos muy chicos o muy ocluidos)_

## 6. Conclusiones y trabajo futuro

**Nivel alcanzado (a confirmar en la defensa oral):** las 9 etapas de "Desarrollo del trabajo" del enunciado están implementadas y corridas con datos reales — F1 de detección 0.593 (≥0.40 aprobado, ≥0.50 promoción), pipeline modularizada y trazada, `SceneJSON` 100% válido contra el schema oficial en las 80 filas de entrega, `questions` determinista y `questions_llm` con LLM ambos poblados para las 20 imágenes de desarrollo, y una comparación controlada con conclusión explícita (sección 4.7). Lo que falta para dar por "cerrado" el nivel de promoción con confianza son las funciones de métrica reutilizables (`mask_f1`, `scene_json_consistency`, `qa_accuracy` propias en `metrics.py` — hoy ese cálculo existe pero disperso en varios scripts) y completar Portada (integrantes, nro. de grupo).

**Lo que mejor funcionó:**
- El detector + SAM2 + `jsonschema` dieron una representación estructurada **100% válida** de punta a punta (80/80 filas), sin ningún caso de JSON malformado o RLE inconsistente.
- La arquitectura de dos pasos (`scene_builder` ensambla, `properties` enriquece) y la reutilización del mismo motor de ejecución entre modo determinista y modo LLM (`qa_engine.answer_question` llamado desde ambos) se sostuvieron sin fricción al agregar funcionalidad nueva.
- El criterio de "ambigüedad → `unsupported`" en vez de adivinar resultó ser una decisión robusta: se mantuvo consistente entre modo determinista y modo LLM (11/20 en ambos), lo cual terminó siendo evidencia útil para diagnosticar que la ambigüedad viene del SceneJSON, no de cómo se interpreta la pregunta.

**Lo que no funcionó tan bien, documentado sin maquillar:**
- El umbral de confianza no tiene un punto óptimo simple: 0.1 maximiza F1 de detección pero genera ambigüedad en QA; 0.3 reduce esa ambigüedad pero resigna F1. No se exploró una zona intermedia (ej. 0.15-0.2) ni otros ejes (otra resolución, otro detector) por tiempo.
- El modo LLM, aun con el modelo más grande probado (1.5B), solo identifica la categoría correcta 2 de cada 3 veces, y la descripción generada alucina en la mitad de los casos pese a instrucción explícita de no hacerlo. Un modelo liviano no es garantía de respuestas confiables sin mecanismos adicionales de verificación que esta sesión no llegó a implementar.
- No se hizo inspección cualitativa caso por caso de falsos positivos/negativos de detección por categoría (sección 5) — el análisis de errores se apoyó en agregados (F1, accuracy) más que en ejemplos puntuales.

**Trabajo futuro, en orden de impacto esperado:**
1. Mitigar la alucinación de `generate_description_llm` — es el riesgo más concreto detectado, y el más barato de intentar mejorar (prompt más restrictivo, post-verificación de sustantivos contra el vocabulario de categorías).
2. Explorar umbrales intermedios (0.15-0.2) en vez de solo 0.1 vs 0.3, para buscar un punto que no sacrifique tanto recall como 0.3 pero sí reduzca algo la ambigüedad de QA.
3. Completar las funciones de métrica reutilizables en `metrics.py` (hoy el cálculo de `qa_accuracy`/`scene_json_consistency` vive disperso en los scripts de cada corrida).
4. Si hay tiempo para una extensión de nivel sobresaliente: usar las máscaras reales (no solo bboxes) para refinar `compute_relations` (`overlaps` vía IoU de máscaras en vez de bboxes sería más preciso en escenas con mucho crowding, donde las bboxes se superponen más de lo que los objetos realmente lo hacen).
