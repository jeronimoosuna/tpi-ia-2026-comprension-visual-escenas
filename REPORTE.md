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

## 5. Discusión y análisis de errores

**Precision (0.512) bastante más baja que recall (0.705):** con un umbral de confianza permisivo (0.1) el detector prioriza no perderse objetos a costa de más falsos positivos — esperable dado que el EDA mostró que la mayoría de los objetos son chicos (mediana `area_ratio`≈0.003) y por lo tanto difíciles de detectar con confianza alta. Queda pendiente la comparación controlada (sección "Proceso experimental" del plan) con un umbral más conservador (ej. 0.3) para ver el trade-off precision/recall y si mejora el F1 compuesto.

_(pendiente: inspección cualitativa de casos fallidos — FP sistemáticos por categoría, FN concentrados en objetos muy chicos o muy ocluidos)_

## 6. Conclusiones y trabajo futuro

_(pendiente)_
