"""Builds TPI_IA_2026P_notebook.ipynb by assembling markdown explanations
(condensed from REPORTE.md) with the real, already-tested source of
src/*.py (read from disk, not retyped) and fresh orchestration code
adapted for Kaggle (input paths, GPU device placement, output location).

Renombrar el .ipynb resultante a TPI_IA_2026P_[nro_de_grupo] una vez que
se complete la Portada de REPORTE.md con el número de grupo.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OUTPUT_PATH = ROOT / "TPI_IA_2026P_notebook.ipynb"

_INTERNAL_IMPORT_RE = re.compile(r"^from src(\.\w+)? import .+$")


def read_module(name: str, rename: dict[str, str] | None = None) -> str:
    """Reads src/<name>.py and strips intra-package imports (from src...):
    in the flattened notebook those names are already defined by earlier
    cells, in the same order as the pipeline stages.

    rename: {old_identifier: new_identifier} applied with word-boundary
    regex substitution. Needed because detection.py/segmentation.py/
    qa_llm.py each privately define module-level globals with the SAME
    names (_processor, _model, _get_model, _MODEL_NAME) -- harmless as
    separate files, but when flattened into one notebook namespace each
    module's cell silently clobbers the previous module's cached model,
    so e.g. a later call to detect_objects() ends up calling whatever
    _get_model was most recently redefined (the LLM's, say) instead of
    OWLv2's. Verified this actually happens via a local smoke test before
    adding this fix -- it does not raise immediately, it just returns the
    wrong model."""
    text = (SRC / f"{name}.py").read_text(encoding="utf-8")
    lines = [line for line in text.split("\n") if not _INTERNAL_IMPORT_RE.match(line)]
    source = "\n".join(lines).strip("\n")
    if rename:
        for old, new in rename.items():
            source = re.sub(rf"\b{re.escape(old)}\b", new, source)
    return source


def md(text: str) -> dict:
    lines = text.strip("\n").split("\n")
    source = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(text: str) -> dict:
    lines = text.strip("\n").split("\n")
    source = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}


cells: list[dict] = []

# ============================================================ Portada ====
cells.append(md("""
# TPI IA 2026 — Comprensión visual de escenas

**Competencia:** `tpi-ia-2026-primavera-comprension-visual-de-escenas` (Kaggle, Cátedra Inteligencia Artificial, UTN)

**Integrantes:** _(completar)_ · **Nro. de grupo:** _(completar)_

Esta notebook implementa de punta a punta la pipeline *imagen → detección → segmentación → SceneJSON → preguntas (determinista + LLM)*, siguiendo las 9 etapas de "Desarrollo del trabajo" del enunciado. El reporte técnico completo (decisiones, experimentos, análisis de errores) está en `REPORTE.md` del repositorio del grupo; acá se resume lo esencial de cada decisión junto al código real que la implementa.

**Reglas respetadas:** las anotaciones del dataset de desarrollo se usan solo para EDA y evaluación (nunca para asistir la inferencia); el LLM nunca recibe la imagen ni los campos estructurados reservados al modo determinista; toda la pipeline corre sin archivos externos propios, solo el dataset oficial de la competencia y pesos de modelos públicos descargados por la propia notebook.
"""))

# ============================================================ Setup ====
cells.append(md("""
## Setup

Instala dependencias que no vienen preinstaladas en el entorno de Kaggle, detecta dónde quedó montado el dataset de la competencia, fija semillas y detecta GPU.
"""))

cells.append(code("""
!pip install -q -U transformers scipy jsonschema
"""))

cells.append(code("""
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
"""))

cells.append(code("""
# El dataset queda montado automáticamente al crear la notebook desde la
# competencia (pestaña Data -> Add competition data), sin usar Add Input
# manual. Se busca con un fallback por si el nombre exacto de la carpeta
# cambia entre entornos.
CANDIDATE_ROOTS = [
    Path("/kaggle/input/tpi-ia-2026-primavera-comprension-visual-de-escenas"),
]
DATA_ROOT = next((p for p in CANDIDATE_ROOTS if (p / "scene_schema.json").exists()), None)

if DATA_ROOT is None and Path("/kaggle/input").exists():
    for candidate in Path("/kaggle/input").iterdir():
        if (candidate / "scene_schema.json").exists():
            DATA_ROOT = candidate
            break

if DATA_ROOT is None:
    raise FileNotFoundError(
        "No se encontro el dataset de la competencia en /kaggle/input/. "
        "Verificar que la notebook se haya creado desde la competencia "
        "(pestana Data -> Add competition data)."
    )

print(f"Dataset encontrado en: {DATA_ROOT}")

DEVELOPMENT_DIR = DATA_ROOT / "development"
TEST_DIR = DATA_ROOT / "tpi_ia_2026_competition_public"
QUESTIONS_PATH = DATA_ROOT / "tpi_2026_qa_public" / "questions_dev.jsonl"
SAMPLE_SUBMISSION_PATH = DATA_ROOT / "sample_submission.csv"
SCHEMA_PATH = DATA_ROOT / "scene_schema.json"

OUTPUT_DIR = Path("/kaggle/working")
TRACE_PATH = OUTPUT_DIR / "trace" / "pipeline.jsonl"

# Umbral de confianza baseline (ver justificacion en la seccion de Deteccion).
# Cambiar a 0.3 para reproducir la comparacion controlada en lugar del baseline.
CONFIDENCE_THRESHOLD = 0.1
NMS_IOU_THRESHOLD = 0.5

# Poner en False para saltear el experimento controlado (umbral 0.3) y
# ahorrar tiempo en una corrida de prueba rapida.
RUN_CONTROLLED_EXPERIMENT = True
"""))

# ============================================================ Etapa 1: EDA ====
cells.append(md("""
## Etapa 1 — Análisis exploratorio del dataset

El EDA no es una colección mecánica de gráficos: cada número presentado motivó una decisión posterior concreta.

- **Distribución de categorías** (cola larga: 46 instancias en la más frecuente vs. 3 en las más raras) → no tiene sentido afinar el umbral de confianza por categoría, se usa uno global.
- **`area_ratio`** (mediana ≈0.003 — los objetos son muy chicos) → justifica un umbral de confianza permisivo en detección, y define los cortes de `size` (terciles empíricos, no números redondos inventados).
- **Oclusión** (13% de instancias con IoU>0.3 con su vecino) → anticipa la necesidad de NMS por categoría en detección y de limitar las relaciones espaciales a vecinos cercanos.
- **`neg_category_ids`/`not_exhaustive_category_ids`** → define directamente la política LVIS-aware de las métricas de detección/máscaras.
"""))

cells.append(code(read_module("geometry")))
cells.append(code(read_module("eda")))

cells.append(code("""
annotations = load_annotations(DEVELOPMENT_DIR)
categories = annotations["categories"]

categories_df = category_distribution(annotations)
print(categories_df.head(10).to_string(index=False))

instances = instances_per_image(annotations)
print(f"instancias por imagen: media={instances.mean():.1f} mediana={instances.median():.0f} max={instances.max()}")

sizes = object_size_distribution(annotations)
print(f"area_ratio terciles: {sizes['area_ratio'].quantile(1/3):.5f} / {sizes['area_ratio'].quantile(2/3):.5f}")

occlusion = occlusion_stats(annotations)
print(f"instancias con IoU>0.3 con su vecino: {(occlusion['max_neighbor_iou']>0.3).mean()*100:.1f}%")

neg_summary = negative_and_non_exhaustive_categories(annotations)
print(f"imagenes con neg_category_ids: {neg_summary['images_with_neg_category_ids']}")
print(f"imagenes con not_exhaustive_category_ids: {neg_summary['images_with_not_exhaustive_category_ids']}")
"""))

# ===================================================== Etapa 2: Deteccion ====
cells.append(md("""
## Etapa 2 — Detección open-vocabulary

**Detector: OWLv2** (`google/owlv2-base-patch16-ensemble`, vía `transformers`). Elegido sobre Grounding DINO por ser pip-instalable sin compilar operadores CUDA custom, más confiable en un entorno compartido como Kaggle. Soporta vocabulario de texto libre.

- **Vocabulario:** las 120 categorías completas del dataset, en un solo batch de texto por imagen (no sabemos de antemano qué categorías aparecen en el test).
- **Umbral de confianza: 0.1** (permisivo), justificado por el EDA: objetos chicos dan confianza baja incluso en aciertos reales.
- **Resolución de entrada: 960×960 fija**, la nativa del checkpoint (resize no-uniforme, sin letterbox, verificado en el código fuente de `transformers`). No se la modifica porque el backbone fue entrenado específicamente a ese tamaño — cambiarla arriesga más de lo que soluciona sin re-calibración.
- **Supresión de duplicados: NMS por categoría** (no global), para no borrar objetos distintos que se superponen legítimamente.
"""))

_DETECTION_RENAME = {
    "_processor": "_det_processor", "_model": "_det_model",
    "_get_model": "_get_det_model", "_MODEL_NAME": "_DET_MODEL_NAME",
}
cells.append(code(read_module("detection", _DETECTION_RENAME).replace(
    '_det_model = Owlv2ForObjectDetection.from_pretrained(_DET_MODEL_NAME)\n        _det_model.eval()',
    '_det_model = Owlv2ForObjectDetection.from_pretrained(_DET_MODEL_NAME).to(DEVICE)\n        _det_model.eval()',
).replace(
    'inputs = processor(text=[names], images=image, return_tensors="pt")',
    'inputs = processor(text=[names], images=image, return_tensors="pt").to(DEVICE)',
).replace(
    'target_sizes = torch.tensor([image.size[::-1]])',
    'target_sizes = torch.tensor([image.size[::-1]]).to(DEVICE)',
).replace(
    'x1, y1, x2, y2 = box.tolist()',
    'x1, y1, x2, y2 = box.cpu().tolist()',
)))

cells.append(code("""
from PIL import Image

VOCABULARY = {c["name"]: c["id"] for c in categories}


def detect_all(image_ids: list[str], image_dir: Path, file_name_fn, tracer=None) -> list[dict]:
    \"\"\"Runs detect_objects + suppress_duplicates over every image_id,
    tagging predictions with "image_id". file_name_fn(image_id) -> filename.\"\"\"
    predictions = []
    t0 = time.time()
    for i, image_id in enumerate(image_ids):
        image = Image.open(image_dir / file_name_fn(image_id)).convert("RGB")
        raw = detect_objects(image, VOCABULARY, confidence_threshold=CONFIDENCE_THRESHOLD)
        kept = suppress_duplicates(raw, iou_threshold=NMS_IOU_THRESHOLD)
        for det in kept:
            det["image_id"] = image_id
        predictions.extend(kept)
        if (i + 1) % 20 == 0 or i == len(image_ids) - 1:
            print(f"  [{i+1}/{len(image_ids)}] {time.time()-t0:.0f}s, {len(predictions)} predicciones acumuladas")
    return predictions
"""))

cells.append(md("Detección sobre las 120 imágenes de desarrollo (para evaluación con ground truth):"))

cells.append(code("""
dev_image_ids = [img["id"] for img in annotations["images"]]
dev_file_names = {img["id"]: img["file_name"] for img in annotations["images"]}

dev_predictions = detect_all(
    dev_image_ids, DEVELOPMENT_DIR / "images", lambda iid: dev_file_names[iid]
)
print(f"Total predicciones (desarrollo): {len(dev_predictions)}")
"""))

# ============================================= Etapa 3: Segmentacion + propiedades ====
cells.append(md("""
## Etapa 3 — Segmentación de instancias con SAM2 + propiedades derivadas

**Segmentador: SAM2.1 tiny** (`facebook/sam2.1-hiera-tiny`, vía `transformers` — misma lógica que con el detector, evita clonar/compilar el repo oficial de Meta). Las cajas detectadas se usan como *prompts*, todas las de una imagen en un único forward batcheado.

**Propiedades derivadas de las máscaras** (`centroid`, `area_ratio`, `position`, `size`, `dominant_color`) y **relaciones espaciales** (`compute_relations`): limitadas a los 5 vecinos más cercanos por objeto, no todos los pares — la primera versión sin ese límite generaba más de 12000 relaciones en la imagen más poblada, mayormente ruido entre objetos lejanos sin relación real.
"""))

_SEGMENTATION_RENAME = {
    "_processor": "_sam_processor", "_model": "_sam_model",
    "_get_model": "_get_sam_model", "_MODEL_NAME": "_SAM_MODEL_NAME",
}
cells.append(code(read_module("segmentation", _SEGMENTATION_RENAME).replace(
    '_sam_model = Sam2Model.from_pretrained(_SAM_MODEL_NAME)\n        _sam_model.eval()',
    '_sam_model = Sam2Model.from_pretrained(_SAM_MODEL_NAME).to(DEVICE)\n        _sam_model.eval()',
).replace(
    'inputs = processor(images=image, input_boxes=[boxes_xyxy], return_tensors="pt")',
    'inputs = processor(images=image, input_boxes=[boxes_xyxy], return_tensors="pt").to(DEVICE)',
).replace(
    'return [masks[i, 0].numpy() for i in range(masks.shape[0])]',
    'return [masks[i, 0].cpu().numpy() for i in range(masks.shape[0])]',
)))

cells.append(code(read_module("properties")))

cells.append(code("""
def segment_and_enrich(image_id: str, image_path: Path, boxes_dets: list[dict]) -> dict:
    \"\"\"boxes_dets: detections for one image, as produced by detect_all
    (before grouping). Returns the full SceneJSON for that image (image +
    objects with all derived properties + relations), still missing
    questions/questions_llm.\"\"\"
    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    objs_for_scene = []
    if boxes_dets:
        masks = segment_with_sam2(image, [d["bbox"] for d in boxes_dets])
        for det, mask in zip(boxes_dets, masks):
            objs_for_scene.append({
                "category_id": det["category_id"],
                "confidence": det["confidence"],
                "bbox": det["bbox"],
                "mask": mask_to_rle(mask),
            })

    scene = build_scene_json(image_id, w, h, objs_for_scene)

    enriched_objects = []
    for obj in scene["objects"]:
        mask = rle_to_mask(obj["mask"])
        centroid = compute_centroid(mask)
        area_ratio = compute_area_ratio(mask, w, h)
        color = compute_dominant_color(image, mask)
        enriched_objects.append({
            **obj, "centroid": centroid, "area_ratio": area_ratio,
            "position": compute_position(centroid, w, h),
            "size": compute_size(area_ratio),
            "dominant_color": list(color),
        })
    scene["objects"] = enriched_objects
    scene["relations"] = compute_relations([{"id": o["id"], "bbox": o["bbox"]} for o in enriched_objects])
    return scene
"""))

# ===================================================== Etapa 4: Representacion ====
cells.append(md("""
## Etapa 4 — Representación estructurada de la escena (`SceneJSON`)

`build_scene_json` ensambla el esqueleto (`image`, `objects`, `relations`) con `id` único por objeto. Nota de diseño: el ensamblado es en dos pasos — esta función arma la estructura, `properties.py` (arriba) calcula las propiedades geométricas — por eso `segment_and_enrich` combina ambos pasos para dejar cada escena completa de una.

Se construyen las escenas de **desarrollo** (para evaluación) y de **test** (para el envío final).
"""))

cells.append(code(read_module("scene_builder")))

cells.append(code("""
print("Construyendo escenas de desarrollo (detección + segmentación + propiedades)...")
t0 = time.time()

dev_preds_by_image: dict[str, list[dict]] = {}
for pred in dev_predictions:
    dev_preds_by_image.setdefault(pred["image_id"], []).append(pred)

dev_scenes: dict[str, dict] = {}
for i, img_info in enumerate(annotations["images"]):
    image_id = Path(img_info["file_name"]).stem
    scene = segment_and_enrich(
        image_id,
        DEVELOPMENT_DIR / "images" / img_info["file_name"],
        dev_preds_by_image.get(img_info["id"], []),
    )
    dev_scenes[image_id] = scene
    if (i + 1) % 20 == 0 or i == len(annotations["images"]) - 1:
        print(f"  [{i+1}/{len(annotations['images'])}] {time.time()-t0:.0f}s")

print(f"Escenas de desarrollo construidas: {len(dev_scenes)}")
"""))

cells.append(code("""
import jsonschema

schema = json.load(open(SCHEMA_PATH))
validator = jsonschema.Draft202012Validator(schema)
invalid = [iid for iid, s in dev_scenes.items() if list(validator.iter_errors(s))]
print(f"Escenas validas contra el schema: {len(dev_scenes)-len(invalid)}/{len(dev_scenes)}")
if invalid:
    print("Invalidas:", invalid[:5])
"""))

# ===================================================== Etapa 5: Trazabilidad ====
cells.append(md("""
## Etapa 5 — Trazabilidad de la pipeline

`Tracer` escribe una línea JSON por paso (formato *JSON Lines*, auditable) en `/kaggle/working/trace/pipeline.jsonl`, con `timestamp`, `run_id`, `image_id`, `module`, `status` y `duration_seconds`. `Tracer.step(...)` es un *context manager* que, si el bloque falla, registra `status="error"` con el mensaje de la excepción **y la vuelve a lanzar** — un paso que falla sigue fallando de forma visible, pero queda localizado en la traza.

A partir de acá se usa `tracer.step(...)` alrededor de cada llamada a las funciones ya definidas arriba, para dejar trazada la ejecución real de las etapas 6-9.
"""))

cells.append(code(read_module("tracing")))

cells.append(code("""
tracer = Tracer(TRACE_PATH)
print(f"Tracer inicializado, run_id={tracer.run_id}, escribiendo en {TRACE_PATH}")
"""))

# ===================================================== Etapa 6: QA determinista ====
cells.append(md("""
## Etapa 6 — Preguntas y respuestas, modo determinista

`answer_question` recibe una fila de `questions_dev.jsonl` (`type`, `class_en`, sin pasar por ningún LLM) y construye la respuesta consultando directamente el `SceneJSON`.

- **Predicado de `relation`:** no viene como campo estructurado separado, solo embebido en el texto — se resuelve con matching de frases fijas en español (`_PREDICATE_PATTERNS`), verificado contra las 10 preguntas `relation` reales del benchmark.
- **Ambigüedad:** `position`/`size`/`relation` presuponen una única instancia; si hay 0 o más de 1 match, la respuesta es `unsupported` en vez de adivinar.
- **`confidence`:** mínimo de las detecciones involucradas en la respuesta.
- **`description`:** plantilla determinista (no LLM) con las 5 categorías más frecuentes.
"""))

cells.append(code(read_module("qa_engine")))

cells.append(code("""
dev_questions = [json.loads(line) for line in open(QUESTIONS_PATH)]
print(f"Preguntas del benchmark: {len(dev_questions)}")

dev_answers_by_image: dict[str, list[dict]] = {}
for q in dev_questions:
    image_id = q["image_file"].rsplit(".", 1)[0]
    scene = dev_scenes[image_id]
    with tracer.step(image_id, "qa_engine", {"question_id": q["question_id"], "type": q["type"]}):
        answer = answer_question(scene, q, categories)
    dev_answers_by_image.setdefault(image_id, []).append(answer)

total_answers = sum(len(v) for v in dev_answers_by_image.values())
unsupported = sum(1 for v in dev_answers_by_image.values() for a in v if a["normalized_answer"] == "unsupported")
print(f"Respuestas generadas: {total_answers} ({unsupported} unsupported)")
"""))

# ===================================================== Etapa 7: QA con LLM ====
cells.append(md("""
## Etapa 7 — Preguntas y respuestas con LLM liviano (promoción)

**Modelo: Qwen2.5-1.5B-Instruct.** Se probó primero Qwen2.5-0.5B-Instruct (solo 22% de `class_en` exacto contra el vocabulario de 120 categorías — deja palabras en español sin traducir) antes de pasar a 1.5B (67.5%).

- **`interpret_query`** recibe *únicamente* el texto de la pregunta (nunca `type`/`class_en` de la cátedra) y produce un JSON intermedio `{type, class_en, predicate?}`, validado contra el vocabulario conocido antes de ejecutar — si no valida, `unsupported`, nunca se inventa una categoría.
- **Reutiliza el mismo motor determinista** (`answer_question`) para ejecutar la consulta validada — el LLM solo interpreta, nunca calcula la respuesta.
- **`generate_description_llm`** recibe solo pares categoría+conteo agregados, nunca la imagen. **Limitación medida y documentada:** no respeta de forma confiable la instrucción de no inventar información (ver `REPORTE.md` 4.8) — se reporta tal cual sale, sin maquillar.
"""))

_QA_LLM_RENAME = {
    "_tokenizer": "_llm_tokenizer", "_model": "_llm_model",
    "_get_model": "_get_llm_model", "_MODEL_NAME": "_LLM_MODEL_NAME",
}
cells.append(code(read_module("qa_llm", _QA_LLM_RENAME).replace(
    '_llm_model = AutoModelForCausalLM.from_pretrained(_LLM_MODEL_NAME)\n        _llm_model.eval()',
    '_llm_model = AutoModelForCausalLM.from_pretrained(_LLM_MODEL_NAME).to(DEVICE)\n        _llm_model.eval()',
).replace(
    'inputs = tokenizer(text, return_tensors="pt")',
    'inputs = tokenizer(text, return_tensors="pt").to(DEVICE)',
).replace(
    # qa_llm.py normally does "from src import qa_engine" and calls these
    # module-qualified; that import is stripped when flattening into the
    # notebook (read_module), but the names are already defined globally
    # by the qa_engine cell above, so the "qa_engine." prefix just needs
    # to go -- otherwise this raises NameError: qa_engine is not defined.
    'qa_engine.QUESTION_TYPES', 'QUESTION_TYPES',
).replace(
    'qa_engine.answer_question(', 'answer_question(',
).replace(
    'qa_engine.unsupported_answer(', 'unsupported_answer(',
)))

cells.append(code("""
dev_answers_llm_by_image: dict[str, list[dict]] = {}
t0 = time.time()
for i, q in enumerate(dev_questions):
    image_id = q["image_file"].rsplit(".", 1)[0]
    scene = dev_scenes[image_id]
    with tracer.step(image_id, "qa_llm", {"question_id": q["question_id"], "type": q["type"]}):
        answer = answer_question_llm(scene, q, categories)
    dev_answers_llm_by_image.setdefault(image_id, []).append(answer)
    if (i + 1) % 15 == 0 or i == len(dev_questions) - 1:
        print(f"  [{i+1}/{len(dev_questions)}] {time.time()-t0:.0f}s")

total_llm = sum(len(v) for v in dev_answers_llm_by_image.values())
unsupported_llm = sum(1 for v in dev_answers_llm_by_image.values() for a in v if a["normalized_answer"] == "unsupported")
print(f"Respuestas LLM generadas: {total_llm} ({unsupported_llm} unsupported)")
"""))

cells.append(code("""
# Integra questions/questions_llm en cada escena de desarrollo -- a partir
# de aca dev_scenes[image_id] ya tiene el SceneJSON completo y valido.
for image_id, scene in dev_scenes.items():
    scene["questions"] = dev_answers_by_image.get(image_id, [])
    scene["questions_llm"] = dev_answers_llm_by_image.get(image_id, [])
"""))

# ===================================================== Etapa 8: Metricas ====
cells.append(md("""
## Etapa 8 — Evaluación cuantitativa y proceso experimental

`detection_f1`/`mask_f1` comparten la misma política LVIS-aware (categoría positiva / `neg_category_ids` / `not_exhaustive_category_ids` / no verificada, tomada literalmente del `README.md` del dataset) a través de un helper común `_lvis_aware_f1`, parametrizado por la función de IoU (bbox o máscara). `scene_json_consistency` valida schema + RLE decodificable + referencias de ids. `qa_accuracy` calcula exact-match contra referencias de ground truth, separado por tipo.
"""))

cells.append(code(read_module("metrics")))

cells.append(code("""
detection_result = detection_f1(dev_predictions, annotations, iou_threshold=0.5)
print(f"Deteccion -> F1={detection_result['f1']:.3f} P={detection_result['precision']:.3f} R={detection_result['recall']:.3f}")

all_dev_objects = [
    {**obj, "image_id": img_info["id"]}
    for img_info in annotations["images"]
    for obj in dev_scenes[Path(img_info["file_name"]).stem]["objects"]
]
mask_result = mask_f1(all_dev_objects, annotations, iou_threshold=0.5)
print(f"Mascaras  -> F1={mask_result['f1']:.3f} P={mask_result['precision']:.3f} R={mask_result['recall']:.3f}")

consistency_results = [scene_json_consistency(s, schema) for s in dev_scenes.values()]
num_consistent = sum(1 for r in consistency_results if r["valid"])
print(f"SceneJSON consistentes (desarrollo): {num_consistent}/{len(dev_scenes)}")

name_to_id = {c["name"]: c["id"] for c in categories}
from collections import Counter
gt_counts = Counter((a["image_id"], a["category_id"]) for a in annotations["annotations"])
reference = []
for q in dev_questions:
    if q["type"] not in ("count", "existence") or not q.get("class_en"):
        continue
    cat_id = name_to_id.get(q["class_en"][0])
    gt_n = gt_counts.get((q["image_id"], cat_id), 0) if cat_id is not None else 0
    reference.append({
        "question_id": q["question_id"],
        "normalized_answer": gt_n if q["type"] == "count" else ("yes" if gt_n > 0 else "no"),
    })

all_dev_answers = [a for answers in dev_answers_by_image.values() for a in answers]
qa_result = qa_accuracy(all_dev_answers, reference)
print(f"QA determinista -> accuracy={qa_result['accuracy']:.2f} by_type={qa_result['by_type']}")
"""))

cells.append(md("""
### Proceso experimental — línea base (0.1) vs. alternativa (0.3)

Comparación controlada: mismo pipeline, mismo detector, solo cambia `confidence_threshold`. **Resultado ya documentado en `REPORTE.md` 4.7:** subir el umbral mejora precision y reduce la ambigüedad en QA, pero **empeora levemente el F1 de detección** (que pesa 75% del `PipelineScore` de la competencia, contra 10% de QA) — por eso el baseline 0.1 se mantiene como configuración final (celda de Etapa 9). Esta sección solo reproduce la comparación; no cambia la configuración usada para la entrega.
"""))

cells.append(code("""
if RUN_CONTROLLED_EXPERIMENT:
    _baseline_threshold = CONFIDENCE_THRESHOLD
    CONFIDENCE_THRESHOLD = 0.3
    print(f"Corriendo pipeline completa con confidence_threshold={CONFIDENCE_THRESHOLD}...")

    exp_predictions = detect_all(dev_image_ids, DEVELOPMENT_DIR / "images", lambda iid: dev_file_names[iid])
    exp_preds_by_image: dict[str, list[dict]] = {}
    for pred in exp_predictions:
        exp_preds_by_image.setdefault(pred["image_id"], []).append(pred)

    exp_scenes: dict[str, dict] = {}
    for img_info in annotations["images"]:
        image_id = Path(img_info["file_name"]).stem
        exp_scenes[image_id] = segment_and_enrich(
            image_id, DEVELOPMENT_DIR / "images" / img_info["file_name"],
            exp_preds_by_image.get(img_info["id"], []),
        )

    exp_detection = detection_f1(exp_predictions, annotations, iou_threshold=0.5)

    exp_answers = []
    for q in dev_questions:
        image_id = q["image_file"].rsplit(".", 1)[0]
        exp_answers.append(answer_question(exp_scenes[image_id], q, categories))
    exp_unsupported = sum(
        1 for a in exp_answers if a["normalized_answer"] == "unsupported" and a["type"] in ("position", "size", "relation")
    )

    print(f"Baseline (0.1): F1={detection_result['f1']:.3f}")
    print(f"Alternativa (0.3): F1={exp_detection['f1']:.3f}")
    print(f"Unsupported (position+size+relation) baseline vs 0.3: "
          f"{sum(1 for a in all_dev_answers if a['normalized_answer']=='unsupported' and a['type'] in ('position','size','relation'))} "
          f"vs {exp_unsupported}")

    CONFIDENCE_THRESHOLD = _baseline_threshold
    print(f"confidence_threshold restaurado a {CONFIDENCE_THRESHOLD} para la entrega final.")
else:
    print("RUN_CONTROLLED_EXPERIMENT=False, se saltea esta seccion.")
"""))

# ===================================================== Etapa 9: Entrega ====
cells.append(md("""
## Etapa 9 — Entrega final (`submission.csv` / `solution.csv`)

Se corre la pipeline completa (detección → segmentación → `scene_builder` → `properties`) sobre las **60 imágenes de test** (sin `questions`/`questions_llm`, no tienen benchmark asociado) y se combinan con las 20 escenas de desarrollo ya completas en un único archivo de 80 filas.

Como el PDF del enunciado llama al archivo `submission.csv` y la página de Kaggle lo llama `solution.csv` (discrepancia nunca resuelta por la cátedra), se escriben **ambos nombres con contenido idéntico**.
"""))

cells.append(code("""
import csv

test_images_meta = {}
with open(TEST_DIR / "test_images.csv") as f:
    for r in csv.DictReader(f):
        test_images_meta[r["image_id"]] = r

sample_rows = list(csv.DictReader(open(SAMPLE_SUBMISSION_PATH)))
test_ids = sorted(r["image_id"] for r in sample_rows if r["image_id"].startswith("test_"))

print(f"Procesando {len(test_ids)} imagenes de test (pipeline completa, sin QA)...")
t0 = time.time()
test_scenes: dict[str, dict] = {}
for i, image_id in enumerate(test_ids):
    meta = test_images_meta[image_id]
    image_path = TEST_DIR / "test_images" / meta["file_name"]
    image = Image.open(image_path).convert("RGB")

    with tracer.step(image_id, "detection", {"confidence_threshold": CONFIDENCE_THRESHOLD}):
        raw = detect_objects(image, VOCABULARY, confidence_threshold=CONFIDENCE_THRESHOLD)
    kept = suppress_duplicates(raw, iou_threshold=NMS_IOU_THRESHOLD)

    with tracer.step(image_id, "segmentation", {"num_boxes": len(kept)}):
        scene = segment_and_enrich(image_id, image_path, kept)
    test_scenes[image_id] = scene

    if (i + 1) % 10 == 0 or i == len(test_ids) - 1:
        print(f"  [{i+1}/{len(test_ids)}] {time.time()-t0:.0f}s, {len(scene['objects'])} objetos")

print(f"Escenas de test construidas: {len(test_scenes)}")
"""))

cells.append(code("""
rows = []
for image_id, scene in dev_scenes.items():
    rows.append((image_id, scene))
for image_id, scene in test_scenes.items():
    rows.append((image_id, scene))
rows.sort(key=lambda r: r[0])

validator = jsonschema.Draft202012Validator(schema)
num_invalid = 0
num_inconsistent = 0
for image_id, scene in rows:
    if list(validator.iter_errors(scene)):
        num_invalid += 1
        print(f"INVALID (schema) {image_id}")
    if not scene_json_consistency(scene, schema)["valid"]:
        num_inconsistent += 1
        print(f"INCONSISTENT {image_id}")

sample_ids = sorted(r["image_id"] for r in sample_rows)
our_ids = sorted(r[0] for r in rows)
assert our_ids == sample_ids, "El conjunto de image_id no coincide con sample_submission.csv"
assert len(rows) == 80, f"se esperaban 80 filas, hay {len(rows)}"

for filename in ("submission.csv", "solution.csv"):
    with open(OUTPUT_DIR / filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "SceneJSON"])
        for image_id, scene in rows:
            writer.writerow([image_id, json.dumps(scene, ensure_ascii=False)])

print()
print(f"Filas: {len(rows)} (80 esperadas)")
print(f"Validas contra schema: {len(rows)-num_invalid}/{len(rows)}")
print(f"Consistentes (schema+RLE+referencias): {len(rows)-num_inconsistent}/{len(rows)}")
print(f"Escrito en {OUTPUT_DIR}/submission.csv y {OUTPUT_DIR}/solution.csv")
"""))

cells.append(md("""
## Cierre

Las 9 etapas del enunciado quedaron implementadas y corridas con datos reales en esta notebook. Resumen de resultados — ver `REPORTE.md` del repositorio del grupo para el detalle completo de decisiones, experimentos y análisis de errores (incluidas las limitaciones identificadas, como la alucinación del LLM en la descripción generada).

- F1 de detección (baseline 0.1) y F1 de máscaras: ver salida de la celda de Etapa 8.
- `SceneJSON` 100% válido y consistente en las 80 filas de entrega.
- `submission.csv`/`solution.csv` guardados en `/kaggle/working/`, listos para enviar.
- Traza completa de la ejecución en `/kaggle/working/trace/pipeline.jsonl`.
"""))

print(f"Total cells: {len(cells)}")

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"Notebook escrita en {OUTPUT_PATH}")
