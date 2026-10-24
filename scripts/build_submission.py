import csv
import json
import sys
from pathlib import Path

import jsonschema
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import detection, properties, scene_builder, segmentation, tracing

ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_DIR = ROOT / "data" / "development"
TEST_DIR = ROOT / "data" / "tpi_ia_2026_competition_public"
QUESTIONS_PATH = ROOT / "data" / "tpi_2026_qa_public" / "questions_dev.jsonl"
SAMPLE_SUBMISSION_PATH = ROOT / "data" / "sample_submission.csv"
SCHEMA_PATH = ROOT / "data" / "scene_schema.json"
DEV_SCENES_PATH = ROOT / "reports" / "scene_json" / "scenes_enriched.json"
DEV_QA_PATH = ROOT / "reports" / "qa" / "answers_dev.json"
DEV_QA_LLM_PATH = ROOT / "reports" / "qa" / "answers_llm_dev.json"
TRACE_PATH = ROOT / "reports" / "trace" / "pipeline.jsonl"

CONFIDENCE_THRESHOLD = 0.1
NMS_IOU_THRESHOLD = 0.5


def build_test_scene(tracer, image_id: str, file_name: str, width: int, height: int, vocabulary: dict) -> dict:
    image = Image.open(TEST_DIR / "test_images" / file_name).convert("RGB")

    with tracer.step(image_id, "detection", {"confidence_threshold": CONFIDENCE_THRESHOLD}):
        raw = detection.detect_objects(image, vocabulary, confidence_threshold=CONFIDENCE_THRESHOLD)
    kept = detection.suppress_duplicates(raw, iou_threshold=NMS_IOU_THRESHOLD)

    objs_for_scene = []
    if kept:
        with tracer.step(image_id, "segmentation", {"num_boxes": len(kept)}):
            masks = segmentation.segment_with_sam2(image, [d["bbox"] for d in kept])
        for det, mask in zip(kept, masks):
            objs_for_scene.append({
                "category_id": det["category_id"],
                "confidence": det["confidence"],
                "bbox": det["bbox"],
                "mask": segmentation.mask_to_rle(mask),
            })

    with tracer.step(image_id, "scene_builder", {"num_input_objects": len(objs_for_scene)}):
        scene = scene_builder.build_scene_json(image_id, width, height, objs_for_scene)

    with tracer.step(image_id, "properties", {"num_objects": len(scene["objects"])}):
        w, h = scene["image"]["width"], scene["image"]["height"]
        enriched_objects = []
        for obj in scene["objects"]:
            mask = segmentation.rle_to_mask(obj["mask"])
            centroid = properties.compute_centroid(mask)
            area_ratio = properties.compute_area_ratio(mask, w, h)
            color = properties.compute_dominant_color(image, mask)
            enriched_objects.append({
                **obj, "centroid": centroid, "area_ratio": area_ratio,
                "position": properties.compute_position(centroid, w, h),
                "size": properties.compute_size(area_ratio),
                "dominant_color": list(color),
            })
        scene["objects"] = enriched_objects
        scene["relations"] = properties.compute_relations(
            [{"id": o["id"], "bbox": o["bbox"]} for o in enriched_objects]
        )

    return scene


def main() -> None:
    tracer = tracing.Tracer(TRACE_PATH)
    dev_annotations = json.load(open(DEVELOPMENT_DIR / "annotations.json"))
    categories = dev_annotations["categories"]
    vocabulary = {c["name"]: c["id"] for c in categories}
    schema = json.load(open(SCHEMA_PATH))

    # --- 20 dev benchmark rows: reuse already-computed scenes + QA answers ---
    dev_scenes = json.load(open(DEV_SCENES_PATH))
    dev_questions = [json.loads(line) for line in open(QUESTIONS_PATH)]
    dev_answers = json.load(open(DEV_QA_PATH))
    dev_answers_llm = json.load(open(DEV_QA_LLM_PATH))

    questions_by_image: dict[str, list] = {}
    questions_llm_by_image: dict[str, list] = {}
    for q, a, a_llm in zip(dev_questions, dev_answers, dev_answers_llm):
        image_id = q["image_file"].rsplit(".", 1)[0]
        questions_by_image.setdefault(image_id, []).append(a)
        questions_llm_by_image.setdefault(image_id, []).append(a_llm)

    rows = []
    for image_id in sorted(questions_by_image.keys()):
        scene = dict(dev_scenes[image_id])
        scene["questions"] = questions_by_image[image_id]
        scene["questions_llm"] = questions_llm_by_image[image_id]
        rows.append((image_id, scene))

    # --- 60 test rows: run the pipeline fresh (never touched before this script) ---
    test_images_meta = {}
    with open(TEST_DIR / "test_images.csv") as f:
        for r in csv.DictReader(f):
            test_images_meta[r["image_id"]] = r

    sample_rows = list(csv.DictReader(open(SAMPLE_SUBMISSION_PATH)))
    test_ids = sorted(r["image_id"] for r in sample_rows if r["image_id"].startswith("test_"))

    for i, image_id in enumerate(test_ids):
        meta = test_images_meta[image_id]
        scene = build_test_scene(
            tracer, image_id, meta["file_name"], int(meta["width"]), int(meta["height"]), vocabulary,
        )
        rows.append((image_id, scene))
        print(f"[{i + 1}/{len(test_ids)}] {image_id}: {len(scene['objects'])} objects", flush=True)

    rows.sort(key=lambda r: r[0])

    # --- validate against the real schema and against sample_submission.csv ---
    validator = jsonschema.Draft202012Validator(schema)
    num_invalid = 0
    for image_id, scene in rows:
        errors = list(validator.iter_errors(scene))
        if errors:
            num_invalid += 1
            print(f"INVALID {image_id}: {errors[0].message}")

    sample_ids = sorted(r["image_id"] for r in sample_rows)
    our_ids = sorted(r[0] for r in rows)
    assert our_ids == sample_ids, "image_id set doesn't match sample_submission.csv"
    assert len(rows) == 80, f"expected 80 rows, got {len(rows)}"

    # Naming is ambiguous between the PDF (submission.csv) and the Kaggle
    # page (solution.csv) -- write both, identical content, so nothing
    # needs renaming later regardless of which one Kaggle actually expects.
    for filename in ("submission.csv", "solution.csv"):
        with open(ROOT / filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image_id", "SceneJSON"])
            for image_id, scene in rows:
                writer.writerow([image_id, json.dumps(scene, ensure_ascii=False)])

    print()
    print(f"Rows: {len(rows)} (80 expected)")
    print(f"Schema valid: {len(rows) - num_invalid}/{len(rows)}")
    print("Wrote submission.csv and solution.csv")


if __name__ == "__main__":
    main()
