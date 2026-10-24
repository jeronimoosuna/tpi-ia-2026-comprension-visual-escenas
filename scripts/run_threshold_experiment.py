import json
import sys
import time
from collections import Counter
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import detection, metrics, properties, qa_engine, scene_builder, segmentation, tracing

ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_DIR = ROOT / "data" / "development"
QUESTIONS_PATH = ROOT / "data" / "tpi_2026_qa_public" / "questions_dev.jsonl"
SCHEMA_PATH = ROOT / "data" / "scene_schema.json"
OUTPUT_DIR = ROOT / "reports" / "experiments" / "threshold_0.3"
TRACE_PATH = ROOT / "reports" / "trace" / "pipeline.jsonl"

CONFIDENCE_THRESHOLD = 0.3
NMS_IOU_THRESHOLD = 0.5


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tracer = tracing.Tracer(TRACE_PATH)
    annotations = json.load(open(DEVELOPMENT_DIR / "annotations.json"))
    categories = annotations["categories"]
    vocabulary = {c["name"]: c["id"] for c in categories}
    schema = json.load(open(SCHEMA_PATH))

    all_predictions = []
    all_objects = []
    scenes = {}

    t_start = time.time()
    for i, img_info in enumerate(annotations["images"]):
        image_id = Path(img_info["file_name"]).stem
        image = Image.open(DEVELOPMENT_DIR / "images" / img_info["file_name"]).convert("RGB")

        with tracer.step(image_id, "detection", {
            "confidence_threshold": CONFIDENCE_THRESHOLD, "experiment": "threshold_0.3",
        }):
            raw = detection.detect_objects(image, vocabulary, confidence_threshold=CONFIDENCE_THRESHOLD)
        kept = detection.suppress_duplicates(raw, iou_threshold=NMS_IOU_THRESHOLD)
        for det in kept:
            det["image_id"] = img_info["id"]
        all_predictions.extend(kept)

        boxes = [p["bbox"] for p in kept]
        objs_for_scene = []
        if boxes:
            with tracer.step(image_id, "segmentation", {"num_boxes": len(boxes), "experiment": "threshold_0.3"}):
                masks = segmentation.segment_with_sam2(image, boxes)
            for det, mask in zip(kept, masks):
                obj = {
                    "category_id": det["category_id"],
                    "confidence": det["confidence"],
                    "bbox": det["bbox"],
                    "mask": segmentation.mask_to_rle(mask),
                }
                all_objects.append({**obj, "image_id": img_info["id"]})
                objs_for_scene.append(obj)

        scene = scene_builder.build_scene_json(image_id, img_info["width"], img_info["height"], objs_for_scene)
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
        scenes[image_id] = scene

        print(f"[{i + 1}/{len(annotations['images'])}] {img_info['file_name']}: "
              f"{len(kept)} objects (threshold={CONFIDENCE_THRESHOLD})", flush=True)

    total_elapsed = time.time() - t_start

    with open(OUTPUT_DIR / "predictions.json", "w", encoding="utf-8") as f:
        json.dump(all_predictions, f)
    with open(OUTPUT_DIR / "scenes_enriched.json", "w", encoding="utf-8") as f:
        json.dump(scenes, f)

    det_result = metrics.detection_f1(all_predictions, annotations, iou_threshold=0.5)

    validator = __import__("jsonschema").Draft202012Validator(schema)
    num_invalid = sum(1 for s in scenes.values() if list(validator.iter_errors(s)))

    questions = [json.loads(line) for line in open(QUESTIONS_PATH)]
    gt_counts = Counter((a["image_id"], a["category_id"]) for a in annotations["annotations"])
    name_to_id = {c["name"]: c["id"] for c in categories}

    qa_answers = []
    count_correct = count_total = 0
    exist_correct = exist_total = 0
    for q in questions:
        image_id = q["image_file"].rsplit(".", 1)[0]
        scene = scenes[image_id]
        answer = qa_engine.answer_question(scene, q, categories)
        qa_answers.append(answer)
        if q["type"] in ("count", "existence") and q.get("class_en"):
            cat_id = name_to_id.get(q["class_en"][0])
            gt_n = gt_counts.get((q["image_id"], cat_id), 0) if cat_id is not None else 0
            if q["type"] == "count":
                count_total += 1
                count_correct += int(answer["normalized_answer"] == gt_n)
            else:
                exist_total += 1
                gt_yn = "yes" if gt_n > 0 else "no"
                exist_correct += int(answer["normalized_answer"] == gt_yn)

    with open(OUTPUT_DIR / "qa_answers.json", "w", encoding="utf-8") as f:
        json.dump(qa_answers, f, ensure_ascii=False, indent=2)

    unsupported_by_type = Counter(a["type"] for a in qa_answers if a["normalized_answer"] == "unsupported")

    summary = {
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "nms_iou_threshold": NMS_IOU_THRESHOLD,
        "total_time_seconds": total_elapsed,
        "num_predictions": len(all_predictions),
        "detection": det_result,
        "schema_valid_scenes": len(scenes) - num_invalid,
        "schema_total_scenes": len(scenes),
        "qa_count_exact_match": count_correct,
        "qa_count_total": count_total,
        "qa_existence_correct": exist_correct,
        "qa_existence_total": exist_total,
        "qa_unsupported_by_type": dict(unsupported_by_type),
    }
    with open(OUTPUT_DIR / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"Total time: {total_elapsed:.1f}s")
    print(f"Predictions: {len(all_predictions)}")
    print(f"Detection F1={det_result['f1']:.3f} P={det_result['precision']:.3f} R={det_result['recall']:.3f}")
    print(f"Schema valid: {len(scenes) - num_invalid}/{len(scenes)}")
    print(f"QA count exact-match: {count_correct}/{count_total}")
    print(f"QA existence accuracy: {exist_correct}/{exist_total}")
    print(f"QA unsupported by type: {dict(unsupported_by_type)}")


if __name__ == "__main__":
    main()
