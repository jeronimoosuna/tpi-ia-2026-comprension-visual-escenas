import json
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import detection, metrics, tracing

ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_DIR = ROOT / "data" / "development"
OUTPUT_DIR = ROOT / "reports" / "detection"
TRACE_PATH = ROOT / "reports" / "trace" / "pipeline.jsonl"

CONFIDENCE_THRESHOLD = 0.1
NMS_IOU_THRESHOLD = 0.5


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tracer = tracing.Tracer(TRACE_PATH)
    annotations = json.load(open(DEVELOPMENT_DIR / "annotations.json"))
    vocabulary = {c["name"]: c["id"] for c in annotations["categories"]}

    all_predictions = []
    per_image_timing = []
    t_start = time.time()

    for i, img_info in enumerate(annotations["images"]):
        image_id = Path(img_info["file_name"]).stem
        image_path = DEVELOPMENT_DIR / "images" / img_info["file_name"]
        image = Image.open(image_path).convert("RGB")

        t0 = time.time()
        with tracer.step(image_id, "detection", {
            "confidence_threshold": CONFIDENCE_THRESHOLD, "vocabulary_size": len(vocabulary),
        }):
            raw = detection.detect_objects(image, vocabulary, confidence_threshold=CONFIDENCE_THRESHOLD)
        with tracer.step(image_id, "nms", {"iou_threshold": NMS_IOU_THRESHOLD, "num_raw": len(raw)}):
            kept = detection.suppress_duplicates(raw, iou_threshold=NMS_IOU_THRESHOLD)
        elapsed = time.time() - t0

        for det in kept:
            det["image_id"] = img_info["id"]
        all_predictions.extend(kept)
        per_image_timing.append(elapsed)

        print(f"[{i + 1}/{len(annotations['images'])}] {img_info['file_name']}: "
              f"{len(raw)} raw -> {len(kept)} after NMS ({elapsed:.1f}s)", flush=True)

    total_elapsed = time.time() - t_start

    with open(OUTPUT_DIR / "predictions.json", "w", encoding="utf-8") as f:
        json.dump(all_predictions, f)

    result = metrics.detection_f1(all_predictions, annotations, iou_threshold=0.5)
    result["confidence_threshold"] = CONFIDENCE_THRESHOLD
    result["nms_iou_threshold"] = NMS_IOU_THRESHOLD
    result["num_images"] = len(annotations["images"])
    result["num_predictions"] = len(all_predictions)
    result["total_time_seconds"] = total_elapsed
    result["avg_time_per_image_seconds"] = sum(per_image_timing) / len(per_image_timing)

    with open(OUTPUT_DIR / "detection_f1_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print()
    print(f"Total: {total_elapsed:.1f}s ({result['avg_time_per_image_seconds']:.1f}s/img avg)")
    print(f"Predictions: {len(all_predictions)}")
    print(f"TP={result['tp']} FP={result['fp']} FN={result['fn']} ignored={result['ignored_predictions']}")
    print(f"Precision={result['precision']:.3f} Recall={result['recall']:.3f} F1={result['f1']:.3f}")


if __name__ == "__main__":
    main()
