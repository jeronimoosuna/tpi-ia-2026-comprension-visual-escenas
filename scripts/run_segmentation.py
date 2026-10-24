import json
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import segmentation

ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_DIR = ROOT / "data" / "development"
DETECTION_PREDICTIONS = ROOT / "reports" / "detection" / "predictions.json"
OUTPUT_DIR = ROOT / "reports" / "segmentation"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    annotations = json.load(open(DEVELOPMENT_DIR / "annotations.json"))
    predictions = json.load(open(DETECTION_PREDICTIONS))

    preds_by_image: dict[int, list[dict]] = {}
    for pred in predictions:
        preds_by_image.setdefault(pred["image_id"], []).append(pred)

    all_objects = []
    per_image_timing = []
    t_start = time.time()

    for i, img_info in enumerate(annotations["images"]):
        image_id = img_info["id"]
        img_preds = preds_by_image.get(image_id, [])
        if not img_preds:
            per_image_timing.append(0.0)
            print(f"[{i + 1}/{len(annotations['images'])}] {img_info['file_name']}: "
                  f"0 boxes, skipped", flush=True)
            continue

        image_path = DEVELOPMENT_DIR / "images" / img_info["file_name"]
        image = Image.open(image_path).convert("RGB")
        boxes = [p["bbox"] for p in img_preds]

        t0 = time.time()
        masks = segmentation.segment_with_sam2(image, boxes)
        elapsed = time.time() - t0

        for pred, mask in zip(img_preds, masks):
            all_objects.append({
                "image_id": image_id,
                "category_id": pred["category_id"],
                "confidence": pred["confidence"],
                "bbox": pred["bbox"],
                "mask": segmentation.mask_to_rle(mask),
            })
        per_image_timing.append(elapsed)

        print(f"[{i + 1}/{len(annotations['images'])}] {img_info['file_name']}: "
              f"{len(boxes)} boxes segmented ({elapsed:.1f}s)", flush=True)

    total_elapsed = time.time() - t_start

    with open(OUTPUT_DIR / "objects_with_masks.json", "w", encoding="utf-8") as f:
        json.dump(all_objects, f)

    nonzero_timing = [t for t in per_image_timing if t > 0]
    summary = {
        "num_images": len(annotations["images"]),
        "num_objects": len(all_objects),
        "total_time_seconds": total_elapsed,
        "avg_time_per_image_with_boxes_seconds": sum(nonzero_timing) / len(nonzero_timing) if nonzero_timing else 0,
    }
    with open(OUTPUT_DIR / "segmentation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"Total: {total_elapsed:.1f}s")
    print(f"Objects segmented: {len(all_objects)}")
    print(f"Avg time per image (with boxes): {summary['avg_time_per_image_with_boxes_seconds']:.1f}s")


if __name__ == "__main__":
    main()
