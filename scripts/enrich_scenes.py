import json
import sys
from pathlib import Path

import jsonschema
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import properties, segmentation, tracing

ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_DIR = ROOT / "data" / "development"
SCENES_PATH = ROOT / "reports" / "scene_json" / "scenes.json"
SCHEMA_PATH = ROOT / "data" / "scene_schema.json"
OUTPUT_DIR = ROOT / "reports" / "scene_json"
TRACE_PATH = ROOT / "reports" / "trace" / "pipeline.jsonl"


def main() -> None:
    tracer = tracing.Tracer(TRACE_PATH)
    annotations = json.load(open(DEVELOPMENT_DIR / "annotations.json"))
    scenes = json.load(open(SCENES_PATH))
    schema = json.load(open(SCHEMA_PATH))

    file_name_by_image_id = {Path(img["file_name"]).stem: img["file_name"] for img in annotations["images"]}

    total_objects = 0
    total_relations = 0
    for image_id, scene in scenes.items():
        image = Image.open(DEVELOPMENT_DIR / "images" / file_name_by_image_id[image_id]).convert("RGB")
        w, h = scene["image"]["width"], scene["image"]["height"]

        with tracer.step(image_id, "properties", {"num_objects": len(scene["objects"])}):
            enriched_objects = []
            for obj in scene["objects"]:
                mask = segmentation.rle_to_mask(obj["mask"])
                centroid = properties.compute_centroid(mask)
                area_ratio = properties.compute_area_ratio(mask, w, h)
                color = properties.compute_dominant_color(image, mask)
                enriched_objects.append({
                    **obj,
                    "centroid": centroid,
                    "area_ratio": area_ratio,
                    "position": properties.compute_position(centroid, w, h),
                    "size": properties.compute_size(area_ratio),
                    "dominant_color": list(color),
                })

            scene["objects"] = enriched_objects
            scene["relations"] = properties.compute_relations(
                [{"id": o["id"], "bbox": o["bbox"]} for o in enriched_objects]
            )
        total_objects += len(enriched_objects)
        total_relations += len(scene["relations"])

    with open(OUTPUT_DIR / "scenes_enriched.json", "w", encoding="utf-8") as f:
        json.dump(scenes, f)

    print(f"Enriched scenes: {len(scenes)}")
    print(f"Total objects: {total_objects}")
    print(f"Total relations: {total_relations}")

    validator = jsonschema.Draft202012Validator(schema)
    num_errors = 0
    for image_id, scene in scenes.items():
        with tracer.step(image_id, "schema_validation", {}):
            errors = list(validator.iter_errors(scene))
            if errors:
                num_errors += 1
                print(f"INVALID {image_id}: {errors[0].message}")

    print(f"Schema validation: {len(scenes) - num_errors}/{len(scenes)} scenes valid against {SCHEMA_PATH.name}")


if __name__ == "__main__":
    main()
