import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import scene_builder, tracing

ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_DIR = ROOT / "data" / "development"
SEGMENTATION_OBJECTS = ROOT / "reports" / "segmentation" / "objects_with_masks.json"
OUTPUT_DIR = ROOT / "reports" / "scene_json"
TRACE_PATH = ROOT / "reports" / "trace" / "pipeline.jsonl"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tracer = tracing.Tracer(TRACE_PATH)
    annotations = json.load(open(DEVELOPMENT_DIR / "annotations.json"))
    objects_with_masks = json.load(open(SEGMENTATION_OBJECTS))

    objects_by_image: dict[int, list[dict]] = {}
    for obj in objects_with_masks:
        objects_by_image.setdefault(obj["image_id"], []).append(obj)

    scenes = {}
    for img_info in annotations["images"]:
        image_id = Path(img_info["file_name"]).stem
        objs = objects_by_image.get(img_info["id"], [])
        with tracer.step(image_id, "scene_builder", {"num_input_objects": len(objs)}):
            scenes[image_id] = scene_builder.build_scene_json(
                image_id, img_info["width"], img_info["height"], objs
            )

    with open(OUTPUT_DIR / "scenes.json", "w", encoding="utf-8") as f:
        json.dump(scenes, f)

    num_objects = sum(len(s["objects"]) for s in scenes.values())
    empty_scenes = sum(1 for s in scenes.values() if not s["objects"])
    print(f"Scenes built: {len(scenes)}")
    print(f"Total objects: {num_objects}")
    print(f"Scenes with 0 objects: {empty_scenes}")


if __name__ == "__main__":
    main()
