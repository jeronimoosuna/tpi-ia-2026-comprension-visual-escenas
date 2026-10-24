def build_scene_json(
    image_id: str,
    width: int,
    height: int,
    objects: list[dict],
) -> dict:
    """objects: list of {category_id, confidence, bbox, mask} (e.g. from
    segmentation.segment_with_sam2 + mask_to_rle, one per detected+segmented
    instance). Assigns each object a unique string id.

    This only assembles the structural skeleton (image, objects, relations)
    required by scene_schema.json: centroid/area_ratio/position/size are
    NOT computed here (src/properties.py fills those into each object
    afterward) and relations starts empty (filled by
    properties.compute_relations once objects have centroids/bboxes)."""
    scene_objects = []
    for i, obj in enumerate(objects, start=1):
        scene_objects.append({
            "id": f"{image_id}_obj{i}",
            "category_id": obj["category_id"],
            "confidence": obj["confidence"],
            "bbox": obj["bbox"],
            "mask": obj["mask"],
        })

    return {
        "image": {"id": image_id, "width": width, "height": height},
        "objects": scene_objects,
        "relations": [],
    }
