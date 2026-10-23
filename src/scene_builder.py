def build_scene_json(
    image_id: str,
    width: int,
    height: int,
    detections: list[dict],
    masks: list[dict],
) -> dict:
    """Assembles the root SceneJSON object (image, objects, relations)."""
    raise NotImplementedError
