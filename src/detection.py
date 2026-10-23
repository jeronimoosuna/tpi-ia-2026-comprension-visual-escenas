from PIL import Image


def detect_objects(
    image: Image.Image,
    vocabulary: list[str],
    confidence_threshold: float,
    input_resolution: int,
) -> list[dict]:
    """Returns a list of {category_id, confidence, bbox} before NMS/dedup."""
    raise NotImplementedError


def suppress_duplicates(detections: list[dict], iou_threshold: float) -> list[dict]:
    raise NotImplementedError
