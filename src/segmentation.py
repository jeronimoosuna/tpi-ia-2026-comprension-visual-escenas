from PIL import Image


def segment_with_sam2(image: Image.Image, boxes: list[list[float]]) -> list[dict]:
    """Returns one binary mask per box, same order as boxes."""
    raise NotImplementedError


def mask_to_rle(mask, order: str = "F") -> dict:
    """Returns {size: [h, w], counts: [int, ...]} per scene_schema.json."""
    raise NotImplementedError
