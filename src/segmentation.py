import numpy as np
import torch
from PIL import Image
from transformers import Sam2Model, Sam2Processor

_MODEL_NAME = "facebook/sam2.1-hiera-tiny"
_processor = None
_model = None


def _get_model():
    global _processor, _model
    if _model is None:
        _processor = Sam2Processor.from_pretrained(_MODEL_NAME)
        _model = Sam2Model.from_pretrained(_MODEL_NAME)
        _model.eval()
    return _processor, _model


def segment_with_sam2(image: Image.Image, boxes: list[list[float]]) -> list[np.ndarray]:
    """boxes: list of [x, y, w, h] (same convention as detection.detect_objects).
    Returns one boolean HxW mask per box, same order as boxes, same size as image."""
    if not boxes:
        return []

    processor, model = _get_model()
    boxes_xyxy = [[x, y, x + w, y + h] for x, y, w, h in boxes]

    inputs = processor(images=image, input_boxes=[boxes_xyxy], return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)

    masks = processor.post_process_masks(outputs.pred_masks, inputs["original_sizes"])[0]
    # masks: (num_boxes, 1, H, W) bool tensor
    return [masks[i, 0].numpy() for i in range(masks.shape[0])]


def mask_to_rle(mask: np.ndarray) -> dict:
    """Uncompressed RLE, column-major (Fortran) order, per scene_schema.json:
    {size: [h, w], counts: [int, ...]} with counts alternating background/
    foreground run-lengths, starting with background (can be 0)."""
    h, w = mask.shape
    flat = mask.flatten(order="F").astype(np.uint8)

    counts: list[int] = []
    prev = 0
    run_length = 0
    for value in flat:
        value = int(value)
        if value == prev:
            run_length += 1
        else:
            counts.append(run_length)
            prev = value
            run_length = 1
    counts.append(run_length)

    return {"size": [h, w], "counts": counts}


def rle_to_mask(rle: dict) -> np.ndarray:
    h, w = rle["size"]
    flat = np.zeros(h * w, dtype=bool)
    pos = 0
    value = False
    for run_length in rle["counts"]:
        if value:
            flat[pos:pos + run_length] = True
        pos += run_length
        value = not value
    return flat.reshape((h, w), order="F")
