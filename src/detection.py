import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor

from src.geometry import bbox_iou

_MODEL_NAME = "google/owlv2-base-patch16-ensemble"
_processor = None
_model = None


def _get_model():
    global _processor, _model
    if _model is None:
        _processor = Owlv2Processor.from_pretrained(_MODEL_NAME)
        _model = Owlv2ForObjectDetection.from_pretrained(_MODEL_NAME)
        _model.eval()
    return _processor, _model


def detect_objects(
    image: Image.Image,
    vocabulary: dict[str, int],
    confidence_threshold: float,
) -> list[dict]:
    """vocabulary maps category name -> category_id (from annotations.json
    categories). Returns a list of {category_id, confidence, bbox} (bbox in
    xywh) before NMS/dedup."""
    processor, model = _get_model()
    names = list(vocabulary.keys())

    inputs = processor(text=[names], images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_grounded_object_detection(
        outputs=outputs,
        threshold=confidence_threshold,
        target_sizes=target_sizes,
        text_labels=[names],
    )[0]

    detections = []
    for score, box, name in zip(results["scores"], results["boxes"], results["text_labels"]):
        x1, y1, x2, y2 = box.tolist()
        detections.append({
            "category_id": vocabulary[name],
            "confidence": float(score),
            "bbox": [x1, y1, x2 - x1, y2 - y1],
        })
    return detections


def suppress_duplicates(detections: list[dict], iou_threshold: float) -> list[dict]:
    """Per-category NMS: keeps the highest-confidence box and drops any
    same-category box overlapping it above iou_threshold."""
    by_category: dict[int, list[dict]] = {}
    for det in detections:
        by_category.setdefault(det["category_id"], []).append(det)

    kept = []
    for dets in by_category.values():
        dets = sorted(dets, key=lambda d: d["confidence"], reverse=True)
        chosen: list[dict] = []
        for det in dets:
            if all(bbox_iou(det["bbox"], c["bbox"]) <= iou_threshold for c in chosen):
                chosen.append(det)
        kept.extend(chosen)
    return kept
