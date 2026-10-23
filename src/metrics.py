def detection_f1(predictions: list[dict], ground_truth: dict, iou_threshold: float = 0.5) -> dict:
    """LVIS-aware: distinguishes positive / neg_category_ids / not_exhaustive_category_ids
    / unverified categories when counting false positives and false negatives."""
    raise NotImplementedError


def mask_f1(predictions: list[dict], ground_truth: dict, iou_threshold: float = 0.5) -> dict:
    raise NotImplementedError


def scene_json_consistency(scene_json: dict, schema: dict) -> dict:
    """Validates against scene_schema.json, RLE decodability, and id references."""
    raise NotImplementedError


def qa_accuracy(predicted_answers: list[dict], reference_answers: list[dict]) -> dict:
    raise NotImplementedError
