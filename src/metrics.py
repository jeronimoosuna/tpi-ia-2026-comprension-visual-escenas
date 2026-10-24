from typing import Callable

import jsonschema
import numpy as np
from PIL import Image, ImageDraw

from src.geometry import bbox_iou
from src.segmentation import rle_to_mask


def _lvis_aware_f1(
    predictions: list[dict],
    ground_truth: dict,
    iou_fn: Callable[[dict, dict], float],
    iou_threshold: float,
) -> dict:
    """Shared policy for detection_f1/mask_f1: a category_id is positive for
    an image if it has GT annotations there. Positive + not_exhaustive:
    unmatched predictions are ignored (not FP). Positive, not
    not_exhaustive: unmatched predictions are FP. In neg_category_ids:
    every prediction is FP (no GT to miss, so no FN). Neither positive nor
    negative (unverified): predictions are ignored entirely, since absence
    of annotations doesn't prove the object is absent. iou_fn(pred, gt_ann)
    returns the overlap metric (bbox or mask IoU) between the two."""
    gt_by_image: dict[int, list[dict]] = {}
    for ann in ground_truth["annotations"]:
        gt_by_image.setdefault(ann["image_id"], []).append(ann)

    preds_by_image: dict[int, list[dict]] = {}
    for pred in predictions:
        preds_by_image.setdefault(pred["image_id"], []).append(pred)

    tp = fp = fn = 0
    ignored_predictions = 0

    for image in ground_truth["images"]:
        image_id = image["id"]
        gts = gt_by_image.get(image_id, [])
        preds = preds_by_image.get(image_id, [])
        neg_ids = set(image.get("neg_category_ids", []))
        not_exh_ids = set(image.get("not_exhaustive_category_ids", []))
        positive_ids = {ann["category_id"] for ann in gts}

        gt_by_category: dict[int, list[dict]] = {}
        for ann in gts:
            gt_by_category.setdefault(ann["category_id"], []).append(ann)
        preds_by_category: dict[int, list[dict]] = {}
        for pred in preds:
            preds_by_category.setdefault(pred["category_id"], []).append(pred)

        all_category_ids = positive_ids | neg_ids | set(preds_by_category.keys())
        for category_id in all_category_ids:
            cat_gts = gt_by_category.get(category_id, [])
            cat_preds = sorted(preds_by_category.get(category_id, []), key=lambda p: -p["confidence"])

            if category_id in positive_ids:
                matched_gt = [False] * len(cat_gts)
                for pred in cat_preds:
                    best_iou, best_idx = 0.0, -1
                    for idx, ann in enumerate(cat_gts):
                        if matched_gt[idx]:
                            continue
                        iou = iou_fn(pred, ann)
                        if iou > best_iou:
                            best_iou, best_idx = iou, idx
                    if best_iou >= iou_threshold:
                        matched_gt[best_idx] = True
                        tp += 1
                    elif category_id in not_exh_ids:
                        ignored_predictions += 1
                    else:
                        fp += 1
                fn += matched_gt.count(False)
            elif category_id in neg_ids:
                fp += len(cat_preds)
            else:
                ignored_predictions += len(cat_preds)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "ignored_predictions": ignored_predictions,
    }


def detection_f1(predictions: list[dict], ground_truth: dict, iou_threshold: float = 0.5) -> dict:
    """predictions: flat list of {image_id, category_id, confidence, bbox}.
    ground_truth: annotations.json dict (images with neg_category_ids /
    not_exhaustive_category_ids, categories, annotations). See
    _lvis_aware_f1 for the matching policy."""
    return _lvis_aware_f1(
        predictions, ground_truth,
        iou_fn=lambda pred, ann: bbox_iou(pred["bbox"], ann["bbox"]),
        iou_threshold=iou_threshold,
    )


def _polygon_to_mask(segmentation: list[list[float]], height: int, width: int) -> np.ndarray:
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    for polygon in segmentation:
        coords = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
        draw.polygon(coords, outline=1, fill=1)
    return np.array(image, dtype=bool)


def mask_f1(predictions: list[dict], ground_truth: dict, iou_threshold: float = 0.5) -> dict:
    """predictions: flat list of {image_id, category_id, confidence, mask}
    (mask = our RLE format, as produced by segmentation.mask_to_rle).
    ground_truth: annotations.json dict, with polygon segmentations
    (COCO-style list of [x1,y1,x2,y2,...]). Same LVIS-aware policy as
    detection_f1, but matching on mask IoU instead of bbox IoU."""
    image_hw = {img["id"]: (img["height"], img["width"]) for img in ground_truth["images"]}
    gt_mask_cache: dict[int, np.ndarray] = {}

    def iou_fn(pred: dict, ann: dict) -> float:
        pred_mask = rle_to_mask(pred["mask"])
        ann_id = ann["id"]
        if ann_id not in gt_mask_cache:
            h, w = image_hw[ann["image_id"]]
            gt_mask_cache[ann_id] = _polygon_to_mask(ann["segmentation"], h, w)
        gt_mask = gt_mask_cache[ann_id]

        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        return float(intersection) / float(union) if union else 0.0

    return _lvis_aware_f1(predictions, ground_truth, iou_fn=iou_fn, iou_threshold=iou_threshold)


def scene_json_consistency(scene_json: dict, schema: dict) -> dict:
    """Validates against scene_schema.json, RLE decodability, and id
    references (relations' subject/object, questions'/questions_llm's
    evidence must all point at known object ids). Returns {valid, errors}."""
    validator = jsonschema.Draft202012Validator(schema)
    errors = [e.message for e in validator.iter_errors(scene_json)]

    object_ids = {o["id"] for o in scene_json.get("objects", [])}
    for i, obj in enumerate(scene_json.get("objects", [])):
        label = obj.get("id", f"index {i}")
        try:
            counts_sum = sum(obj["mask"]["counts"])
            h, w = obj["mask"]["size"]
            if counts_sum != h * w:
                errors.append(
                    f"object {label}: RLE counts sum to {counts_sum}, expected height*width={h * w}"
                )
            mask = rle_to_mask(obj["mask"])
            if mask.shape != (h, w):
                errors.append(f"object {label}: decoded mask shape {mask.shape} != declared size {(h, w)}")
        except Exception as exc:
            errors.append(f"object {label}: RLE decode failed ({exc})")

    for i, rel in enumerate(scene_json.get("relations", [])):
        if rel.get("subject") not in object_ids:
            errors.append(f"relation {i}: subject {rel.get('subject')!r} is not a known object id")
        if rel.get("object") not in object_ids:
            errors.append(f"relation {i}: object {rel.get('object')!r} is not a known object id")

    for field in ("questions", "questions_llm"):
        for q in scene_json.get(field, []):
            for evidence_id in q.get("evidence", []):
                if evidence_id not in object_ids:
                    errors.append(
                        f"{field} {q.get('question_id')!r}: evidence {evidence_id!r} is not a known object id"
                    )

    return {"valid": len(errors) == 0, "errors": errors}


def qa_accuracy(predicted_answers: list[dict], reference_answers: list[dict]) -> dict:
    """predicted_answers: questionAnswer dicts (question_id, type,
    normalized_answer, ...), e.g. from qa_engine.answer_question.
    reference_answers: [{question_id, normalized_answer}, ...] built from
    ground truth (e.g. real counts from annotations.json) -- only
    questions with a known reference are scored. "unsupported" predictions
    are excluded from the accuracy denominator and reported separately,
    since they're an explicit abstention, not a wrong guess."""
    reference_by_id = {r["question_id"]: r["normalized_answer"] for r in reference_answers}

    correct = 0
    scored = 0
    unsupported = 0
    by_type_correct: dict[str, int] = {}
    by_type_scored: dict[str, int] = {}

    for pred in predicted_answers:
        question_id = pred["question_id"]
        if question_id not in reference_by_id:
            continue
        qtype = pred["type"]
        if pred["normalized_answer"] == "unsupported":
            unsupported += 1
            continue
        scored += 1
        by_type_scored[qtype] = by_type_scored.get(qtype, 0) + 1
        if pred["normalized_answer"] == reference_by_id[question_id]:
            correct += 1
            by_type_correct[qtype] = by_type_correct.get(qtype, 0) + 1

    return {
        "accuracy": correct / scored if scored else 0.0,
        "correct": correct,
        "scored": scored,
        "unsupported": unsupported,
        "by_type": {t: by_type_correct.get(t, 0) / n for t, n in by_type_scored.items()},
    }
