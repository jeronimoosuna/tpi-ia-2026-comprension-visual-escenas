from src.geometry import bbox_iou


def detection_f1(predictions: list[dict], ground_truth: dict, iou_threshold: float = 0.5) -> dict:
    """predictions: flat list of {image_id, category_id, confidence, bbox}.
    ground_truth: annotations.json dict (images with neg_category_ids /
    not_exhaustive_category_ids, categories, annotations).

    LVIS-aware policy: a category_id is positive for an image if it has
    GT annotations there. Positive + not_exhaustive: unmatched predictions
    are ignored (not FP). Positive, not not_exhaustive: unmatched
    predictions are FP. In neg_category_ids: every prediction is FP (no
    GT to miss, so no FN). Neither positive nor negative (unverified):
    predictions are ignored entirely, since absence of annotations doesn't
    prove the object is absent."""
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
                        iou = bbox_iou(pred["bbox"], ann["bbox"])
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


def mask_f1(predictions: list[dict], ground_truth: dict, iou_threshold: float = 0.5) -> dict:
    raise NotImplementedError


def scene_json_consistency(scene_json: dict, schema: dict) -> dict:
    """Validates against scene_schema.json, RLE decodability, and id references."""
    raise NotImplementedError


def qa_accuracy(predicted_answers: list[dict], reference_answers: list[dict]) -> dict:
    raise NotImplementedError
