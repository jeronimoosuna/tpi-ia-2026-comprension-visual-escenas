import json
from itertools import combinations
from pathlib import Path

import pandas as pd

from src.geometry import bbox_iou


def load_annotations(development_dir: Path) -> dict:
    with open(Path(development_dir) / "annotations.json", "r", encoding="utf-8") as f:
        return json.load(f)


def category_distribution(annotations: dict) -> pd.DataFrame:
    cat_names = {c["id"]: c["name"] for c in annotations["categories"]}
    ann_df = pd.DataFrame(annotations["annotations"])

    num_annotations = ann_df.groupby("category_id").size()
    num_images = ann_df.groupby("category_id")["image_id"].nunique()

    df = pd.DataFrame({
        "category_id": num_annotations.index,
        "name": [cat_names[c] for c in num_annotations.index],
        "num_annotations": num_annotations.values,
        "num_images": num_images.reindex(num_annotations.index).values,
    })
    return df.sort_values("num_annotations", ascending=False).reset_index(drop=True)


def instances_per_image(annotations: dict) -> pd.Series:
    ann_df = pd.DataFrame(annotations["annotations"])
    counts = ann_df.groupby("image_id").size()
    all_image_ids = [img["id"] for img in annotations["images"]]
    return counts.reindex(all_image_ids, fill_value=0).rename("num_instances")


def object_size_distribution(annotations: dict) -> pd.DataFrame:
    img_dims = {img["id"]: img["width"] * img["height"] for img in annotations["images"]}
    rows = []
    for ann in annotations["annotations"]:
        image_area = img_dims[ann["image_id"]]
        _, _, w, h = ann["bbox"]
        rows.append({
            "annotation_id": ann.get("id"),
            "image_id": ann["image_id"],
            "category_id": ann["category_id"],
            "area": ann.get("area", w * h),
            "area_ratio": ann.get("area", w * h) / image_area,
            "bbox_w": w,
            "bbox_h": h,
        })
    return pd.DataFrame(rows)


def occlusion_stats(annotations: dict) -> pd.DataFrame:
    ann_df = pd.DataFrame(annotations["annotations"])
    rows = []
    for image_id, group in ann_df.groupby("image_id"):
        boxes = group["bbox"].tolist()
        ann_ids = group["id"].tolist() if "id" in group else list(range(len(boxes)))
        max_iou = {aid: 0.0 for aid in ann_ids}
        for (i, bbox_a), (j, bbox_b) in combinations(enumerate(boxes), 2):
            iou = bbox_iou(bbox_a, bbox_b)
            if iou > max_iou[ann_ids[i]]:
                max_iou[ann_ids[i]] = iou
            if iou > max_iou[ann_ids[j]]:
                max_iou[ann_ids[j]] = iou
        for aid, iou in max_iou.items():
            rows.append({"image_id": image_id, "annotation_id": aid, "max_neighbor_iou": iou})
    return pd.DataFrame(rows)


def negative_and_non_exhaustive_categories(annotations: dict) -> dict:
    cat_names = {c["id"]: c["name"] for c in annotations["categories"]}

    neg_counts: dict[int, int] = {}
    not_exhaustive_counts: dict[int, int] = {}
    images_with_neg = 0
    images_with_not_exhaustive = 0

    for img in annotations["images"]:
        neg_ids = img.get("neg_category_ids", [])
        not_exh_ids = img.get("not_exhaustive_category_ids", [])
        if neg_ids:
            images_with_neg += 1
        if not_exh_ids:
            images_with_not_exhaustive += 1
        for cid in neg_ids:
            neg_counts[cid] = neg_counts.get(cid, 0) + 1
        for cid in not_exh_ids:
            not_exhaustive_counts[cid] = not_exhaustive_counts.get(cid, 0) + 1

    return {
        "images_with_neg_category_ids": images_with_neg,
        "images_with_not_exhaustive_category_ids": images_with_not_exhaustive,
        "neg_category_counts": {
            cat_names[cid]: n for cid, n in sorted(neg_counts.items(), key=lambda kv: -kv[1])
        },
        "not_exhaustive_category_counts": {
            cat_names[cid]: n for cid, n in sorted(not_exhaustive_counts.items(), key=lambda kv: -kv[1])
        },
    }
