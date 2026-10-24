import numpy as np
from PIL import Image

from src.geometry import bbox_iou, bbox_to_corners

POSITIONS = [
    "top_left", "top_center", "top_right",
    "middle_left", "center", "middle_right",
    "bottom_left", "bottom_center", "bottom_right",
]
SIZES = ["small", "medium", "large"]
PREDICATES = ["left_of", "right_of", "above", "below", "overlaps"]

# Empirical tertiles of area_ratio over the 1790 real annotations of the
# development set (reports/eda/object_sizes.csv), not arbitrary round
# numbers: ensures small/medium/large are roughly balanced on this dataset.
SMALL_MEDIUM_THRESHOLD = 0.00126
MEDIUM_LARGE_THRESHOLD = 0.00765

# Minimum bbox IoU to call two objects "overlaps" rather than ignoring
# the pair; avoids tagging near-zero grazing overlaps as a relation.
OVERLAP_IOU_THRESHOLD = 0.01


def compute_centroid(mask: np.ndarray) -> list[float]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return [0.0, 0.0]
    return [float(xs.mean()), float(ys.mean())]


def compute_area_ratio(mask: np.ndarray, image_width: int, image_height: int) -> float:
    return float(mask.sum()) / (image_width * image_height)


def compute_position(centroid: list[float], image_width: int, image_height: int) -> str:
    cx, cy = centroid
    col = min(2, int(cx / image_width * 3))
    row = min(2, int(cy / image_height * 3))
    return POSITIONS[row * 3 + col]


def compute_size(area_ratio: float) -> str:
    if area_ratio < SMALL_MEDIUM_THRESHOLD:
        return "small"
    if area_ratio < MEDIUM_LARGE_THRESHOLD:
        return "medium"
    return "large"


def compute_dominant_color(image: Image.Image, mask: np.ndarray, bin_size: int = 24) -> tuple[int, int, int]:
    """Quantizes masked pixels into bin_size-wide RGB bins and returns the
    center of the most frequent bin (the actual dominant color, not just
    a central-tendency average that could land outside any real color)."""
    arr = np.array(image.convert("RGB"))
    pixels = arr[mask]
    if len(pixels) == 0:
        return (0, 0, 0)

    quantized = (pixels // bin_size) * bin_size + bin_size // 2
    colors, counts = np.unique(quantized, axis=0, return_counts=True)
    dominant = colors[np.argmax(counts)]
    return tuple(int(c) for c in dominant)


NEAREST_NEIGHBORS = 5


def _bbox_center(bbox: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox_to_corners(bbox)
    return (x1 + x2) / 2, (y1 + y2) / 2


def compute_relations(objects: list[dict], k: int = NEAREST_NEIGHBORS) -> list[dict]:
    """objects: list of {id, bbox} (bbox in xywh) for a single image.
    Returns [{subject, predicate, object}, ...] using object ids.

    Relations are only computed between each object and its k nearest
    neighbors (by centroid distance), not all O(n^2) pairs: on a crowded
    image (e.g. 257 objects) "left_of"/"above" between every axis-aligned
    pair produced >12000 near-meaningless relations (unrelated objects on
    opposite sides of the frame that happen to share a row/column). Capping
    at k nearest neighbors keeps relations tied to objects that are
    actually close to each other, which is what the "útil" requirement in
    the assignment is after.

    For each kept pair:
    - "overlaps" when bbox IoU > OVERLAP_IOU_THRESHOLD.
    - otherwise "left_of" when the pair's vertical (y) ranges intersect
      (roughly the same height), oriented so subject has the smaller
      center x.
    - otherwise "above" when the pair's horizontal (x) ranges intersect
      (roughly the same column), oriented so subject has the smaller
      center y (image y grows downward).
    Only one canonical direction is stored per axis (left_of, above); the
    inverse (right_of, below) is implied and left for the QA engine to
    check both directions when asked."""
    n = len(objects)
    if n < 2:
        return []

    centers = [_bbox_center(o["bbox"]) for o in objects]
    pairs: set[tuple[int, int]] = set()
    for i in range(n):
        distances = sorted(
            (
                (centers[i][0] - centers[j][0]) ** 2 + (centers[i][1] - centers[j][1]) ** 2,
                j,
            )
            for j in range(n) if j != i
        )
        for _, j in distances[:k]:
            pairs.add((i, j) if i < j else (j, i))

    relations = []
    for i, j in sorted(pairs):
        a, b = objects[i], objects[j]
        iou = bbox_iou(a["bbox"], b["bbox"])
        if iou > OVERLAP_IOU_THRESHOLD:
            relations.append({"subject": a["id"], "predicate": "overlaps", "object": b["id"]})
            continue

        ax1, ay1, ax2, ay2 = bbox_to_corners(a["bbox"])
        bx1, by1, bx2, by2 = bbox_to_corners(b["bbox"])
        y_overlap = ay1 < by2 and by1 < ay2
        x_overlap = ax1 < bx2 and bx1 < ax2

        if y_overlap:
            acx, bcx = (ax1 + ax2) / 2, (bx1 + bx2) / 2
            left, right = (a, b) if acx < bcx else (b, a)
            relations.append({"subject": left["id"], "predicate": "left_of", "object": right["id"]})
        if x_overlap:
            acy, bcy = (ay1 + ay2) / 2, (by1 + by2) / 2
            top, bottom = (a, b) if acy < bcy else (b, a)
            relations.append({"subject": top["id"], "predicate": "above", "object": bottom["id"]})
    return relations
