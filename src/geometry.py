def bbox_to_corners(bbox: list[float]) -> tuple[float, float, float, float]:
    x, y, w, h = bbox
    return x, y, x + w, y + h


def bbox_iou(bbox_a: list[float], bbox_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = bbox_to_corners(bbox_a)
    bx1, by1, bx2, by2 = bbox_to_corners(bbox_b)

    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter_area / (area_a + area_b - inter_area)
