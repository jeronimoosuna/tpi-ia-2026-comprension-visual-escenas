POSITIONS = [
    "top_left", "top_center", "top_right",
    "middle_left", "center", "middle_right",
    "bottom_left", "bottom_center", "bottom_right",
]
SIZES = ["small", "medium", "large"]
PREDICATES = ["left_of", "right_of", "above", "below", "overlaps"]


def compute_centroid(mask) -> list[float]:
    raise NotImplementedError


def compute_area_ratio(mask, image_width: int, image_height: int) -> float:
    raise NotImplementedError


def compute_position(centroid: list[float], image_width: int, image_height: int) -> str:
    raise NotImplementedError


def compute_size(area_ratio: float) -> str:
    raise NotImplementedError


def compute_dominant_color(image, mask) -> tuple[int, int, int]:
    raise NotImplementedError


def compute_relations(objects: list[dict]) -> list[dict]:
    """Returns [{subject, predicate, object}, ...] using object ids."""
    raise NotImplementedError
