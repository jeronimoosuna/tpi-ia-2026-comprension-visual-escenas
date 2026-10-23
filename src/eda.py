from pathlib import Path


def load_annotations(development_dir: Path) -> dict:
    raise NotImplementedError


def category_distribution(annotations: dict) -> dict:
    raise NotImplementedError


def instances_per_image(annotations: dict) -> dict:
    raise NotImplementedError


def object_size_distribution(annotations: dict) -> dict:
    raise NotImplementedError


def occlusion_stats(annotations: dict) -> dict:
    raise NotImplementedError


def negative_and_non_exhaustive_categories(annotations: dict) -> dict:
    raise NotImplementedError
