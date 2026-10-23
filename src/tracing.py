from pathlib import Path


class Tracer:
    def __init__(self, output_path: Path):
        raise NotImplementedError

    def log_step(self, image_id: str, module: str, params: dict, status: str, **extra) -> None:
        raise NotImplementedError
