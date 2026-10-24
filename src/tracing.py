import json
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


class Tracer:
    """Appends one JSON line per step to output_path (JSON Lines: easy to
    audit, grep, and reload without parsing the whole file as one JSON
    document). Every record carries a run_id so traces from different runs
    can share a file without being confused for each other."""

    def __init__(self, output_path: Path):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = uuid.uuid4().hex[:8]

    def log_step(self, image_id: str, module: str, params: dict, status: str, **extra) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "image_id": image_id,
            "module": module,
            "status": status,
            "params": params,
            **extra,
        }
        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    @contextmanager
    def step(self, image_id: str, module: str, params: dict | None = None):
        """Times the block and logs it automatically: status="ok" with
        duration_seconds on success, or status="error" with the exception
        message on failure (then re-raises, so the step still fails loudly
        -- it just also leaves a record of where it failed)."""
        start = time.time()
        try:
            yield
        except Exception as exc:
            self.log_step(image_id, module, params or {}, status="error",
                          duration_seconds=time.time() - start, error=str(exc))
            raise
        else:
            self.log_step(image_id, module, params or {}, status="ok",
                          duration_seconds=time.time() - start)


def load_trace(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
