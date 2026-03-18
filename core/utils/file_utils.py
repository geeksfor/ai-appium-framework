from pathlib import Path
from datetime import datetime


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def timestamp_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_screenshot_path(base_dir: str, prefix: str = "screenshot") -> str:
    evidence_dir = ensure_dir(base_dir)
    file_name = f"{prefix}_{timestamp_str()}.png"
    return str(evidence_dir / file_name)