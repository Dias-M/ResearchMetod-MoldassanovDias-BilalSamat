import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402


def load_config(path: str | None = None) -> dict:
    with open(path or ROOT / "config" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def p(cfg: dict, key: str) -> Path:
    d = ROOT / cfg["paths"][key]
    d.mkdir(parents=True, exist_ok=True)
    return d
