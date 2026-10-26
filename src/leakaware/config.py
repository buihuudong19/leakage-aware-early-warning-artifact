"""Configuration loading. Relative paths in config.yaml are resolved from the repository root."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path=None):
    path = Path(path) if path else ROOT / "config.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["data_dir"] = (ROOT / cfg["data_dir"]).resolve() if not Path(cfg["data_dir"]).is_absolute() else Path(cfg["data_dir"])
    cfg["out_dir"] = (ROOT / cfg["out_dir"]).resolve()
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)
    return cfg
