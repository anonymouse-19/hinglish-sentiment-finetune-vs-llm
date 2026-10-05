from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(path: str | Path) -> Path:
    """Interpret config paths relative to the project root, so scripts work from any cwd."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path
