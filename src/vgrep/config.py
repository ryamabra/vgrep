"""Paths and tunables. Everything lives under one directory so `rm -rf` is a clean uninstall."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Model identity is recorded in the DB. If it changes, existing vectors are
# meaningless and must be re-encoded -- see db.check_model_compatibility.
DEFAULT_MODEL = "google/siglip2-base-patch16-224"
EMBED_DIM = 768

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".gif", ".tiff", ".bmp"}

SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    ".vgrep", "Library", ".Trash", ".cache",
}


def data_dir() -> Path:
    """Root for all vgrep state. Override with VGREP_HOME for tests."""
    root = os.environ.get("VGREP_HOME")
    p = Path(root) if root else Path.home() / ".vgrep"
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass(frozen=True)
class Settings:
    model: str = DEFAULT_MODEL
    dim: int = EMBED_DIM
    batch_size: int = 16
    # Number of processes decoding/resizing images to feed the encoder.
    # Decode is usually the bottleneck, not the model.
    loader_workers: int = max(2, (os.cpu_count() or 4) - 2)
    top_k: int = 10

    @property
    def db_path(self) -> Path:
        return data_dir() / "vgrep.db"

    @property
    def index_path(self) -> Path:
        return data_dir() / "vgrep.npz"


SETTINGS = Settings()
