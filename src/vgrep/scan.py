"""Walk a directory for indexable files, respecting ignore rules."""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Iterator

from .config import IMAGE_SUFFIXES, SKIP_DIRS


def load_ignore_patterns(root: Path) -> list[str]:
    """Read .vgrepignore (gitignore-style globs, one per line)."""
    f = root / ".vgrepignore"
    if not f.exists():
        return []
    return [
        line.strip()
        for line in f.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def _ignored(rel: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(Path(rel).name, p) for p in patterns)


def iter_images(root: Path) -> Iterator[Path]:
    """Yield image files under root, skipping noise directories and ignored globs."""
    root = root.expanduser().resolve()
    patterns = load_ignore_patterns(root)

    for path in root.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.name.startswith("."):
            continue
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if patterns and _ignored(str(path.relative_to(root)), patterns):
            continue
        yield path
