"""Shared small utilities: UTF-8 console, JSONL IO, seeding, paths."""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent


def force_utf8_stdout() -> None:
    """Windows consoles default to cp1252 and cannot print Sinhala."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass


def seed_everything(seed: int) -> random.Random:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random.Random(seed)


def read_jsonl(path: str | Path) -> Iterator[dict]:
    p = Path(path)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict], mode: str = "w") -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open(mode, encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl(path: str | Path, row: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    import yaml
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    with p.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
