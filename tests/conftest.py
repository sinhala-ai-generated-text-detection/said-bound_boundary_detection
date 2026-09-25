"""Make the source folders importable from every test module."""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
for sub in ("", "dataset", "detect"):
    sys.path.insert(0, str(SRC / sub))
