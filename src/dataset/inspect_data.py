"""Load and profile the raw Wikipedia parquet dump.

Schema is not assumed: we print columns, dtypes, row count and samples, then
run a few heuristic probes so downstream stages can be adapted to reality.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import force_utf8_stdout  # noqa: E402


def profile(path: Path, n_samples: int = 3, sample_chars: int = 700) -> dict:
    pf = pq.ParquetFile(path)
    meta = pf.metadata
    print("=" * 78)
    print(f"FILE: {path}  ({path.stat().st_size / 1e6:.1f} MB)")
    print(f"rows={meta.num_rows}  row_groups={meta.num_row_groups}  cols={meta.num_columns}")
    print("=" * 78)

    print("\n--- ARROW SCHEMA ---")
    print(pf.schema_arrow)

    # Read a bounded slice for dtype/sample inspection; full file may be large.
    head = next(pf.iter_batches(batch_size=2000)).to_pandas()
    print("\n--- PANDAS DTYPES (first batch) ---")
    print(head.dtypes.to_string())

    print(f"\n--- {n_samples} SAMPLE ROWS ---")
    for i in range(min(n_samples, len(head))):
        row = head.iloc[i]
        print(f"\n===== ROW {i} =====")
        for col in head.columns:
            val = row[col]
            s = str(val)
            if len(s) > sample_chars:
                s = s[:sample_chars] + f" ... [TRUNCATED, total len={len(str(val))}]"
            print(f"  [{col}] {s}")

    # Heuristic probes on the text-like column(s)
    text_cols = [c for c in head.columns if head[c].dtype == object
                 and head[c].astype(str).str.len().mean() > 200]
    print(f"\n--- CANDIDATE TEXT COLUMNS: {text_cols} ---")

    stats: dict = {"rows": meta.num_rows, "columns": list(head.columns),
                   "text_columns": text_cols}

    if text_cols:
        col = text_cols[0]
        lens = head[col].astype(str).str.len()
        print(f"\nchar-length of '{col}' (first {len(head)} rows):")
        print(lens.describe().to_string())
        stats["text_col"] = col
        stats["char_len_describe"] = {k: float(v) for k, v in lens.describe().items()}

        # markup / page-type residue probes
        probes = {
            "has_{{template}}": r"\{\{",
            "has_[[link]]": r"\[\[",
            "has_table_{|": r"\{\|",
            "has_html_comment": r"<!--",
            "has_ref_tag": r"<ref",
            "has_Category:": r"\[\[\s*(?:Category|ප්‍රවර්ගය)\s*:",
            "has_File/ගොනු": r"\[\[\s*(?:File|Image|ගොනු|රූපය)\s*:",
            "has_REDIRECT": r"(?i)#(?:REDIRECT|යළි[\s\S]{0,12}යොමු)",
            "disambiguation": r"බහුරුත්හරණය",
            "stub_tag": r"දියුණු\s*කරන්න",
            "sinhala_chars": r"[\u0D80-\u0DFF]",
        }
        print("\nmarkup/page-type probe hit-rate (fraction of first batch):")
        stats["probes"] = {}
        for name, pat in probes.items():
            frac = head[col].astype(str).str.contains(pat, regex=True, na=False).mean()
            stats["probes"][name] = float(frac)
            print(f"  {name:22s} {frac:6.1%}")

    # title-column probe for namespace prefixes
    title_cols = [c for c in head.columns if "title" in c.lower() or "name" in c.lower()]
    if title_cols:
        tc = title_cols[0]
        stats["title_col"] = tc
        ns = head[tc].astype(str).str.extract(r"^([^:]{2,20}):")[0].value_counts().head(15)
        print(f"\n--- namespace-like prefixes in '{tc}' (first batch) ---")
        print(ns.to_string() if len(ns) else "  (none)")
    return stats


def main() -> None:
    force_utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="wikipedia.parquet")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()
    stats = profile(Path(a.path), a.samples)
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
