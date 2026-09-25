"""Matched cross-model comparison.

Runs every generator on the SAME source window, the SAME construction type and
the SAME span plan, so differences in the output are attributable to the model
and nothing else. The pilot cannot answer this on its own: it assigns different
windows to each generator, and the held-out generator only sees dev/test.

Writes reports/generation/model_comparison.md:
  - an at-a-glance table, one row per example, one column per model
  - a per-example table putting the human original beside every model's text

Results are cached in reports/generation/evidence/model_comparison.jsonl, so re-running to
re-render the markdown costs nothing.

    python src/compare_models.py --per-type 3
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate import (  # noqa: E402
    ALL_TYPES,
    Generator,
    load_prompts,
)
from openrouter import OpenRouterClient  # noqa: E402
from segment import segment_sentences, word_count  # noqa: E402
from split import load_split_map  # noqa: E402
from utils import (  # noqa: E402
    force_utf8_stdout,
    load_config,
    read_jsonl,
    write_jsonl,
)

CACHE = "reports/generation/evidence/model_comparison.jsonl"


def human_counterpart(window: dict, plan: dict) -> list[str]:
    """The human sentences the model was asked to replace or continue past.

    Type 1 discards the true remainder, but it is recoverable from the window,
    which is what makes a like-for-like human column possible.
    """
    if plan["construction_type"] == ALL_TYPES[0]:
        return list(plan["removed"])
    return [s for target in plan["targets"] for s in target]


def collect(cfg: dict, per_type: int, generators: list[str]) -> list[dict]:
    force_utf8_stdout()
    windows = list(read_jsonl(Path(cfg["paths"]["sources"]).parent / "windows.jsonl"))
    split_map = load_split_map(cfg)

    # Every generator must be allowed to write the chosen split, otherwise the
    # held-out model would be missing from the comparison.
    allowed = set.intersection(*(set(cfg["generators"][g]["splits"])
                                 for g in generators))
    pool = [w for w in windows if split_map.get(w["source_id"]) in allowed]
    rng = random.Random(cfg["seed"])
    rng.shuffle(pool)
    print(f"eligible windows in splits {sorted(allowed)}: {len(pool)}")

    prompts = load_prompts(cfg)
    client = OpenRouterClient(cfg)
    gen = Generator(cfg, client, prompts)

    done = {(r["example_id"], r["generator"]) for r in read_jsonl(CACHE)}
    rows: list[dict] = list(read_jsonl(CACHE))

    used: set[str] = set()
    for ctype in ALL_TYPES:
        picked = 0
        for w in pool:
            if picked >= per_type:
                break
            if w["window_id"] in used:
                continue
            if w["n_sentences"] < cfg["constructions"][ctype]["min_sentences"]:
                continue
            # One plan, shared by every model.
            plan = gen.plan_for(w, ctype, generators[0])
            if plan is None:
                continue
            used.add(w["window_id"])
            picked += 1
            example_id = f"{w['window_id']}__{ctype}"
            split = split_map[w["source_id"]]
            human = human_counterpart(w, plan)

            for g in generators:
                if (example_id, g) in done:
                    print(f"  cached {example_id} / {g}")
                    continue
                print(f"  generating {example_id} / {g}", flush=True)
                rec = gen.generate_record(w, ctype, g, split, plan=plan)
                ai = []
                if rec is not None:
                    co = rec["cleaned_output"]
                    ai = [co] if isinstance(co, str) else list(co)
                rows.append({
                    "example_id": example_id,
                    "window_id": w["window_id"],
                    "source_id": w["source_id"],
                    "title": w["title"],
                    "url": w.get("url"),
                    "construction_type": ctype,
                    "split": split,
                    "generator": g,
                    "ok": rec is not None,
                    "human_span": " ".join(human),
                    "human_sentences": human,
                    "ai_text": " ".join(ai),
                    "ai_spans": ai,
                    "spans": plan.get("spans"),
                    "boundary_index": plan.get("boundary_index"),
                    "target_sentence_count": plan["target_sentence_count"],
                    "target_word_count": plan["target_word_count"],
                    "retry_count": rec.get("retry_count") if rec else None,
                    "labels": rec.get("labels") if rec else None,
                    "full_text": rec.get("text") if rec else None,
                })
                write_jsonl(CACHE, rows)

    print(f"\ncomparison spend: ${client.total_cost:.4f}  "
          f"calls={client.n_calls}")
    client.close()
    return rows


# ------------------------------------------------------------- rendering ---

def _cell(text: str, limit: int | None = None) -> str:
    """Make text safe for a markdown table cell."""
    t = (text or "").replace("|", "\\|").replace("\n", " ").strip()
    if limit and len(t) > limit:
        t = t[:limit].rstrip() + " …"
    return t or "_(failed validation)_"


TYPE_LABEL = {
    "type1_single_boundary": "Type 1 · continuation",
    "type2_single_internal_segment": "Type 2 · one span",
    "type3_multiple_internal_segments": "Type 3 · multi-span",
}


def render(cfg: dict, rows: list[dict], generators: list[str]) -> str:
    by_example: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_example.setdefault(r["example_id"], {})[r["generator"]] = r

    # Order: construction type, then complete examples first, so the tables
    # lead with rows where all three models can actually be compared.
    type_rank = {t: i for i, t in enumerate(ALL_TYPES)}

    def sort_key(ex: str):
        m = by_example[ex]
        any_row = next(iter(m.values()))
        complete = all(m.get(g, {}).get("ok") for g in generators)
        return (type_rank[any_row["construction_type"]], not complete, ex)

    order = sorted(by_example, key=sort_key)

    L: list[str] = []
    L.append("# Model comparison — same passage, same span, three models")
    L.append("")
    L.append("Every model below was given the **same source window, the same "
             "construction type and the same span plan**, so any difference in "
             "the output is attributable to the model alone.")
    L.append("")
    L.append("The pilot itself cannot show this: it assigns different windows "
             "to each generator, and the held-out generator only ever sees "
             "dev/test. These examples were generated specifically for "
             "comparison and are drawn from splits every generator is allowed "
             "to write.")
    L.append("")
    L.append("**How to read the Type 1 human column.** Type 1 discards the "
             "true human remainder and asks the model to write a new "
             "continuation, so the human text is *not* a reference answer — "
             "it is what actually followed in Wikipedia. For Types 2 and 3 the "
             "human column is exactly the span the model was told to rewrite, "
             "so it is a direct like-for-like comparison.")
    L.append("")

    # ------------------------------------------------- summary metrics -----
    L.append("## Summary — length fidelity against the human span")
    L.append("")
    L.append("How closely each model matched the length of the human text it "
             "replaced. `Δ words` is the mean signed difference and `|Δ| words` "
             "the mean absolute difference, over examples where the model "
             "produced validated output.")
    L.append("")
    L.append("| model | produced | mean Δ words | mean \\|Δ\\| words | "
             "mean Δ sentences | mean retries |")
    L.append("|---|---|---|---|---|---|")
    for g in generators:
        got = [r for r in rows if r["generator"] == g and r["ok"]]
        tot = len([r for r in rows if r["generator"] == g])
        if not got:
            L.append(f"| {g} | 0/{tot} | — | — | — | — |")
            continue
        dw, adw, ds, rt = [], [], [], []
        for r in got:
            hw = word_count(r["human_span"])
            nw = word_count(r["ai_text"])
            hs = len(segment_sentences(r["human_span"]))
            ns = len(segment_sentences(r["ai_text"]))
            dw.append(nw - hw)
            adw.append(abs(nw - hw))
            ds.append(ns - hs)
            rt.append(r.get("retry_count") or 0)
        mean = lambda a: sum(a) / len(a)  # noqa: E731
        L.append(f"| {g} | {len(got)}/{tot} | {mean(dw):+.1f} | "
                 f"{mean(adw):.1f} | {mean(ds):+.2f} | {mean(rt):.2f} |")
    L.append("")

    # ---------------------------------------------------------- index ------
    L.append("## Examples")
    L.append("")
    L.append("`complete` marks examples where all three models produced "
             "validated output — those are the strict like-for-like rows.")
    L.append("")
    L.append("| # | example | type | split | article | complete |")
    L.append("|---|---|---|---|---|---|")
    for i, ex in enumerate(order, 1):
        m = by_example[ex]
        any_row = next(iter(m.values()))
        complete = all(m.get(g, {}).get("ok") for g in generators)
        L.append(f"| {i} | `{ex}` | {TYPE_LABEL[any_row['construction_type']]} "
                 f"| {any_row['split']} | {any_row['title']} "
                 f"| {'yes' if complete else 'no'} |")
    L.append("")

    # ------------------------------------------------- at-a-glance table ---
    L.append("## At a glance — same example across models")
    L.append("")
    L.append("Text truncated to keep the table readable; full text in the "
             "per-example sections below.")
    L.append("")
    header = "| # | type | human original | " + " | ".join(generators) + " |"
    L.append(header)
    L.append("|---|---|---|" + "---|" * len(generators))
    for i, ex in enumerate(order, 1):
        rowmap = by_example[ex]
        any_row = next(iter(rowmap.values()))
        cells = [_cell(rowmap[g]["ai_text"], 180) if g in rowmap else "—"
                 for g in generators]
        L.append(f"| {i} | {TYPE_LABEL[any_row['construction_type']]} | "
                 + _cell(any_row["human_span"], 180) + " | "
                 + " | ".join(cells) + " |")
    L.append("")

    # ------------------------------------------------ per-example detail ---
    L.append("## Full comparisons")
    L.append("")
    for i, ex in enumerate(order, 1):
        rowmap = by_example[ex]
        any_row = next(iter(rowmap.values()))
        L.append(f"### Example {i} — {TYPE_LABEL[any_row['construction_type']]}")
        L.append("")
        L.append(f"**Article:** {any_row['title']} · source `"
                 f"{any_row['source_id']}` · split `{any_row['split']}`")
        if any_row.get("spans"):
            L.append(f"**Replaced sentence span(s):** `{any_row['spans']}`")
        if any_row.get("boundary_index") is not None:
            L.append(f"**Boundary after sentence:** "
                     f"`{any_row['boundary_index']}`")
        L.append(f"**Target:** {any_row['target_sentence_count']} sentences, "
                 f"{any_row['target_word_count']} words")
        L.append("")

        hs = len(segment_sentences(any_row["human_span"]))
        hw = word_count(any_row["human_span"])
        L.append("| source | text | sents | words | Δ words vs human | retries |")
        L.append("|---|---|---|---|---|---|")
        L.append(f"| **HUMAN** (original) | {_cell(any_row['human_span'])} "
                 f"| {hs} | {hw} | — | — |")
        for g in generators:
            r = rowmap.get(g)
            if r is None:
                L.append(f"| {g} | — | | | | |")
                continue
            if not r["ok"]:
                L.append(f"| {g} | _rejected: no output passed validation "
                         f"in 3 attempts_ | | | | |")
                continue
            ns = len(segment_sentences(r["ai_text"]))
            nw = word_count(r["ai_text"])
            delta = f"{nw - hw:+d} ({(nw-hw)/hw:+.0%})" if hw else "—"
            L.append(f"| {g} | {_cell(r['ai_text'])} | {ns} | {nw} | "
                     f"{delta} | {r.get('retry_count', 0)} |")
        L.append("")

        # Sentence-level label view for one model, to show the boundary.
        shown = next((rowmap[g] for g in generators
                      if g in rowmap and rowmap[g]["ok"]
                      and rowmap[g].get("labels")), None)
        if shown:
            L.append(f"<details><summary>Assembled document with labels "
                     f"({shown['generator']})</summary>")
            L.append("")
            L.append("| # | label | sentence |")
            L.append("|---|---|---|")
            sents = segment_sentences(shown["full_text"])
            for idx, (s, lab) in enumerate(zip(sents, shown["labels"])):
                L.append(f"| {idx} | {'**AI**' if lab else 'human'} | "
                         f"{_cell(s)} |")
            L.append("")
            L.append("</details>")
            L.append("")

    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--per-type", type=int, default=3)
    ap.add_argument("--generators", nargs="*", default=None)
    ap.add_argument("--render-only", action="store_true",
                    help="re-render markdown from cache, no API calls")
    a = ap.parse_args()
    cfg = load_config(a.config)
    gens = a.generators or list(cfg["generators"].keys())

    force_utf8_stdout()
    if a.render_only:
        rows = list(read_jsonl(CACHE))
        if not rows:
            raise SystemExit(f"No cache at {CACHE}; run without --render-only.")
    else:
        rows = collect(cfg, a.per_type, gens)

    out = Path(cfg["paths"]["reports_dir"]) / "generation" / "model_comparison.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(cfg, rows, gens), encoding="utf-8")
    print(f"wrote -> {out}")


if __name__ == "__main__":
    main()
