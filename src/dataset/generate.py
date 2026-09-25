"""Type 1 / 2 / 3 construction and generation orchestration.

Two primitives only:
  continuation                (Type 1)
  context-aware span replacement (Types 2 and 3)

Span replacement never inserts invented sentences: an existing human span is
handed to the model and rewritten in place, so the AI text occupies exactly the
position the human text did.

Type 3 generates each span INDEPENDENTLY from the original human document -
span 2 is never conditioned on span 1's output - then assembles. All spans in a
document use the same generator.

Everything is resumable: a record whose record_id already exists in the output
file is skipped, so a re-run continues rather than regenerates.
"""
from __future__ import annotations

import argparse
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from openrouter import (  # noqa: E402
    InsufficientCredits,
    MissingAPIKey,
    OpenRouterClient,
    OpenRouterError,
)
from segment import segment_sentences, word_count  # noqa: E402
from split import load_split_map  # noqa: E402
from utils import (  # noqa: E402
    append_jsonl,
    force_utf8_stdout,
    load_config,
    read_jsonl,
)
from validate import Validator, clean_output  # noqa: E402

TYPE1 = "type1_single_boundary"
TYPE2 = "type2_single_internal_segment"
TYPE3 = "type3_multiple_internal_segments"
ALL_TYPES = (TYPE1, TYPE2, TYPE3)

OUT_FILES = {
    TYPE1: "type1_single_boundary.jsonl",
    TYPE2: "type2_single_internal_segment.jsonl",
    TYPE3: "type3_multiple_internal_segments.jsonl",
}


# =========================================================== planning ======
# A plan says WHICH sentences become AI, before any API call is made. Plans are
# pure and deterministic given (window, rng), which keeps generation auditable.

def plan_type1(window: dict, cfg: dict, rng: random.Random) -> dict | None:
    """Human prefix -> AI continuation. One boundary."""
    c = cfg["constructions"][TYPE1]
    sents = window["sentences"]
    n = len(sents)
    if n < c["min_sentences"]:
        return None

    # Sentence-aligned split sampled in [low, high] of the passage, so the
    # boundary is not always near the middle or the start.
    lo_idx = max(c["min_human_sentences"], int(round(n * c["boundary_low"])))
    hi_idx = min(n - c["min_ai_sentences"], int(round(n * c["boundary_high"])))
    if lo_idx > hi_idx:
        return None
    b = rng.randint(lo_idx, hi_idx)

    human = sents[:b]
    removed = sents[b:]          # discarded; the model must invent a new tail
    return {
        "construction_type": TYPE1,
        "primitive": "single_boundary_prefix",
        "boundary_index": b,
        "human_prefix": human,
        "removed": removed,
        "target_sentence_count": len(removed),
        "target_word_count": word_count(" ".join(removed)),
    }


def plan_type2(window: dict, cfg: dict, rng: random.Random) -> dict | None:
    """One internal span replaced. Two boundaries."""
    c = cfg["constructions"][TYPE2]
    sents = window["sentences"]
    n = len(sents)
    if n < c["min_sentences"]:
        return None

    ctx = c["min_context_sentences"]
    candidates: list[tuple[int, int]] = []
    for size in range(c["span_min_sentences"], c["span_max_sentences"] + 1):
        ratio = size / n
        if not (c["ai_ratio_min"] <= ratio <= c["ai_ratio_max"]):
            continue
        # Span must not touch sentence 0 or the final sentence.
        for start in range(1, n - size):
            end = start + size
            if end > n - 1:
                continue
            if start >= ctx and (n - end) >= ctx:
                candidates.append((start, end))

    if not candidates:
        # Relax the preferred context margin, keep the hard start/end rule.
        for size in range(c["span_min_sentences"], c["span_max_sentences"] + 1):
            ratio = size / n
            if not (c["ai_ratio_min"] <= ratio <= c["ai_ratio_max"]):
                continue
            for start in range(1, n - size):
                if start + size <= n - 1:
                    candidates.append((start, start + size))
    if not candidates:
        return None

    start, end = rng.choice(candidates)
    target = sents[start:end]
    return {
        "construction_type": TYPE2,
        "primitive": "span_replacement",
        "spans": [(start, end)],
        "targets": [target],
        "target_sentence_count": len(target),
        "target_word_count": word_count(" ".join(target)),
    }


def plan_type3(window: dict, cfg: dict, rng: random.Random) -> dict | None:
    """2-3 non-adjacent internal spans. Each generated independently."""
    c = cfg["constructions"][TYPE3]
    sents = window["sentences"]
    n = len(sents)
    if n < c["min_sentences"]:
        return None

    gap = c["min_gap_sentences"]
    n_spans = rng.randint(c["num_spans_min"], c["num_spans_max"])

    for _ in range(60):                      # rejection sampling
        spans: list[tuple[int, int]] = []
        cursor = 1                           # never start at sentence 0
        ok = True
        for k in range(n_spans):
            size = rng.randint(c["span_min_sentences"], c["span_max_sentences"])
            remaining_spans = n_spans - k - 1
            # Leave room for later spans, their gaps, and the final sentence.
            reserve = remaining_spans * (c["span_min_sentences"] + gap) + 1
            latest = n - reserve - size
            if cursor > latest:
                ok = False
                break
            start = rng.randint(cursor, latest)
            end = start + size
            if end > n - 1:                  # must not cover the last sentence
                ok = False
                break
            spans.append((start, end))
            cursor = end + gap               # >=1 human sentence between spans
        if not ok or len(spans) != n_spans:
            continue
        ai_sents = sum(e - s for s, e in spans)
        if ai_sents / n > c["ai_ratio_max"]:
            continue
        return {
            "construction_type": TYPE3,
            "primitive": "span_replacement",
            "spans": spans,
            "targets": [sents[s:e] for s, e in spans],
            "target_sentence_count": ai_sents,
            "target_word_count": word_count(
                " ".join(" ".join(sents[s:e]) for s, e in spans)),
        }
    return None


PLANNERS: dict[str, Callable[[dict, dict, random.Random], dict | None]] = {
    TYPE1: plan_type1,
    TYPE2: plan_type2,
    TYPE3: plan_type3,
}


# =========================================================== assembly =====

def assemble(window: dict, plan: dict, ai_texts: list[list[str]]) -> dict:
    """Splice AI sentences into the human document and derive labels."""
    sents = window["sentences"]

    if plan["construction_type"] == TYPE1:
        final = list(plan["human_prefix"]) + list(ai_texts[0])
        labels = [0] * len(plan["human_prefix"]) + [1] * len(ai_texts[0])
    else:
        final, labels = [], []
        cursor = 0
        for (s, e), ai in zip(plan["spans"], ai_texts):
            final.extend(sents[cursor:s])
            labels.extend([0] * (s - cursor))
            final.extend(ai)
            labels.extend([1] * len(ai))
            cursor = e
        final.extend(sents[cursor:])
        labels.extend([0] * (len(sents) - cursor))

    boundaries = [i for i in range(1, len(labels)) if labels[i] != labels[i - 1]]
    first_norm = (boundaries[0] / len(labels)) if boundaries else None
    return {
        "text": " ".join(final),
        "sentences": final,
        "labels": labels,
        "boundaries": boundaries,
        "boundary_position_normalized": first_norm,
        "boundary_positions_normalized": [b / len(labels) for b in boundaries],
        "ai_sentence_ratio": (sum(labels) / len(labels)) if labels else 0.0,
    }


# =========================================================== generation ===

class Generator:
    def __init__(self, cfg: dict, client: OpenRouterClient,
                 prompts: dict, dry_run: bool = False) -> None:
        self.cfg = cfg
        self.client = client
        self.prompts = prompts
        self.validator = Validator(cfg)
        self.dry_run = dry_run
        self.logs = Path(cfg["paths"]["logs_dir"])
        self.max_retries = cfg["validate"]["max_retries"]
        self._lock = threading.Lock()
        self.stats: dict[str, int] = {}

    def _bump(self, key: str, n: int = 1) -> None:
        with self._lock:
            self.stats[key] = self.stats.get(key, 0) + n

    def _family(self, domain: str, primitive: str) -> dict:
        try:
            return self.prompts[domain][primitive]
        except KeyError as e:
            raise KeyError(
                f"No prompt family '{primitive}' for domain '{domain}' in "
                f"{self.cfg['paths']['prompts']}") from e

    def _word_bounds(self, target_words: int) -> tuple[int, int]:
        """Acceptable word range, derived from the validator's own tolerance.

        Telling the model the exact window the validator will enforce is what
        stopped it under-generating; deriving it here means the prompt and the
        check can never drift apart.
        """
        tol = self.cfg["validate"]["length_tolerance"]
        return (int(round(target_words * (1 - tol))),
                int(round(target_words * (1 + tol))))

    def _render(self, fam: dict, values: dict[str, Any]) -> tuple[str, str]:
        if "target_word_count" in values:
            lo, hi = self._word_bounds(values["target_word_count"])
            values = {**values, "target_word_min": lo, "target_word_max": hi}
        missing = set(fam["fields"]) - set(values)
        if missing:
            raise KeyError(f"prompt {fam['prompt_id']} missing fields: {missing}")
        return fam["system"], fam["user"].format(**values)

    def _generate_segment(
        self,
        *,
        gen_name: str,
        domain: str,
        primitive: str,
        values: dict[str, Any],
        target_sentence_count: int,
        target_word_count: int,
        original_span: str | None,
        meta: dict,
    ) -> tuple[list[str] | None, str, str, int, list[str]]:
        """Generate one segment with validation retries.

        Returns (sentences|None, raw_output, cleaned_output, retry_count, errors).
        Output is never hand-repaired; a failure is retried from scratch.
        """
        fam = self._family(domain, primitive)
        system, user = self._render(fam, values)

        raw = cleaned = ""
        errors: list[str] = []
        for attempt in range(self.max_retries):
            try:
                comp = self.client.complete(
                    gen_name, system, user,
                    meta={**meta, "attempt": attempt,
                          "target_sentence_count": target_sentence_count,
                          "target_word_count": target_word_count},
                    # Vary the seed per attempt so a retry is a fresh sample.
                    seed=self.cfg["seed"] + attempt * 1000,
                )
            except InsufficientCredits:
                # Not recoverable and not request-specific: stop the run rather
                # than issue thousands more calls that cannot succeed.
                raise
            except OpenRouterError as e:
                errors = [f"api_error: {e}"]
                self._bump("api_errors")
                break

            raw = comp.text
            cleaned = clean_output(raw)
            res = self.validator.validate(
                cleaned,
                target_sentence_count=target_sentence_count,
                target_word_count=target_word_count,
                original_span=original_span,
            )
            if res.passed:
                return (segment_sentences(cleaned, self.cfg), raw, cleaned,
                        attempt, [])
            errors = res.errors
            self._bump("validation_failures")
            for e in res.errors:
                self._bump("fail:" + e.split(":")[0])
            append_jsonl(self.logs / "rejected.jsonl", {
                "ts": time.time(), **meta, "generator": gen_name,
                "attempt": attempt, "errors": res.errors,
                "details": res.details, "cleaned_output": cleaned[:2000],
            })

        return None, raw, cleaned, self.max_retries, errors

    def plan_for(self, window: dict, ctype: str, gen_name: str) -> dict | None:
        """The plan this (window, type, generator) would use. Deterministic."""
        rng = random.Random(
            f"{self.cfg['seed']}:{window['window_id']}:{ctype}:{gen_name}")
        return PLANNERS[ctype](window, self.cfg, rng)

    def generate_record(self, window: dict, ctype: str, gen_name: str,
                        split: str, plan: dict | None = None) -> dict | None:
        """Build one dataset record, or None if it could not be validated.

        `plan` may be supplied to force a specific span selection. Normally the
        plan is derived per generator, but a matched cross-model comparison
        needs every model to rewrite the *same* span.
        """
        cfg = self.cfg
        domain = window["domain"]

        if plan is None:
            plan = self.plan_for(window, ctype, gen_name)
        if plan is None:
            self._bump("skipped_ineligible")
            return None

        gen_cfg = cfg["generators"][gen_name]
        record_id = f"{window['window_id']}__{ctype}__{gen_name}"
        base_meta = {"record_id": record_id, "source_id": window["source_id"],
                     "construction_type": ctype, "split": split}

        fam_name = plan["primitive"]
        fam = self._family(domain, fam_name)

        ai_texts: list[list[str]] = []
        raws: list[str] = []
        cleans: list[str] = []
        retries = 0
        all_errors: list[str] = []

        if ctype == TYPE1:
            values = {
                "title": window["title"],
                "prefix": " ".join(plan["human_prefix"]),
                "target_sentence_count": plan["target_sentence_count"],
                "target_word_count": plan["target_word_count"],
            }
            sents, raw, cleaned, rc, errs = self._generate_segment(
                gen_name=gen_name, domain=domain, primitive=fam_name,
                values=values,
                target_sentence_count=plan["target_sentence_count"],
                target_word_count=plan["target_word_count"],
                original_span=None,          # nothing to copy or preserve
                meta=base_meta,
            )
            if sents is None:
                self._bump("skipped_failed_validation")
                return None
            ai_texts, raws, cleans, retries, all_errors = (
                [sents], [raw], [cleaned], rc, errs)
        else:
            sents_all = window["sentences"]
            for k, ((s, e), target) in enumerate(zip(plan["spans"], plan["targets"])):
                # Context comes from the ORIGINAL human document every time, so
                # span k is never conditioned on span k-1's generated output.
                left = " ".join(sents_all[max(0, s - 3):s])
                right = " ".join(sents_all[e:e + 3])
                target_text = " ".join(target)
                values = {
                    "title": window["title"],
                    "left": left,
                    "target": target_text,
                    "right": right,
                    "target_sentence_count": len(target),
                    "target_word_count": word_count(target_text),
                }
                sents, raw, cleaned, rc, errs = self._generate_segment(
                    gen_name=gen_name, domain=domain, primitive=fam_name,
                    values=values,
                    target_sentence_count=len(target),
                    target_word_count=word_count(target_text),
                    original_span=target_text,
                    meta={**base_meta, "span_index": k, "span": [s, e]},
                )
                if sents is None:
                    self._bump("skipped_failed_validation")
                    return None
                ai_texts.append(sents)
                raws.append(raw)
                cleans.append(cleaned)
                retries += rc
                all_errors.extend(errs)

        asm = assemble(window, plan, ai_texts)
        self._bump("generated")

        return {
            "record_id": record_id,
            "source_id": window["source_id"],
            "window_id": window["window_id"],
            "title": window["title"],
            "url": window.get("url"),
            "domain": domain,
            "generator": gen_name,
            "generator_slug": gen_cfg["slug"],
            "generator_role": gen_cfg["role"],
            "construction_type": ctype,
            "split": split,
            "text": asm["text"],
            "sentences": asm["sentences"],
            "labels": asm["labels"],
            "boundaries": asm["boundaries"],
            "boundary_position_normalized": asm["boundary_position_normalized"],
            "boundary_positions_normalized": asm["boundary_positions_normalized"],
            "ai_sentence_ratio": asm["ai_sentence_ratio"],
            "spans": plan.get("spans"),
            "boundary_index": plan.get("boundary_index"),
            "prompt_id": fam["prompt_id"],
            "raw_output": raws if len(raws) > 1 else raws[0],
            "cleaned_output": cleans if len(cleans) > 1 else cleans[0],
            "temperature": gen_cfg["temperature"],
            "top_p": gen_cfg["top_p"],
            "retry_count": retries,
            "validation_passed": True,
            "validation_errors": all_errors,
            "n_sentences": len(asm["sentences"]),
            "n_words": word_count(asm["text"]),
            "generated_at": time.time(),
        }


# =========================================================== orchestration =

def load_prompts(cfg: dict) -> dict:
    p = Path(cfg["paths"]["prompts"])
    with p.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def completed_ids(cfg: dict) -> set[str]:
    """record_ids already written - the basis of resumability."""
    done: set[str] = set()
    gdir = Path(cfg["paths"]["generated_dir"])
    for fname in OUT_FILES.values():
        for row in read_jsonl(gdir / fname):
            rid = row.get("record_id")
            if rid:
                done.add(rid)
    return done


def build_tasks(cfg: dict, windows: list[dict], split_map: dict[str, str],
                per_generator: int, type_mix: dict[str, int],
                generators: list[str], rng: random.Random,
                splits_filter: list[str] | None = None) -> list[tuple]:
    """(window, ctype, generator, split) tuples respecting the held-out rule."""
    tasks: list[tuple] = []
    for gen_name in generators:
        gcfg = cfg["generators"][gen_name]
        allowed = set(gcfg["splits"])
        if splits_filter:
            allowed &= set(splits_filter)

        pool = [w for w in windows if split_map.get(w["source_id"]) in allowed]
        rng.shuffle(pool)

        used: set[str] = set()
        for ctype, want in type_mix.items():
            cons = cfg["constructions"][ctype]
            if not cons.get("enabled", True):
                continue
            picked = 0
            for w in pool:
                if picked >= want:
                    break
                if w["window_id"] in used:
                    continue
                if w["n_sentences"] < cons["min_sentences"]:
                    continue          # never pad a short article into a type
                used.add(w["window_id"])
                tasks.append((w, ctype, gen_name, split_map[w["source_id"]]))
                picked += 1
            if picked < want:
                print(f"  ! {gen_name}/{ctype}: only {picked}/{want} eligible "
                      f"windows available")
    return tasks


def select_fraction(tasks: list[tuple], fraction: float, seed: int) -> list[tuple]:
    """Take a deterministic, representative prefix of the full task list.

    The full list is always built first and then shuffled with a fixed seed, so
    the order does not depend on `fraction`. A 20%% run is therefore exactly the
    first 20%% of the 100%% run: finishing the rest later resumes and extends the
    same dataset instead of picking different windows and colliding.

    Shuffling before slicing is what keeps the subset representative - the
    unshuffled list is grouped by generator, so a raw prefix would be one model.
    """
    if fraction >= 1.0:
        return tasks
    ordered = list(tasks)
    random.Random(f"{seed}:task-order").shuffle(ordered)
    return ordered[:int(round(len(ordered) * fraction))]


def describe_tasks(tasks: list[tuple], cfg: dict, label: str) -> None:
    from collections import Counter
    by_gen: Counter = Counter(t[2] for t in tasks)
    by_type: Counter = Counter(t[1] for t in tasks)
    by_split: Counter = Counter(t[3] for t in tasks)
    print(f"\n{label}: {len(tasks)} tasks")
    print("  by generator: " + "  ".join(
        f"{k}={v}" for k, v in sorted(by_gen.items())))
    print("  by type:      " + "  ".join(
        f"{k.replace('_single','').replace('_multiple','')}={v}"
        for k, v in sorted(by_type.items())))
    print("  by split:     " + "  ".join(
        f"{k}={by_split.get(k,0)}" for k in ("train", "dev", "test")))


def run(cfg: dict, *, mode: str = "pilot", dry_run: bool = False,
        generators: list[str] | None = None, limit: int | None = None,
        fraction: float = 1.0) -> dict:
    force_utf8_stdout()
    t0 = time.time()

    windows_path = Path(cfg["paths"]["sources"]).parent / "windows.jsonl"
    windows = list(read_jsonl(windows_path))
    if not windows:
        raise SystemExit(f"No windows at {windows_path}; run src/window.py first.")
    split_map = load_split_map(cfg)

    gen_names = generators or list(cfg["generators"].keys())
    if mode == "pilot":
        per_gen = cfg["pilot"]["docs_per_generator"]
        type_mix = dict(cfg["pilot"]["type_mix"])
        splits_filter = cfg["pilot"].get("splits")
    else:
        per_gen_map = cfg["full_run"]["docs_per_generator"]
        ratio = cfg["full_run"]["type_mix_ratio"]
        per_gen = None
        type_mix = None
        splits_filter = None

    prompts = load_prompts(cfg)
    try:
        client = OpenRouterClient(cfg, dry_run=dry_run)
    except MissingAPIKey as e:
        raise SystemExit(f"\nFATAL: {e}\n")

    print("=" * 70)
    print(f"GENERATION  mode={mode}  dry_run={dry_run}")
    print("=" * 70)
    if not dry_run:
        print("Verifying model slugs and pricing against OpenRouter:")
        client.verify_models()

    rng = random.Random(cfg["seed"])
    tasks: list[tuple] = []
    if mode == "pilot":
        tasks = build_tasks(cfg, windows, split_map, per_gen, type_mix,
                            gen_names, rng, splits_filter)
    else:
        # Always build the FULL list, then slice. Keeps partial runs nested.
        for g in gen_names:
            n = per_gen_map[g]
            mix = {t: int(round(n * r)) for t, r in ratio.items()}
            tasks += build_tasks(cfg, windows, split_map, n, mix, [g], rng, None)
        if fraction < 1.0:
            describe_tasks(tasks, cfg, "FULL plan")
            tasks = select_fraction(tasks, fraction, cfg["seed"])
            describe_tasks(tasks, cfg, f"SELECTED {fraction:.0%} of full plan")

    done = completed_ids(cfg)
    pending = []
    for w, ctype, gname, split in tasks:
        rid = f"{w['window_id']}__{ctype}__{gname}"
        if rid in done:
            continue
        pending.append((w, ctype, gname, split))
    skipped_done = len(tasks) - len(pending)
    if limit:
        pending = pending[:limit]

    print(f"\ntasks planned : {len(tasks)}")
    print(f"already done  : {skipped_done} (resumed, will not regenerate)")
    print(f"to generate   : {len(pending)}\n")
    if not pending:
        print("Nothing to do.")
        return {"generated": 0, **client.summary()}

    gen = Generator(cfg, client, prompts, dry_run=dry_run)
    gdir = Path(cfg["paths"]["generated_dir"])
    gdir.mkdir(parents=True, exist_ok=True)
    write_lock = threading.Lock()
    n_ok = 0

    def work(task):
        w, ctype, gname, split = task
        try:
            return ctype, gen.generate_record(w, ctype, gname, split)
        except Exception as e:                       # never kill the whole run
            gen._bump("unexpected_errors")
            append_jsonl(Path(cfg["paths"]["logs_dir"]) / "rejected.jsonl", {
                "ts": time.time(), "window_id": w["window_id"],
                "construction_type": ctype, "generator": gname,
                "errors": [f"unexpected: {type(e).__name__}: {e}"],
            })
            return ctype, None

    with ThreadPoolExecutor(max_workers=cfg["api"]["concurrency"]) as pool:
        futures = [pool.submit(work, t) for t in pending]
        for i, fut in enumerate(as_completed(futures), 1):
            ctype, rec = fut.result()
            if rec is not None:
                with write_lock:
                    append_jsonl(gdir / OUT_FILES[ctype], rec)
                    append_jsonl(Path(cfg["paths"]["logs_dir"]) / "generations.jsonl",
                                 {"ts": time.time(), "record_id": rec["record_id"],
                                  "generator": rec["generator"],
                                  "construction_type": rec["construction_type"],
                                  "split": rec["split"],
                                  "retry_count": rec["retry_count"]})
                    n_ok += 1
            if i % 10 == 0 or i == len(futures):
                print(f"  [{i}/{len(futures)}] ok={n_ok} "
                      f"cost=${client.total_cost:.4f}", flush=True)

    summary = client.summary()
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("GENERATION SUMMARY")
    print("=" * 70)
    print(f"  records written : {n_ok}/{len(pending)}")
    for k in sorted(gen.stats):
        print(f"  {k:<28s} {gen.stats[k]}")
    print(f"  api calls       : {summary['calls']}  retries={summary['retries']}")
    print(f"  tokens          : in={summary['prompt_tokens']} "
          f"out={summary['completion_tokens']}")
    print(f"  TOTAL COST      : ${summary['total_cost_usd']:.4f}")
    print(f"  elapsed         : {elapsed:.1f}s")
    client.close()
    return {"generated": n_ok, "stats": gen.stats, **summary}


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate boundary-detection data")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--dry-run", action="store_true",
                    help="exercise the full pipeline with zero API calls")
    ap.add_argument("--generators", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fraction", type=float, default=1.0,
                    help="run this fraction of the full plan (e.g. 0.2). The "
                         "subset is a deterministic prefix, so finishing the "
                         "rest later resumes and extends the same dataset.")
    a = ap.parse_args()
    if not 0 < a.fraction <= 1.0:
        raise SystemExit("--fraction must be in (0, 1]")
    run(load_config(a.config), mode=a.mode, dry_run=a.dry_run,
        generators=a.generators, limit=a.limit, fraction=a.fraction)


if __name__ == "__main__":
    main()
