# Sinhala Human–AI Boundary-Detection Dataset Pipeline

Turns human-written Sinhala Wikipedia articles into a labelled human–AI
boundary-detection dataset. Every document mixes human and AI sentences; each
sentence carries a `0` (human) / `1` (AI) label, and the label-change indices
are the boundaries a model must learn to find.

## Setup

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt  # POSIX
```

The OpenRouter key is read from the `OPENROUTER_API_KEY` environment variable,
or from a `.env` file at the repo root (gitignored):

```
OPENROUTER_API_KEY=sk-or-v1-...
```

If the key is missing the pipeline fails loudly rather than silently skipping
generation. The key is never hardcoded.

## Run order

Each stage is independent and re-runnable. Prefix with `PYTHONIOENCODING=utf-8`
on Windows so Sinhala prints correctly.

| # | Command | What it does |
|---|---|---|
| 1 | `python src/inspect_data.py` | Profile `wikipedia.parquet`: schema, dtypes, samples, markup probes |
| 2 | `python src/clean.py` | Namespace + page-type filter, markup strip, prose gate → `sources/human_sources.jsonl` |
| 3 | `python -m pytest tests/ -q` | Unit tests for clean, segment, validate, generate |
| 4 | `python src/window.py` | Contiguous 6–12 sentence windows → `sources/windows.jsonl` |
| 5 | `python src/split.py` | 70/15/15 split over **source ids** → `splits/*.txt` |
| 6 | `python src/generate.py --mode pilot` | Pilot generation (gated — see below) |
| 7 | `python src/build_dataset.py --strict` | Assemble + verify → `generated/combined.jsonl` |
| 8 | `python src/audit.py` | Balance and boundary-position report → `reports/audit.md` |

Rehearse the whole path with **zero API calls and zero cost**:

```bash
python src/generate.py --mode pilot --dry-run
```

Other flags: `--generators deepseek_v3 mistral_nemo`, `--limit N`,
`--config other.yaml`.

## The mandatory pilot gate

`--mode full` must not be run until the pilot has been reviewed and approved.
The pilot produces `reports/pilot_report.md` with per-generator validation
pass-rates, sample generations, a full-run cost estimate, and a Sinhala-fluency
assessment for each generator.

```bash
python src/generate.py --mode full          # only after approval
```

## Design

**Cleaning source.** Cleaning runs on `raw_mediawiki`, not the shipped `text`
column. `text` still leaks `Category:` lines, `{{#ifexpr}}` residue and
`__NOTOC__` markers, and it flattens list bullets into leading spaces, which
corrupts sentence segmentation.

**One segmenter everywhere.** `src/segment.py::segment_sentences` is used for
source prep, generation length targets, labelling and evaluation, so a
"sentence index" means the same thing at every stage. It splits on `.`, `?`,
`!`, `෴` and `…` with guards for abbreviations, decimals, Latin initials,
ellipses and quoted sentences. Note that a lone Sinhala letter before a full
stop is the sentence-final particle (`… සම්මාන ය.`), *not* an initial —
treating it as one silently merges sentences.

**Split before generation.** The 70/15/15 partition is over *source article
ids*, written before any API call. Every derivative — all construction types,
all generators — inherits its source's partition, so no sentence from a train
article can appear in dev or test. Seen generators (DeepSeek V3, Mistral Nemo)
write to all three splits; the held-out generator (Gemini 2.5 Pro) writes only
to dev and test. `build_dataset.py` re-checks both rules.

**Two primitives only.**

- *Continuation* (Type 1): a human prefix is kept, the true human remainder is
  discarded, and the model writes a new continuation. One boundary.
- *Context-aware span replacement* (Types 2 and 3): an existing human span is
  handed to the model with its left/right context and rewritten in place.
  Never insertion of invented sentences — the AI text occupies exactly the
  position the human text did. Two boundaries per span.

For Type 3, each span is generated **independently from the original human
document** — span 2 is never conditioned on span 1's output — and all spans in
a document use the same generator.

**Validation.** Seven automatic checks (sentence count, length ±25%, no
preamble, no markdown, predominantly Sinhala script, not a copy of the
original, numbers unchanged). Failures are retried up to 3 times from scratch;
model output is **never hand-repaired**. Every rejection is logged with reasons
to `logs/rejected.jsonl`.

**Reasoning models.** For any reasoning-capable generator the client sends
`reasoning: {exclude: true}` *and* reads only `message.content`, never
`message.reasoning`, so a reasoning trace cannot reach the stored text even if
a provider ignores the flag.

**Resumability.** Records are keyed by a deterministic `record_id`
(`{window_id}__{type}__{generator}`). A re-run skips ids already present in
`generated/*.jsonl`, so an interrupted run continues rather than regenerating.

## Layout

```
config.yaml                    all knobs: models, sizes, ranges, seeds, paths
prompts/task2_prompts.yaml     Sinhala prompt templates per domain x primitive
src/inspect_data.py            load + profile the parquet
src/clean.py                   page-type filter + markup strip + prose gate
src/segment.py                 THE deterministic Sinhala segmenter
src/window.py                  contiguous passage windowing
src/split.py                   source-level 70/15/15 split
src/openrouter.py              API client: retries, backoff, cost logging
src/generate.py                Type 1/2/3 construction + orchestration
src/validate.py                the 7 validation checks
src/build_dataset.py           assemble records, verify integrity
src/audit.py                   balance + boundary-position plots
sources/human_sources.jsonl    cleaned human articles
sources/windows.jsonl          windowed passages
generated/*.jsonl              per-type records + combined.jsonl
logs/                          generations.jsonl, rejected.jsonl, cost.jsonl
splits/                        train/dev/test source id lists
reports/                       pilot_report.md, audit.md, plots
tests/                         unit tests
```

## Record schema

One JSON object per generated document in `generated/combined.jsonl`:

| field | meaning |
|---|---|
| `record_id` | `{window_id}__{type}__{generator}`, stable across runs |
| `source_id` | originating Wikipedia page id |
| `domain` | `wikipedia` |
| `generator` / `generator_slug` / `generator_role` | which model wrote the AI spans; `seen` or `held_out` |
| `construction_type` | `type1_single_boundary` / `type2_single_internal_segment` / `type3_multiple_internal_segments` |
| `split` | `train` / `dev` / `test`, inherited from `source_id` |
| `text` | the assembled document (`" ".join(sentences)`) |
| `sentences[]` | sentence list, from the shared segmenter |
| `labels[]` | `0` human / `1` AI, parallel to `sentences` |
| `boundaries[]` | indices where the label changes |
| `boundary_position_normalized` | first boundary / number of sentences |
| `spans` | replaced sentence ranges (Types 2/3) |
| `prompt_id` | template that produced the AI text |
| `raw_output` / `cleaned_output` | model output before/after packaging strip |
| `temperature` / `top_p` / `retry_count` | generation settings and retries used |
| `validation_passed` / `validation_errors[]` | validator verdict |

## Configuration

Everything tunable lives in `config.yaml` — model slugs and prices, sampling
bands, span sizes, prose-gate thresholds, validation tolerances, pilot and
full-run sizes, concurrency and backoff. There are no magic numbers in the
code. Model slugs and prices are re-verified against OpenRouter at the start of
every run, and price drift is reported.
