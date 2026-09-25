# Sinhala Human–AI Authorship Boundary Detection

A dataset and a set of detectors for **finding where authorship changes from
human to machine inside a Sinhala document**. Each document mixes human
sentences with sentences written by a large language model. Every sentence is
labelled `0` (human) or `1` (machine), and a *boundary* is any point where the
label changes.

The project covers the whole path:

- **Dataset construction** from Sinhala Wikipedia: cleaning, sentence
  segmentation, three construction types, three generators (one held out),
  automatic validation, and leak-free splits.
- **Detectors**: trivial and positional baselines, a character n-gram linear
  model, masked-LM likelihood features, and a fine-tuned XLM-RoBERTa sentence
  tagger.
- **Counterfactual twin training and evaluation**: every mixed document has an
  exactly aligned all-human counterpart, used to measure (and reduce) how often
  a detector invents machine text in purely human documents.

A full walkthrough of every design decision and result is in
[docs/COMPLETE_EXPLAINER.md](docs/COMPLETE_EXPLAINER.md).

## Key results

Test set, 918 documents. Exact-boundary F1 is the primary metric: it is the
only one that the trivial baselines cannot game.

| model | sentence F1 (machine) | exact-boundary F1 | held-out generator F1 |
|---|---|---|---|
| position only (reads no text) | 0.567 | 0.284 | 0.550 |
| linear, character n-grams + context | 0.678 | 0.438 | 0.571 |
| linear + likelihood features | 0.704 | 0.450 | 0.615 |
| **XLM-RoBERTa tagger** | **0.802** | **0.603** | **0.742** |

On the **all-human twins** of the test documents. The fine-tuned model is
compared with a control that got the same extra training without twins
(mean ± std over 3 seeds; the from-scratch model is a single run):

| | XLM-R + 3 epochs (control) | + 3 epochs **with twins** | twins, from scratch |
|---|---|---|---|
| human documents with ≥1 sentence flagged | 95.4% ± 1.3 | **58.3% ± 1.1** | 49.0% |
| lowest false-alarm rate reachable at any threshold | 93% ± 2 | 48% ± 6 | **24%** |
| exact-boundary F1, mixed documents only (Viterbi) | **0.607 ± 0.012** | 0.596 ± 0.002 | 0.580 |
| exact-boundary F1, mixed documents + twins | 0.430 ± 0.010 | **0.472 ± 0.004** | 0.464 |

Twin fine-tuning cuts false alarms by 37 points on every seed, for about one
point of mixed-document accuracy. See
[reports/detection/twin_ft_seeds.md](reports/detection/twin_ft_seeds.md),
[reports/detection/twin_tradeoff.md](reports/detection/twin_tradeoff.md) and
Part 4 of the explainer.

## Dataset

**4,244 documents · 39,358 labelled sentences · 2,630 source articles ·
36.4% machine sentences.**

The human text comes from a 2022 Sinhala Wikipedia snapshot, taken before the
public release of ChatGPT.

| construction | what the model writes | boundaries | documents |
|---|---|---|---|
| Type 1, continuation | a new ending after a human prefix (split at 20–80%) | 1 | 1,675 |
| Type 2, single span | a rewrite of one internal 1–3 sentence span | 2 | 1,367 |
| Type 3, multiple spans | independent rewrites of 2–3 internal 1–2 sentence spans | up to 6 | 1,202 |

| generator | role | train | dev | test |
|---|---|---|---|---|
| DeepSeek V3 | seen | 1,186 | 273 | 302 |
| GPT-4o (2024-11-20) | seen | 1,304 | 283 | 311 |
| Gemini 2.5 Pro | held out | 0 | 280 | 305 |

Splits are assigned over **source articles before any generation**, so no human
passage appears in more than one split. Machine text always *replaces* human
text in place, never inserted, so the surrounding text stays genuine. Seven
automatic checks gate every generation (sentence count, length ±25%, no
preamble, no markdown, Sinhala script, not a copy, numbers unchanged). Failures
are regenerated, never hand-edited.

Generating the full corpus took 11,587 API calls and cost $26.73.

## Repository layout

```
config.yaml                 every tunable setting: models, sizes, thresholds, paths
prompts/task2_prompts.yaml  Sinhala prompt templates
src/
  inspect_data.py           profile the source parquet
  clean.py                  page-type filtering, markup stripping, prose gate
  segment.py                the single deterministic Sinhala sentence segmenter
  window.py                 contiguous 6-12 sentence windows
  split.py                  source-level 70/15/15 split
  openrouter.py             API client: retries, backoff, cost logging
  generate.py               Type 1/2/3 construction and orchestration
  validate.py               the seven validation checks
  build_dataset.py          assemble and verify generated/combined.jsonl
  audit.py                  dataset balance and boundary-position audit
  pilot_report.py           generator pilot report
  compare_models.py         matched cross-generator comparison
  probe_models.py           try arbitrary generator slugs
  make_figures.py           result figures
  make_abd_figures.py       dataset figures
  serve.py                  local web app for trying the detector
  detect/
    data.py                 dataset loading, counterfactual twins
    metrics.py              sentence, boundary and twin metrics
    models.py               baselines and the linear detector
    likelihood.py           masked-LM likelihood features
    transformer.py          XLM-R sentence tagger, twin training
    run_experiments.py      baselines and linear models
    run_transformer.py      XLM-R training and evaluation
    twin_tradeoff.py        threshold sweep across saved models
    summarize_seeds.py      mean, spread and paired differences across seeds
tests/                      unit tests
reports/
  dataset/                  dataset audit
  generation/               pilot, fluency, cross-model comparison, evidence
  detection/                all detector results (Markdown + JSON)
docs/COMPLETE_EXPLAINER.md  full technical walkthrough
```

Not tracked (regenerated locally): `wikipedia.parquet`, `sources/`,
`generated/`, `logs/`, `models/`, `cache/`, `reports/figures/`.

## Setup

Python 3.11, and a CUDA GPU for the transformer (6 GB is enough).

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt     # Windows
# source .venv/bin/activate && pip install -r requirements.txt   # Linux/macOS
```

`torch` is pinned to a CUDA 12.4 build. Install it from the PyTorch index if
pip cannot find it:
`pip install torch --index-url https://download.pytorch.org/whl/cu124`.

Generation needs an [OpenRouter](https://openrouter.ai) key, read from the
`OPENROUTER_API_KEY` environment variable or a gitignored `.env` file:

```
OPENROUTER_API_KEY=sk-or-v1-...
```

The pipeline stops with an error if the key is missing.

On Windows, set `PYTHONIOENCODING=utf-8` so Sinhala prints correctly.

## Usage

### 1. Build the dataset

Place `wikipedia.parquet` (Sinhala Wikipedia export) in the repository root.

```bash
python src/inspect_data.py                # profile the source
python src/clean.py                       # -> sources/human_sources.jsonl
python src/window.py                      # -> sources/windows.jsonl
python src/split.py                       # -> splits/*.txt
python src/generate.py --mode pilot       # pilot run, reviewed before the full run
python src/generate.py --mode full        # add --fraction 0.2 for a nested partial run
python src/build_dataset.py --strict      # -> generated/combined.jsonl
python src/audit.py                       # -> reports/dataset/audit.md
```

`python src/generate.py --mode pilot --dry-run` rehearses the whole path with no
API calls. Generation is resumable: records are keyed by a deterministic
`record_id`, and completed ones are skipped on re-run.

### 2. Train and evaluate detectors

```bash
python src/detect/likelihood.py --build          # likelihood features -> cache/
python src/detect/run_experiments.py             # baselines + linear -> reports/detection/linear.*
python src/detect/run_transformer.py --epochs 8  # XLM-R -> reports/detection/xlmr.*, models/xlmr_tagger
```

The protocol is fixed: train on `train` (seen generators only), select epochs
and thresholds on `dev` **restricted to seen generators**, and report on `test`
overall, seen, and held-out.

### 3. Counterfactual twin training

```bash
# twins with paired margin and consistency terms, 2-epoch warm-up
python src/detect/run_transformer.py --epochs 8 --twins --twin-warmup 2 --tag twin_warm

# ablation: twins as plain extra human documents
python src/detect/run_transformer.py --epochs 8 --twins --tag twin_plain \
    --margin-weight 0 --consistency-weight 0

# fine-tune an already trained tagger with twins
python src/detect/run_transformer.py --init-from models/xlmr_tagger \
    --epochs 3 --lr 1e-5 --twins --tag twin_ft

# re-score a saved model (adds twin metrics) without retraining
python src/detect/run_transformer.py --eval-only models/xlmr_tagger --tag xlmr_rescored

# compare saved models across all decision thresholds
python src/detect/twin_tradeoff.py base=models/xlmr_tagger \
    twin_plain=models/xlmr_twin_plain twin_warm=models/xlmr_twin_warm twin_ft=models/xlmr_twin_ft
```

`--tag NAME` writes `reports/detection/NAME.{md,json}` and saves the model to
`models/xlmr_NAME`. `--seed N` overrides the config seed, and `--limit N` runs
a quick smoke test on N documents.

To compare two methods across seeds (runs paired by seed):

```bash
python src/detect/summarize_seeds.py twin_ft=twin_ft,twin_ft_s43,twin_ft_s44 \n    control=xlmr_ft,xlmr_ft_s43,xlmr_ft_s44 --out reports/detection/twin_ft_seeds
```

### 4. Try the detector

```bash
python src/serve.py        # http://127.0.0.1:5000
```

Paste Sinhala text and each sentence is labelled human or machine. The model
runs locally, so text never leaves the machine.

### Figures and tests

```bash
python src/make_figures.py && python src/make_abd_figures.py   # -> reports/figures/
python -m pytest tests/ -q
```

## Record format

One JSON object per document in `generated/combined.jsonl`:

| field | meaning |
|---|---|
| `record_id` | `{window_id}__{type}__{generator}`, stable across runs |
| `source_id`, `window_id` | originating article and window |
| `generator`, `generator_role` | model that wrote the machine text; `seen` or `held_out` |
| `construction_type` | `type1_single_boundary`, `type2_single_internal_segment`, `type3_multiple_internal_segments` |
| `split` | `train`, `dev` or `test`, inherited from the source article |
| `sentences[]`, `labels[]` | sentences and parallel `0`/`1` labels |
| `boundaries[]` | indices where the label changes (index *i* = between *i−1* and *i*) |
| `spans` | replaced sentence ranges (Types 2 and 3) |
| `raw_output`, `cleaned_output` | model output before and after packaging removal |
| `prompt_id`, `temperature`, `top_p`, `retry_count` | generation settings |
| `validation_passed`, `validation_errors[]` | validator verdict |

## Reports

| file | contents |
|---|---|
| [reports/dataset/audit.md](reports/dataset/audit.md) | class balance and boundary-position audit |
| [reports/generation/pilot_report.md](reports/generation/pilot_report.md) | generator pilot and cost estimate |
| [reports/generation/fluency_assessment.md](reports/generation/fluency_assessment.md) | per-generator fluency review, including the rejected Mistral Nemo |
| [reports/generation/model_comparison.md](reports/generation/model_comparison.md) | the three generators on identical inputs |
| [reports/detection/linear.md](reports/detection/linear.md) | baselines, linear and likelihood models |
| [reports/detection/xlmr.md](reports/detection/xlmr.md) | XLM-R tagger (headline model) |
| [reports/detection/xlmr_likelihood.md](reports/detection/xlmr_likelihood.md) | XLM-R with likelihood features (no gain) |
| [reports/detection/xlmr_rescored.md](reports/detection/xlmr_rescored.md) | XLM-R re-scored with twin metrics |
| [reports/detection/twin_plain.md](reports/detection/twin_plain.md) | twins as extra human documents |
| [reports/detection/twin_warm.md](reports/detection/twin_warm.md) | twin training with paired terms and warm-up |
| [reports/detection/twin_ft.md](reports/detection/twin_ft.md) | the trained tagger fine-tuned with twins |
| [reports/detection/xlmr_ft.md](reports/detection/xlmr_ft.md) | control: the same fine-tuning without twins |
| [reports/detection/twin_ft_seeds.md](reports/detection/twin_ft_seeds.md) | twin fine-tuning vs control over 3 seeds (runs `*_s43`, `*_s44`) |
| [reports/detection/twin_no_warmup.md](reports/detection/twin_no_warmup.md) | paired terms from step 0: collapsed (kept as a negative result) |
| [reports/detection/twin_tradeoff.md](reports/detection/twin_tradeoff.md) | all models across all decision thresholds |

## Limitations

Wikipedia is the only domain, and generalisation is measured against a single
held-out generator. Human sentences carry typographical noise that generator
output lacks, so part of any detector's score may come from formatting.
Twin fine-tuning and its control have three seeds; other transformer results
are single runs. No human-annotation ceiling has been established. Part 5 of
the explainer discusses each point.
