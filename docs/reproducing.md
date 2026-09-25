# Reproducing

[README](../README.md) · [Dataset](dataset.md) · [Detectors](detectors.md) · [Likelihood](likelihood.md) · [Counterfactual twins](counterfactual-twins.md) · [Limitations](limitations.md) · [Reproducing](reproducing.md)

Every number in the README and these pages comes from a result file under
`results/`, and every figure is regenerated from those files by
`docs/scripts/make_figures.py`. This page covers the environment, what is and
is not versioned, the run order, and how long each stage takes.

## Environment

Two machines are supported. Results up to the three-seed runs were produced on
the laptop; the DGX Spark reproduces the baseline to within 0.001 on every
metric (GPU floating-point differences).

| component | laptop | DGX Spark |
|---|---|---|
| Python | 3.11.9 | 3.12.3 |
| PyTorch | 2.6.0 + CUDA 12.4 | 2.14.0 + CUDA 13.0 |
| Transformers | 5.16.1 | 5.16.1 |
| scikit-learn | 1.7.2 | 1.7.2 |
| GPU | NVIDIA RTX 4050 Laptop, 6 GB | NVIDIA GB10, 128 GB unified memory |
| OS | Windows 11 (x86-64) | Ubuntu 24.04 (ARM64) |

Install PyTorch first, as the build that matches the machine's CUDA, then the
rest:

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scriptsctivate
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130   # DGX Spark
# pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124  # x86 + CUDA 12.4
pip install -r requirements.txt
```

On Windows, set `PYTHONIOENCODING=utf-8` so Sinhala prints correctly.

Dataset generation calls three models through [OpenRouter](https://openrouter.ai).
The key is read from `OPENROUTER_API_KEY` or a gitignored `.env` file (see
`.env.example`); the pipeline stops with an error if it is missing. Detection
needs no API access.

## What is versioned

| versioned | not versioned (regenerated or local) |
|---|---|
| code, config, prompts, tests | `wikipedia.parquet`, the source export (173 MB) |
| `splits/`: the train/dev/test source-id lists | `sources/`: cleaned articles and windows |
| `results/`: every report and result JSON | `generated/`: the built dataset |
| `docs/assets/`: the figures | `logs/`: API, cost and rejection logs |
| | `models/`: fine-tuned checkpoints (1.1 GB each) |
| | `cache/`: likelihood features |

The splits are versioned so that anyone rebuilding the dataset gets the same
partition of source articles.

## Run order

### 1. Build the dataset

Place `wikipedia.parquet` (a 2022 Sinhala Wikipedia export) in the repository
root, then:

```bash
python src/dataset/inspect_data.py              # profile the export
python src/dataset/clean.py                     # -> sources/human_sources.jsonl
python src/dataset/window.py                    # -> sources/windows.jsonl
python src/dataset/split.py                     # -> splits/*.txt
python src/dataset/generate.py --mode pilot     # pilot; review results/generation/pilot_report.md
python src/dataset/generate.py --mode full      # --fraction 0.2 runs a nested partial plan
python src/dataset/build_dataset.py --strict    # -> generated/combined.jsonl
python src/dataset/audit.py                     # -> results/dataset/audit.md
```

`python src/dataset/generate.py --mode pilot --dry-run` rehearses the whole path
with no API calls. Generation is resumable: records are keyed by a
deterministic `record_id`, and completed ones are skipped on a re-run.

### 2. Baselines, linear models and likelihood features

```bash
python src/detect/likelihood.py --build         # -> cache/likelihood_features.jsonl
python src/detect/run_experiments.py            # -> results/detection/baselines/linear.*
```

`run_experiments.py` adds the two likelihood rows automatically when the
feature cache exists.

### 3. XLM-R tagger

```bash
python src/detect/run_transformer.py --epochs 8                 # -> results/detection/xlmr/xlmr.*
python src/detect/run_transformer.py --epochs 8 --likelihood \
    --json-out results/detection/xlmr/xlmr_likelihood.json \
    --out results/detection/xlmr/xlmr_likelihood.md --save-to models/xlmr_likelihood
```

The protocol is fixed in the runner: train on `train` (seen generators only),
choose the epoch and decision thresholds on `dev` restricted to seen
generators, report on `test` overall, seen and held out.

### 4. Counterfactual twins

Runs given `--tag NAME` write `results/detection/twins/NAME.{md,json}` and save
the model to `models/xlmr_NAME`.

```bash
# the baseline, re-scored with twin metrics (no training)
python src/detect/run_transformer.py --eval-only models/xlmr_tagger --tag xlmr_rescored

# from scratch: twins as extra human documents, then with the paired terms
python src/detect/run_transformer.py --epochs 8 --twins --tag twin_plain \
    --margin-weight 0 --consistency-weight 0
python src/detect/run_transformer.py --epochs 8 --twins --twin-warmup 2 --tag twin_warm

# fine-tuning the trained tagger with twins, and its control, for seeds 42-44
for s in 42 43 44; do
  python src/detect/run_transformer.py --init-from models/xlmr_tagger --epochs 3 \
      --lr 1e-5 --twins --seed $s --tag twin_ft_s$s
  python src/detect/run_transformer.py --init-from models/xlmr_tagger --epochs 3 \
      --lr 1e-5 --seed $s --tag xlmr_ft_s$s --save-to models/xlmr_ft_s$s
done
```

The committed seed-42 runs are named `twin_ft` and `xlmr_ft` (they predate
`--seed`); seeds 43 and 44 carry the `_s43` and `_s44` suffixes.

```bash
# threshold sweep across saved models, and the three-seed comparison
python src/detect/twin_tradeoff.py base=models/xlmr_tagger \
    twin_plain=models/xlmr_twin_plain twin_warm=models/xlmr_twin_warm \
    twin_ft=models/xlmr_twin_ft base_ft=models/xlmr_ft
python src/detect/summarize_seeds.py twin_ft=twin_ft,twin_ft_s43,twin_ft_s44 \
    control=xlmr_ft,xlmr_ft_s43,xlmr_ft_s44 --out results/detection/twins/twin_ft_seeds
```

`--limit N` runs any training command on N documents per split as a smoke test.

### 5. Figures, tests and the demo

```bash
python docs/scripts/make_figures.py     # -> docs/assets/*.png (light and dark)
python -m pytest tests/ -q
python src/app/serve.py                 # try the tagger at http://127.0.0.1:5000
```

## Runtimes and cost

Measured on the hardware above.

| stage | time |
|---|---|
| dataset generation (11,587 API calls) | ~$26.73 in API cost |
| likelihood features, full corpus | 45 min |
| XLM-R tagger, 8 epochs | 109 min |
| twin training from scratch, 8 epochs | 72–159 min |
| twin fine-tuning, 3 epochs | 29–40 min |
| control fine-tuning, 3 epochs | 17–30 min |
| re-scoring a saved model (`--eval-only`) | ~1.5 min |

Twin training runs twice as many documents per epoch as the baseline, since
every document is paired with its twin.
