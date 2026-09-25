# Sinhala human–AI authorship boundary detection

**Can a detector find where a Sinhala document switches from human to machine writing?**<br>
A 4,244-document benchmark built from Sinhala Wikipedia, detectors from character n-grams to a
fine-tuned XLM-RoBERTa tagger, and counterfactual-twin training for purely human text.

[![Python](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white)](docs/reproducing.md#environment)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%20·%20CUDA%2012.4-EE4C2C?logo=pytorch&logoColor=white)](docs/reproducing.md#environment)
[![Transformers](https://img.shields.io/badge/transformers-5.16-FFD21E)](docs/reproducing.md#environment)
[![Tests](https://img.shields.io/badge/tests-121%20passing-1baf7a)](tests/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Every document mixes human sentences with sentences written by a large language model. Each
sentence is labelled human or machine, and a **boundary** is any point where the label changes.
The task is to recover those boundaries from the text alone.

## Key findings

- **Encoding the whole document is the largest gain.** A fine-tuned XLM-RoBERTa tagger reaches
  0.603 exact-boundary F1, against 0.438 for a per-sentence linear model and 0.284 for a
  baseline that reads no text at all.
- **Unseen generators and embedded rewrites remain hard.** Exact-boundary F1 falls from 0.628
  to 0.556 on a generator held out of training, and sentence F1 drops from 0.925 on
  continuations to 0.65–0.70 on spans rewritten inside human text.
- **Detectors trained on mixed documents invent machine text in human documents.** Shown the
  all-human version of each test document, the tagger flags machine text in **95%** of them,
  and **no decision threshold brings that below 93%**.
- **Counterfactual twins fix much of it.** Fine-tuning on each document together with its
  all-human twin cuts those false alarms from 95% to **58%**, on every one of three seeds,
  and raises exact-boundary F1 over mixed and human documents by 4 points. The cost is about
  1 point of F1 on mixed documents alone.
- **Language-model likelihood transfers across generators.** On the unseen generator, a
  detector given only likelihood features, and no text, beats character n-grams on sentence
  F1 (0.588 vs 0.571).

## Results

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/fig2-models-dark.png">
  <img alt="Sentence F1 and exact-boundary F1 on the test set for four detectors. Position only: 0.567 and 0.284. Linear character n-grams: 0.678 and 0.438. Linear plus likelihood: 0.704 and 0.450. XLM-R tagger: 0.802 and 0.603." src="docs/assets/fig2-models-light.png">
</picture>

**Detectors.** Exact-boundary F1 is the primary metric: a predicted boundary counts only at
the exact sentence, and the trivial all-human and all-machine predictors score 0 on it. Only
the document-level tagger clears 0.5.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/fig3-breakdown-dark.png">
  <img alt="XLM-R on seen versus held-out generators: sentence F1 0.830 vs 0.742, exact-boundary F1 0.628 vs 0.556, documents fully correct 0.418 vs 0.239. By construction type, sentence F1 is 0.925 for continuation, 0.698 for single span and 0.653 for multi span." src="docs/assets/fig3-breakdown-light.png">
</picture>

**Generalisation and difficulty.** The gap to the held-out generator widens as the metric gets
stricter: the share of documents with every boundary right nearly halves. Continuation is far
easier than rewriting spans inside human text.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/fig4-twin-tradeoff-dark.png">
  <img alt="Exact-boundary F1 on mixed documents against the share of all-human documents with a false alarm, swept over decision thresholds. The XLM-R baseline stays between 93 and 98 percent false alarms. Twin-trained models reach 43, 53 and 24 percent at their lowest." src="docs/assets/fig4-twin-tradeoff-light.png">
</picture>

**False alarms on human text.** Each curve sweeps one model's decision threshold from 0.05 to
0.95. The baseline cannot be made quiet on human documents at any threshold. Twin fine-tuning
keeps most of its accuracy while halving false alarms; twin training from scratch reaches the
lowest rate, 24%, at a larger accuracy cost.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/fig5-seeds-dark.png">
  <img alt="Three seeds each of extra training with and without twins. False alarms: 95.4 vs 58.3 percent. Exact-boundary F1 on mixed plus human documents: 0.430 vs 0.472. Exact-boundary F1 on mixed documents only: 0.607 vs 0.596." src="docs/assets/fig5-seeds-light.png">
</picture>

**Three seeds.** The trained tagger fine-tuned for three more epochs with twins, against a
control given the same three epochs without them. The false-alarm reduction is 37.1 ± 0.7
points and holds on every seed; the mixed-document cost is 1.0 ± 1.0 points.

### At a glance

| | Question | Answer | Evidence |
|---|---|---|---|
| **1** | Can text be localised better than position alone? | Yes: 0.603 vs 0.284 exact-boundary F1 | [xlmr](results/detection/xlmr/xlmr.md) |
| **2** | Does it hold for an unseen generator? | Partly: 0.556 vs 0.628 | [xlmr](results/detection/xlmr/xlmr.md) |
| **3** | Do likelihood features help? | Linear model yes, XLM-R no (redundant) | [linear](results/detection/baselines/linear.md), [xlmr_likelihood](results/detection/xlmr/xlmr_likelihood.md) |
| **4** | Does the tagger stay quiet on human documents? | No: 95% false alarms at any threshold | [twin_tradeoff](results/detection/twins/twin_tradeoff.md) |
| **5** | Do counterfactual twins reduce that? | Yes: 95% → 58%, 3/3 seeds | [twin_ft_seeds](results/detection/twins/twin_ft_seeds.md) |
| **6** | At what cost on mixed documents? | About 1 point exact-boundary F1 | [twin_ft_seeds](results/detection/twins/twin_ft_seeds.md) |

## Dataset

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/fig1-constructions-dark.png">
  <img alt="The three construction types as sentence sequences. Type 1: four human sentences then four machine sentences. Type 2: one internal machine span. Type 3: several separate machine spans." src="docs/assets/fig1-constructions-light.png">
</picture>

**4,244 documents · 39,358 labelled sentences · 2,630 source articles · 36.4% machine
sentences.** Human text comes from a 2022 Sinhala Wikipedia snapshot, taken before the public
release of ChatGPT. Machine text always *replaces* human sentences in place, so the surrounding
text stays genuine and every mixed document has an exactly aligned all-human twin.

| generator | role | train | dev | test | total |
|---|---|---|---|---|---|
| DeepSeek V3 | seen | 1,186 | 273 | 302 | 1,761 |
| GPT-4o (2024-11-20) | seen | 1,304 | 283 | 311 | 1,898 |
| Gemini 2.5 Pro | held out | 0 | 280 | 305 | 585 |
| **total** | | **2,490** | **836** | **918** | **4,244** |

Splits are assigned over source articles before any generation, so no human passage appears in
more than one split. Seven automatic checks gate every generation; failures are regenerated,
never hand-edited. See [Dataset construction](docs/dataset.md).

## Study design

```mermaid
flowchart LR
    W["<b>Sinhala Wikipedia</b><br/>26,470 articles<br/>2022 snapshot"]
    D["<b>Dataset</b><br/>clean · segment · window<br/>split by source article"]
    G["<b>Generation</b><br/>3 construction types<br/>3 generators, 1 held out<br/>7 validation checks"]
    M["<b>Detectors</b><br/>position · character n-grams<br/>likelihood · XLM-R tagger"]
    T["<b>Twin evaluation</b><br/>the all-human version<br/>of every test document"]
    F["<b>Twin training</b><br/>paired margin +<br/>consistency losses"]
    W --> D --> G --> M --> T --> F
```

The protocol is fixed throughout: train on seen generators only, choose epochs and thresholds on
development data restricted to seen generators, and report on test overall, seen and held out.

## Repository layout

```
.
├── src/
│   ├── dataset/            # cleaning, segmentation, windowing, splits, generation, validation
│   ├── detect/             # baselines, linear, likelihood, XLM-R tagger, twin training and analysis
│   ├── app/serve.py        # local web demo of the tagger
│   ├── segment.py          # the single Sinhala sentence segmenter, shared by every stage
│   └── utils.py
├── results/
│   ├── dataset/            # dataset audit
│   ├── generation/         # generator pilot, fluency review, cross-model comparison, evidence
│   └── detection/
│       ├── baselines/      # trivial, positional and linear detectors
│       ├── xlmr/           # the XLM-R tagger, with and without likelihood features
│       └── twins/          # counterfactual-twin runs, seeds, threshold sweep
├── docs/                   # methodology pages
│   ├── assets/             #   figures (light and dark)
│   └── scripts/make_figures.py  # regenerates every figure from results/
├── tests/                  # 121 unit tests
├── splits/                 # train/dev/test source-article ids
├── prompts/                # Sinhala prompt templates
├── config.yaml             # every tunable setting
└── requirements.txt
```

The source export, the built dataset, model checkpoints and caches are not versioned; see
[Reproducing](docs/reproducing.md#what-is-versioned).

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu130   # the build for your CUDA
pip install -r requirements.txt
python -m pytest tests/ -q
```

```bash
python src/detect/run_transformer.py --epochs 8                          # XLM-R tagger
python src/detect/run_transformer.py --init-from models/xlmr_tagger \
    --epochs 3 --lr 1e-5 --twins --tag twin_ft                            # twin fine-tuning
python src/app/serve.py                                                   # demo at 127.0.0.1:5000
```

Building the dataset needs the Wikipedia export and an OpenRouter API key (about $27 in API
cost). Training runs on a single 6 GB laptop GPU or an NVIDIA DGX Spark. [Reproducing](docs/reproducing.md) has the full run
order and runtimes.

## Documentation

| | |
|---|---|
| [Dataset construction](docs/dataset.md) | Sources, cleaning, segmentation, splits, generators, construction types, prompts, validation |
| [Detectors](docs/detectors.md) | The task, every metric, the protocol, baselines, the linear model, the XLM-R tagger |
| [Likelihood features](docs/likelihood.md) | Masked-LM likelihood as a detection signal, and why it is redundant inside XLM-R |
| [Counterfactual twins](docs/counterfactual-twins.md) | False alarms on human text, twin training, the collapse and its fix, seeds, operating points |
| [Limitations](docs/limitations.md) | Surface-noise confound, positional and length priors, selection protocol, scope |
| [Reproducing](docs/reproducing.md) | Environment, what is versioned, run order, runtimes |

## Limitations

- **One domain.** Only encyclopedic Wikipedia prose is covered.
- **One held-out generator.** Generalisation is measured against Gemini 2.5 Pro alone.
- **Surface cues.** Human Wikipedia sentences carry typographical noise that generator output
  lacks (a space before punctuation in 3.6% of human sentences against 0.01% of machine ones),
  so part of any detector's score may come from formatting.
- **Seeds.** Twin fine-tuning and its control have three seeds; other transformer results are
  single runs.
- **No human ceiling.** No annotation study establishes how well Sinhala readers do on the task.

## License

The code in this repository is released under the [MIT License](LICENSE).

The Sinhala Wikipedia text it builds on is licensed CC BY-SA 4.0 and is not redistributed here;
neither is the generated dataset.
