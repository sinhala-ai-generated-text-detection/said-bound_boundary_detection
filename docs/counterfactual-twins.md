# Counterfactual twins

[README](../README.md) · [Dataset](dataset.md) · [Detectors](detectors.md) · [Likelihood](likelihood.md) · [Counterfactual twins](counterfactual-twins.md) · [Limitations](limitations.md) · [Reproducing](reproducing.md)

## The problem nobody measures

Every document the detector is trained on contains at least one boundary, by
construction. So does every test document. A detector can therefore learn
**"there is always machine text somewhere, find the most machine-like
sentence"** and still score well, because the benchmark never shows it a
document with no machine text.

In real use, most documents a detector sees are entirely human. So the question
the mixed-document metrics cannot answer is: **what does the detector do on
purely human text?**

## The twin: a free, exactly matched control

The construction in [The three construction types](dataset.md#the-three-construction-types) *replaces* human sentences rather than inserting new
ones. So for every mixed document, its source window is the **same document
with every AI sentence swapped back for the human sentence it displaced**:

```
mixed:  [H0] [H1] [AI2] [H3] [H4]      labels 0 0 1 0 0
twin:   [H0] [H1] [H2 ] [H3] [H4]      labels 0 0 0 0 0
```

Same topic, same length, same positions. Only authorship differs, and only at
the replaced positions. This was verified, not assumed: all 4,244 documents
align sentence-for-sentence with their window in `sources/windows.jsonl`, and
the loader (`data.load_twins`) refuses to build a twin if any human sentence
does not match.

Many boundary-detection datasets cannot do this. Datasets built by insertion,
or by continuing a prompt with no human remainder kept, have no aligned human
counterpart.

## What the baseline does on twins

The saved XLM-R tagger, re-scored with no retraining
(`results/detection/twins/xlmr_rescored.md`), threshold decoding:

| on the 578 unique test twins | overall | seen | held-out |
|---|---|---|---|
| twins with ≥1 sentence flagged AI | **95.3%** | 94.4% | 95.8% |
| boundaries predicted per twin (correct: 0) | **3.07** | 3.01 | 3.15 |
| sentence false-positive rate | **26.2%** | 26.0% | 26.5% |

Three things make this worse than it looks:

- **The false-positive rate is higher with no machine text present.** Human
  sentences inside mixed documents are flagged about 18% of the time. In pure
  human documents it is 26%. The detector searches for machine text and
  "finds" it.
- **31% of the human originals at replaced positions are flagged as AI**, even
  though they are the genuine human sentences. Part of what the model calls
  "machine" is position and topic, not authorship.
- **No threshold fixes it.** Sweeping the threshold from 0.05 to 0.95 ([Operating points: is the cost real?](#operating-points-is-the-cost-real)),
  the baseline never flags fewer than 93% of twins. Its scores are too extreme
  for any cut-off to separate them.

The Viterbi decoder is worse still: 99.5% of twins flagged, 4.55 boundaries per
twin.

A useful single number is **exact-boundary F1 over the mixed documents and the
twins together**, where any boundary predicted in a twin counts as a false
positive. That is closer to deployment than mixed-only scoring, and for the
baseline it drops from **0.603 to 0.434**.

## The training method

With `--twins`, each mixed training document is batched **together with its
twin**, and the loss has three terms (`transformer.twin_loss`):

1. **Tagging loss on both documents.** Cross-entropy over the mixed document's
   labels and over the twin (all human). With no twins this is exactly the
   baseline objective.
2. **Margin term.** At every replaced position, the AI sentence's score must
   exceed the score of the human sentence it replaced by a margin (2.0 in logit
   space). Position and topic are identical across the pair, so neither can
   satisfy this term. Only authorship can.
3. **Consistency term.** For every sentence that is human in *both* documents,
   the two predictions should match (symmetric KL). Its author does not
   change, so its prediction should not depend on whether machine text
   appears elsewhere. This term targets the "there is always a boundary" prior
   directly.

`loss = CE + ramp × (λ_margin × margin + λ_consistency × consistency)`, with
both λ = 1.

### Details that matter

- **Shared windows are down-weighted.** One window can produce several records
  (different construction types or generators). In train, 856 windows back two
  records and 51 back three or four. Without correction their twins would count
  2–4 times, so each twin's loss is weighted by 1/k.
- **Pairs are matched per sentence, not per position.** A twin sentence can
  tokenize to a different length than the AI sentence it stands in for, so a
  long document may split into 512-token chunks at different places. The
  collate function (`collate_pairs`) records, for each sentence, where its
  marker landed in each document, and the loss compares those. A unit test
  forces the two documents to chunk differently and checks the mapping.
- **Memory is unchanged.** Batches hold 2 pairs (4 documents), the same as the
  baseline's 4 documents.

## The first attempt collapsed, and why

With the paired terms on from the first step, the model **collapsed to a
constant output** (`results/detection/twins/twin_no_warmup.md`):

| epoch | CE | margin | consistency | dev boundary F1 |
|---|---|---|---|---|
| 1 | 0.670 | 1.991 | 0.020 | 0.453 |
| 2–7 | ~0.642 | ~2.000 | 0.002–0.005 | 0.444, frozen |

Reading the log:

- **Margin stuck at 2.00** means the score gap between an AI span and its human
  original was always zero. The margin loss equals the margin exactly when the
  two scores are identical.
- **Consistency near zero** was achieved the cheap way: a model that outputs
  the same score for every sentence satisfies it perfectly.
- On test, every sentence got nearly the same score (paired win rate 0.03),
  and the threshold decoder labelled everything AI.

This is the same trap described in [The transformer detector — how it actually works](detectors.md#the-transformer-detector--how-it-actually-works): XLM-R spends its first epochs
predicting one class before it learns to separate them. Early on, the encoder
cannot tell a document from its twin, so the margin gradient cancels itself
out, while the consistency term actively *rewards* ignoring the input. The
paired terms held the model in the collapsed state it needed to escape.

**How that diagnosis was confirmed.** The `twin_plain` ablation uses the same
twins with both paired terms switched off. It trained normally (CE 0.556 →
0.039 over 8 epochs), so neither the extra human data nor the class balance
caused the collapse; the paired terms did.

**The fix: a warm-up.** `--twin-warmup 2` keeps the paired terms off for two
epochs, then ramps them in linearly over the third. With it, the model escapes
the single-class phase during epoch 2 (CE 0.645 → 0.443), and when the paired
terms arrive, training continues normally (margin 0.33 → 0.008 by epoch 8).

## Results

Single seed, 8 epochs each, threshold decoding unless noted. Twin columns are
on the unique test twins.

| | baseline | twins only (`twin_plain`) | twins + paired terms (`twin_warm`) |
|---|---|---|---|
| twin false-alarm rate | 95.3% | 58.7% | **49.0%** |
| boundaries per twin | 3.07 | 1.72 | **1.19** |
| sentence FPR on twins | 26.2% | 15.7% | **10.0%** |
| context stability | 0.885 | 0.931 | **0.956** |
| paired win rate | 0.878 | 0.921 | 0.921 |
| exact-boundary F1, mixed + twins | 0.434 | 0.448 | **0.464** |
| exact-boundary F1, mixed only (Viterbi) | **0.603** | 0.595 | 0.580 |
| held-out exact-boundary F1 (Viterbi) | **0.548** | 0.527 | 0.524 |

*Context stability* is how often an untouched human sentence gets the same
label in the mixed document and in its twin. *Paired win rate* is how often an
AI sentence scores above the human sentence it replaced.

What this shows:

- **Adding twins at all fixes much of the problem.** Twins as plain extra human
  documents cut false alarms by 37 points.
- **The paired terms go further**: fewer false alarms (49% vs 59%), fewer
  hallucinated boundaries (1.19 vs 1.72), better context stability.
- **The consistency term does the work; the margin term adds nothing
  measurable.** The paired win rate is 0.921 with or without the paired terms.
  Plain cross-entropy on the twin already pushes the human original down.
- **There is a cost on the original benchmark.** Mixed-only exact-boundary F1
  falls, and the held-out generator falls more.

## Operating points: is the cost real?

A single tuned threshold compares models at whatever point the selection rule
happened to pick. Here that rule saw only mixed dev documents, and the
threshold grid (0.20–0.80) cut off both the baseline (tuned to 0.80) and
`twin_plain` (tuned to 0.20). So `src/detect/twin_tradeoff.py` sweeps the
threshold from 0.05 to 0.95 for every saved model
(`results/detection/twins/twin_tradeoff_laptop.md`):

| model | twin false alarm, across all thresholds | best mixed-only exact-boundary F1 |
|---|---|---|
| baseline | 98% → 93% | **0.603** |
| twins only | 63% → 43% | 0.554 |
| twins + paired terms | **84% → 24%** | 0.554 |

- **The baseline cannot be made quiet on human text.** At every threshold it
  flags at least 93% of pure-human documents. This is the clearest finding of
  the whole study, and it is a property of the model, not of the tuning.
- **Twins alone lower the curve but flatten it.** False alarms bottom out at
  43%, and the threshold barely moves them.
- **Paired training is the only model with a usable operating range.** At
  threshold 0.925 it flags 28% of human documents while keeping 0.502 mixed-only
  exact-boundary F1. Neither other model can reach a 30% false-alarm rate at
  any threshold.
- **The mixed-only cost is real, not a tuning artifact.** Even at its best
  threshold, each twin model tops out at 0.554, against the baseline's 0.603.

Choosing each model's threshold on dev *mixed documents plus their twins*
(a deployment-aware rule) gives exact-boundary F1 over mixed + twins of 0.438
(baseline), 0.465 (twins only) and 0.464 (paired). On the held-out generator:
0.359, 0.364 and **0.381**.

## Fine-tuning the trained baseline with twins

Training from scratch with twins costs about 5 points of mixed-only accuracy.
An alternative is to start from the already trained baseline and **fine-tune
it with twins** (`--init-from models/xlmr_tagger`, 3 epochs, learning rate
1e-5, same loss, no warm-up needed since the model is already past the
single-class phase). This is `twin_ft`.

That comparison needs a control. The baseline was still improving when its
training stopped, so any gain could simply come from training longer.
`xlmr_ft` gets the same 3 extra epochs at the same learning rate, on mixed
documents only.

First run (seed 42; [Three seeds](#three-seeds) repeats it on three seeds and revises part of this
reading):

| test set | baseline | control `xlmr_ft` | **`twin_ft`** |
|---|---|---|---|
| exact-boundary F1, mixed only (Viterbi) | 0.603 | 0.593 | **0.594** |
| held-out exact-boundary F1 (Viterbi) | 0.548 | 0.543 | **0.548** |
| sentence F1 (Viterbi) | 0.792 | 0.789 | **0.794** |
| twin false-alarm rate (threshold) | 95.3% | 96.7% | **58.8%** |
| boundaries per twin | 3.07 | 3.16 | **1.61** |
| sentence FPR on twins | 26.2% | 27.5% | **14.4%** |
| exact-boundary F1, mixed + twins | 0.434 | 0.421 | **0.472** |
| … on the held-out generator | 0.353 | 0.337 | **0.401** |

- **On this seed, twin fine-tuning matched the control on mixed documents**
  with Viterbi decoding (0.594 vs 0.593). Three seeds show that this seed was
  the favourable one; there is a small cost ([Three seeds](#three-seeds)). With threshold decoding the
  cost was already visible: its best mixed-only score across thresholds is
  0.578, against 0.597 for the control.
- **Extra training alone does not help.** The control is slightly *worse* than
  the original baseline, and just as prone to false alarms (96.7%).
- **It gives the best deployment-view score of any model**: 0.481 exact-boundary
  F1 over mixed documents plus twins at its best dev-selected threshold, and
  0.408 on the held-out generator.
- **The trade-off: a narrower range.** Across thresholds, `twin_ft`'s false
  alarms move between 69% and 53%. It cannot reach the 24% of the
  from-scratch model.

So the two variants serve different needs. **Fine-tuning** stays close to the
baseline's accuracy and roughly halves false alarms. **Training from
scratch** reaches much lower false alarms at a cost of about 5 points.

## Three seeds

`twin_ft` and its control were repeated with seeds 43 and 44 (`--seed`), and
`src/detect/summarize_seeds.py` compares them
(`results/detection/twins/twin_ft_seeds.md`). Runs are **paired by seed**: each
pair shares its random initialisation of the head and its data order, so
noise common to both cancels in the difference.

All six runs fine-tune the same trained baseline. The seeds therefore vary the
fine-tuning, not the baseline's own training. That is enough to test the claim
being made (twin fine-tuning versus the same amount of ordinary training), but
it is not a full replication from scratch.

| test set | `twin_ft` | control | paired difference | twin better in |
|---|---|---|---|---|
| twin false-alarm rate | 58.3% ± 1.1 | 95.4% ± 1.3 | **−37.1 ± 0.7 pts** | 3/3 seeds |
| boundaries per twin | 1.60 ± 0.01 | 3.11 ± 0.11 | −1.50 ± 0.10 | 3/3 |
| sentence FPR on twins | 14.1% ± 0.4 | 26.4% ± 1.6 | −12.3 ± 1.2 pts | 3/3 |
| exact-boundary F1, mixed + twins | 0.472 ± 0.004 | 0.430 ± 0.010 | **+0.042 ± 0.009** | 3/3 |
| … on the held-out generator | 0.404 ± 0.003 | 0.348 ± 0.011 | **+0.056 ± 0.008** | 3/3 |
| exact-boundary F1, mixed only (Viterbi) | 0.596 ± 0.002 | 0.607 ± 0.012 | −0.010 ± 0.010 | 1/3 |
| … on the held-out generator | 0.554 ± 0.005 | 0.548 ± 0.006 | +0.006 ± 0.007 | 2/3 |
| exact-boundary F1, mixed only (threshold) | 0.574 ± 0.004 | 0.599 ± 0.010 | −0.025 ± 0.010 | 0/3 |
| sentence F1 (Viterbi) | 0.791 ± 0.004 | 0.789 ± 0.000 | +0.002 ± 0.004 | 2/3 |

Mean ± sample standard deviation over three seeds.

- **The false-alarm result is robust.** A 37-point reduction with a spread
  under one point, on every seed, as are the gains in the deployment view (+4.2
  points overall, +5.6 on the held-out generator).
- **There is a small mixed-document cost.** About 1 point with Viterbi decoding,
  which is within about one standard deviation, and about 2.5 points with
  threshold decoding, which holds on every seed. The seed-42 result in [Fine-tuning the trained baseline with twins](#fine-tuning-the-trained-baseline-with-twins)
  (0.594 vs 0.593) was the seed on which twin training did best.
- **No cost on the held-out generator**, where the two are level.
- **Twin training is more stable.** Its mixed-only exact-boundary F1 varies by
  ±0.002 across seeds, against ±0.012 for the control.

The threshold sweep ([Operating points: is the cost real?](#operating-points-is-the-cost-real)) over all six models agrees. The lowest false-alarm
rate reachable at any threshold is **48% ± 6** for `twin_ft` against
**93% ± 2** for the control. With each model's threshold chosen on dev mixed
documents plus twins, exact-boundary F1 over mixed documents plus twins is
0.476 ± 0.006 against 0.440 ± 0.010 (held-out generator: 0.402 ± 0.009
against 0.358 ± 0.011).

## Is it the twin, or any human text?

Twin fine-tuning changes three things at once: it adds human-only documents,
those documents are content-matched to the mixed ones, and it adds the paired
margin and consistency terms. Four conditions separate them. Each fine-tunes
the trained tagger for 3 epochs (learning rate 1e-5, batch 4, seeds 42–44),
all on the DGX Spark with the same code:

| condition | human documents | loss |
|---|---|---|
| `ft_control` | none | CE |
| `ft_unrelated` | an **unrelated** human window per twin | CE |
| `ft_twins_unpaired` | the matched twin | CE |
| `ft_twins_paired` | the matched twin | CE + margin + consistency |

The unrelated windows (`data.load_unrelated_twins`, `--twin-source unrelated`)
come from train-split articles that no record in any split uses. Each window a
training document uses is mapped to a distinct unrelated window with exactly
the same number of sentences, and documents that share a window share its
stand-in with the same 1/k weight. So the amount, length and weighting of human
text are identical to the twin conditions; only content matching differs. The
unrelated windows are about 5% longer in characters (1,134 vs 1,082 per
document), since only sentence counts are matched.

A second false-alarm test set answers the worry that twin *evaluation* favours
twin *training*: the **618 test-split windows that no mixed document uses**
(5,502 sentences), with no content link to anything a model was trained or
tested on.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/fig6-conditions-dark.png">
  <img alt="Four conditions over three seeds. Unrelated human documents with a false alarm, threshold decoding: no human documents 93%, unrelated human 60%, twins CE only 62%, twins plus paired terms 58%. With Viterbi decoding: 99%, 91%, 84%, 76%. Exact-boundary F1 on mixed documents: 0.617, 0.618, 0.615, 0.604." src="assets/fig6-conditions-light.png">
</picture>

Test set, mean ± standard deviation over three seeds
([`four_conditions.md`](../results/detection/twins/four_conditions.md)):

| | control | unrelated | twins, CE only | twins + paired |
|---|---|---|---|---|
| unrelated-human false alarm (threshold) | 93.1% ± 2.5 | 59.7% ± 8.0 | 61.5% ± 4.0 | **57.7% ± 2.4** |
| unrelated-human false alarm (Viterbi) | 99.3% ± 0.1 | 90.6% ± 3.6 | 83.9% ± 4.1 | **75.9% ± 3.5** |
| boundaries per unrelated human document | 2.85 ± 0.15 | 1.49 ± 0.17 | 1.58 ± 0.11 | **1.44 ± 0.05** |
| twin false alarm (threshold) | 94.9% ± 2.0 | 63.2% ± 9.2 | 63.3% ± 6.2 | **58.4% ± 3.0** |
| twin false alarm (Viterbi) | 99.7% ± 0.3 | 92.7% ± 2.5 | 86.3% ± 4.0 | **79.2% ± 3.5** |
| exact-boundary F1, mixed + twins | 0.432 ± 0.002 | 0.459 ± 0.006 | **0.470 ± 0.010** | 0.469 ± 0.008 |
| exact-boundary F1, mixed only (Viterbi) | 0.617 ± 0.004 | **0.618 ± 0.008** | 0.615 ± 0.004 | 0.604 ± 0.001 |
| exact-boundary F1, mixed only (threshold) | **0.602 ± 0.004** | 0.566 ± 0.012 | 0.580 ± 0.003 | 0.569 ± 0.011 |
| sentence F1 (Viterbi) | 0.785 ± 0.003 | 0.778 ± 0.010 | **0.792 ± 0.004** | 0.787 ± 0.002 |

False alarms are the share of human documents with any sentence flagged, at
each model's dev-tuned threshold or bias.

Paired differences by seed, one factor at a time (seeds on which the first
condition is better in brackets; false alarms in points):

| | (a) unrelated − control | (b) twins CE − unrelated | (c) paired − twins CE |
|---|---|---|---|
| unrelated-human false alarm (threshold) | **−33.4 ± 9.9** (3/3) | +1.8 ± 4.3 (1/3) | −3.9 ± 2.0 (3/3) |
| unrelated-human false alarm (Viterbi) | −8.7 ± 3.6 (3/3) | −6.7 ± 0.9 (3/3) | −8.0 ± 6.8 (3/3) |
| twin false alarm (threshold) | **−31.7 ± 10.3** (3/3) | +0.1 ± 3.2 (2/3) | −5.0 ± 3.8 (3/3) |
| twin false alarm (Viterbi) | −7.0 ± 2.4 (3/3) | −6.4 ± 2.6 (3/3) | −7.0 ± 4.0 (3/3) |
| exact-boundary F1, mixed + twins | +0.027 ± 0.006 (3/3) | +0.011 ± 0.012 (2/3) | −0.001 ± 0.011 (1/3) |
| exact-boundary F1, mixed only (Viterbi) | +0.001 ± 0.009 (1/3) | −0.003 ± 0.011 (1/3) | −0.012 ± 0.005 (0/3) |
| sentence F1 (Viterbi) | −0.007 ± 0.011 (0/3) | +0.014 ± 0.008 (3/3) | −0.005 ± 0.003 (0/3) |

The full method against unrelated human text directly (b + c,
[`four_conditions_twins_paired_vs_unrelated.md`](../results/detection/twins/four_conditions_twins_paired_vs_unrelated.md)):
−2.0 ± 6.2 points of unrelated-human false alarms with threshold decoding (2/3
seeds), −14.7 ± 6.6 with Viterbi (3/3), +0.010 ± 0.003 exact-boundary F1 over
mixed documents and twins (3/3), and −0.015 ± 0.006 on mixed documents alone
(0/3).

**Across thresholds**
([`twin_tradeoff.md`](../results/detection/twins/twin_tradeoff.md), sweeping
0.05–0.95 and measuring false alarms on the unrelated human documents):

| | lowest false alarm reachable | best mixed-only exact-boundary F1 at ≤ 50% false alarms |
|---|---|---|
| control | 89.7% (88–92) | unreachable |
| unrelated | **21.2%** (9–35) | 0.554 (0.544–0.560) |
| twins, CE only | 35.0% (29–38) | **0.576** (0.572–0.580) |
| twins + paired | 30.4% (20–39) | 0.564 (0.554–0.571) |

Read off the test curve, so these describe the trade-off each model offers, not
a tuned result. At a matched false-alarm level both twin conditions keep more
mixed-document accuracy than unrelated text on every seed; the unrelated model
reaches the lowest floor, but only at extreme thresholds and with a wide seed
spread.

### Reading

- **(a) Most of the effect comes from adding any human-only text.** At the
  tuned operating point, unrelated human documents alone cut false alarms from
  93% to 60%, against 58% for the full twin method: about nine-tenths of the
  reduction. This is the answer to the question a reviewer will ask first, and
  it shrinks what the twin itself can claim.
- **(b) Content matching on its own does nothing for threshold false alarms**
  (+1.8 ± 4.3 points). It does help in two places, consistently but modestly:
  with Viterbi decoding (−6.7 ± 0.9 points, 3/3 seeds) and in the trade-off
  (+2.3 points of mixed-only F1 at a matched ≤ 50% false-alarm rate, 3/3).
  Sentence F1 is also higher on every seed (+0.014).
- **(c) The paired terms add a small, consistent further reduction**: −3.9
  points (threshold) and −8.0 points (Viterbi) of unrelated-human false alarms,
  on every seed, and the lowest seed-to-seed spread of any condition. They cost
  1.2 ± 0.5 points of mixed-only exact-boundary F1 (Viterbi), on every seed.
  The paired terms need an aligned twin, so this is the part of the gain that
  only twins make possible.
- **The twin evaluation does not favour twin training.** Each condition's mean
  false-alarm rate on unrelated human documents is within 1–4 points of its
  rate on twins, with either decoder, and the ordering of the conditions is the
  same on both sets.
- **The earlier headline replicates on this machine**: twins + paired terms
  flag 58.4% of twins (laptop: 58.3%), against 94.9% for the control.

So the fair statement is: *adding human-only documents* fixes most of the
false-alarm problem; content-matched twins and the paired terms they enable
add a smaller, consistent gain, larger under structured decoding, at about one
point of mixed-only accuracy. The twin also remains what makes the problem
measurable at no extra cost.

## Where this stands

- **Established:** a detector trained only on mixed documents invents
  authorship changes in nearly every human document, and no threshold fixes
  it. More training on mixed documents does not fix it either. The twin-based
  evaluation that shows this costs nothing extra to build.
- **Established (three seeds):** fine-tuning with twins cuts false alarms on
  human documents from 95% to 58% and improves exact-boundary F1 over mixed
  plus human documents by 4 points (6 on the unseen generator), for a cost of
  about 1 point on mixed documents alone (Viterbi decoding).
- **Established (three seeds, four conditions):** most of that reduction comes
  from adding human-only documents at all; unrelated human text reaches 60%
  against the twins' 58%. Content matching and the paired terms add a smaller,
  gain on top: 2–5 points with threshold decoding (not on every seed) and
  14–15 points with Viterbi decoding (every seed)
  ([Is it the twin, or any human text?](#is-it-the-twin-or-any-human-text)).
- **Promising, single seed:** training from scratch with twins reaches a 24%
  false-alarm rate, at a cost of about 5 points. The consistency term is what
  matters.

Next experiments: drop the margin term and vary the consistency weight;
fine-tune for longer or at a higher consistency weight to push the
false-alarm floor lower; seeds for the from-scratch variant; select
checkpoints on dev mixed + twins rather than mixed only.
