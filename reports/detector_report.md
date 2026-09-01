# Boundary detector results

Dataset: **854 documents**, 7935 labelled sentences (train 502 / dev 172 / test 180 documents).

## Protocol

- Trained on `train`, which contains seen generators only (the held-out generator is barred from train by construction).
- Hyperparameters tuned on `dev` **restricted to seen generators**. The held-out generator appears in dev as well as test, so tuning on all of dev would select settings using the generator whose novelty is then being measured.
- Reported on `test`, split into seen vs held-out. The gap between them is the real result: it distinguishes a detector that learned *AI text* from one that learned *these two models' text*.

Read **F1 on the AI class**, not accuracy: the AI class is a minority, so `all-human` scores well on accuracy while detecting nothing.

## What these numbers say

**1. The position-only baseline is the bar, and the text models barely clear it.** Knowing nothing but where a sentence sits in the document scores 0.518 F1(AI) and 0.421 exact-boundary F1. The best text model reaches 0.586 F1(AI) but only 0.295 on exact boundaries - *worse* than reading no text at all. Character n-grams are picking up some signal about which sentences are machine-written, but not about where authorship changes.

**2. Held-out generalisation gap is real.** The best model scores 0.638 F1(AI) on the seen generators and 0.508 on the held-out one, a drop of 0.131. On the held-out generator it is level with the position baseline (0.487), i.e. it has learned these two models' habits rather than machine text in general.

**3. Continuation is easy; span replacement is hard.** Type 1 reaches 0.787 F1(AI), while Type 2 gets 0.357 and Type 3 0.346. A single trailing AI block is detectable; short rewritten spans surrounded by human text largely are not. This is the dataset working as intended - Types 2 and 3 exist precisely because they are the hard case.

**4. Run smoothing hurt, and the reason is informative.** Merging author runs shorter than two sentences cut exact-boundary F1 further. Type 3 spans are 1-2 sentences by design, so the smoother deletes genuine single-sentence AI spans along with the noise. Any sequence model here must be able to emit one-sentence spans.

**5. Read exact-boundary F1, not ±1.** With ±1 tolerance even the random baseline scores 0.542, because scattering boundaries liberally puts one near almost every true change. The tolerant metric rewards over-prediction; the exact one does not.

### Implication

A bag-of-character-n-grams classifier scoring each sentence independently is the wrong shape for this task: it cannot represent *discontinuity* between neighbours, which is the signal boundary detection actually rests on. These results are a floor to beat, not a solution. A sequence model over the whole document - fine-tuned multilingual encoder with per-sentence outputs - is the natural next step.

## Overall test set

| model | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| baseline: all-human | 0.644 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| baseline: all-AI | 0.356 | 0.526 | 0.356 | 1.000 | 0.000 | 0.000 | 0.000 |
| baseline: random (train prior) | 0.552 | 0.371 | 0.371 | 0.370 | 0.360 | 0.542 | 0.000 |
| baseline: position only (no text) | 0.686 | 0.518 | 0.571 | 0.473 | 0.421 | 0.612 | 0.000 |
| text only | 0.707 | 0.574 | 0.596 | 0.553 | 0.377 | 0.572 | 0.061 |
| text + context | 0.720 | 0.586 | 0.619 | 0.556 | 0.295 | 0.508 | 0.094 |
| text + context + smoothing | 0.716 | 0.561 | 0.624 | 0.510 | 0.201 | 0.375 | 0.106 |
| text + position (diagnostic) | 0.674 | 0.580 | 0.537 | 0.630 | 0.366 | 0.581 | 0.089 |

## Seen generators (DeepSeek V3, GPT-4o)

| model | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| baseline: all-human | 0.643 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| baseline: all-AI | 0.357 | 0.526 | 0.357 | 1.000 | 0.000 | 0.000 | 0.000 |
| baseline: random (train prior) | 0.552 | 0.377 | 0.374 | 0.380 | 0.372 | 0.587 | 0.019 |
| baseline: position only (no text) | 0.699 | 0.540 | 0.593 | 0.496 | 0.442 | 0.627 | 0.000 |
| text only | 0.733 | 0.629 | 0.623 | 0.634 | 0.403 | 0.610 | 0.086 |
| text + context | 0.745 | 0.638 | 0.646 | 0.631 | 0.328 | 0.529 | 0.133 |
| text + context + smoothing | 0.746 | 0.625 | 0.660 | 0.594 | 0.246 | 0.424 | 0.152 |
| text + position (diagnostic) | 0.715 | 0.639 | 0.583 | 0.706 | 0.405 | 0.623 | 0.114 |

## Held-out generator (Gemini 2.5 Pro)

| model | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| baseline: all-human | 0.644 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| baseline: all-AI | 0.356 | 0.525 | 0.356 | 1.000 | 0.000 | 0.000 | 0.000 |
| baseline: random (train prior) | 0.542 | 0.349 | 0.353 | 0.345 | 0.374 | 0.563 | 0.000 |
| baseline: position only (no text) | 0.668 | 0.487 | 0.541 | 0.443 | 0.395 | 0.592 | 0.000 |
| text only | 0.672 | 0.490 | 0.549 | 0.443 | 0.346 | 0.526 | 0.027 |
| text + context | 0.686 | 0.508 | 0.574 | 0.455 | 0.255 | 0.482 | 0.040 |
| text + context + smoothing | 0.675 | 0.464 | 0.561 | 0.396 | 0.141 | 0.312 | 0.040 |
| text + position (diagnostic) | 0.619 | 0.495 | 0.469 | 0.525 | 0.319 | 0.530 | 0.053 |

## Best model by generator

| generator | role | docs | sent F1(AI) | bound F1 ±1 |
|---|---|---|---|---|
| deepseek_v3 | seen | 52 | 0.605 | 0.301 |
| gemini_2_5_pro | held_out | 75 | 0.464 | 0.312 |
| gpt_4o | seen | 53 | 0.642 | 0.545 |

## Best model by construction type

| construction | docs | sent F1(AI) | bound F1 exact | bound F1 ±1 |
|---|---|---|---|---|
| type1_single_boundary | 59 | 0.787 | 0.262 | 0.631 |
| type2_single_internal_segment | 73 | 0.357 | 0.226 | 0.434 |
| type3_multiple_internal_segments | 48 | 0.346 | 0.152 | 0.210 |

Selected hyperparameters: `{'ngram': '(3, 5)', 'C': '0.5', 'context': '0'}`
