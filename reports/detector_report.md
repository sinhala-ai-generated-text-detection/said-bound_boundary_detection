# Boundary detector results

Dataset: **4244 documents**, 39358 labelled sentences (train 2490 / dev 836 / test 918 documents).

## Protocol

- Trained on `train`, which contains seen generators only (the held-out generator is barred from train by construction).
- Hyperparameters tuned on `dev` **restricted to seen generators**. The held-out generator appears in dev as well as test, so tuning on all of dev would select settings using the generator whose novelty is then being measured.
- Reported on `test`, split into seen vs held-out. The gap between them is the real result: it distinguishes a detector that learned *AI text* from one that learned *these two models' text*.

Read **F1 on the AI class**, not accuracy: the AI class is a minority, so `all-human` scores well on accuracy while detecting nothing.

## What these numbers say

**1. The position-only baseline is the bar, and the text models barely clear it.** Knowing nothing but where a sentence sits in the document scores 0.567 F1(AI) and 0.284 exact-boundary F1. The best text model reaches 0.678 F1(AI) but only 0.438 on exact boundaries - *worse* than reading no text at all. Character n-grams are picking up some signal about which sentences are machine-written, but not about where authorship changes.

**2. Held-out generalisation gap is real.** The best model scores 0.724 F1(AI) on the seen generators and 0.571 on the held-out one, a drop of 0.153. On the held-out generator it is level with the position baseline (0.550), i.e. it has learned these two models' habits rather than machine text in general.

**3. Continuation is easy; span replacement is hard.** Type 1 reaches 0.821 F1(AI), while Type 2 gets 0.474 and Type 3 0.432. A single trailing AI block is detectable; short rewritten spans surrounded by human text largely are not. This is the dataset working as intended - Types 2 and 3 exist precisely because they are the hard case.

**4. Run smoothing hurt, and the reason is informative.** Merging author runs shorter than two sentences cut exact-boundary F1 further. Type 3 spans are 1-2 sentences by design, so the smoother deletes genuine single-sentence AI spans along with the noise. Any sequence model here must be able to emit one-sentence spans.

**5. Read exact-boundary F1, not ±1.** With ±1 tolerance even the random baseline scores 0.542, because scattering boundaries liberally puts one near almost every true change. The tolerant metric rewards over-prediction; the exact one does not.

### Implication

A bag-of-character-n-grams classifier scoring each sentence independently is the wrong shape for this task: it cannot represent *discontinuity* between neighbours, which is the signal boundary detection actually rests on. These results are a floor to beat, not a solution. A sequence model over the whole document - fine-tuned multilingual encoder with per-sentence outputs - is the natural next step.

## Overall test set

| model | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| baseline: all-human | 0.628 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| baseline: all-AI | 0.372 | 0.542 | 0.372 | 1.000 | 0.000 | 0.000 | 0.000 |
| baseline: random (train prior) | 0.534 | 0.367 | 0.371 | 0.364 | 0.357 | 0.542 | 0.005 |
| baseline: position only (no text) | 0.673 | 0.567 | 0.558 | 0.577 | 0.284 | 0.558 | 0.000 |
| text only | 0.757 | 0.675 | 0.671 | 0.678 | 0.482 | 0.639 | 0.130 |
| text + context | 0.761 | 0.678 | 0.680 | 0.677 | 0.438 | 0.635 | 0.155 |
| text + context + smoothing | 0.752 | 0.651 | 0.684 | 0.621 | 0.314 | 0.491 | 0.183 |
| text + position (diagnostic) | 0.771 | 0.696 | 0.687 | 0.704 | 0.474 | 0.653 | 0.167 |

## Seen generators (DeepSeek V3, GPT-4o)

| model | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| baseline: all-human | 0.625 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| baseline: all-AI | 0.375 | 0.545 | 0.375 | 1.000 | 0.000 | 0.000 | 0.000 |
| baseline: random (train prior) | 0.542 | 0.387 | 0.389 | 0.385 | 0.374 | 0.547 | 0.005 |
| baseline: position only (no text) | 0.678 | 0.576 | 0.569 | 0.584 | 0.286 | 0.568 | 0.000 |
| text only | 0.780 | 0.719 | 0.690 | 0.750 | 0.503 | 0.660 | 0.155 |
| text + context | 0.784 | 0.724 | 0.694 | 0.756 | 0.464 | 0.659 | 0.196 |
| text + context + smoothing | 0.772 | 0.698 | 0.692 | 0.704 | 0.339 | 0.518 | 0.225 |
| text + position (diagnostic) | 0.794 | 0.736 | 0.708 | 0.766 | 0.500 | 0.675 | 0.201 |

## Held-out generator (Gemini 2.5 Pro)

| model | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| baseline: all-human | 0.634 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| baseline: all-AI | 0.366 | 0.536 | 0.366 | 1.000 | 0.000 | 0.000 | 0.000 |
| baseline: random (train prior) | 0.524 | 0.341 | 0.346 | 0.337 | 0.362 | 0.548 | 0.013 |
| baseline: position only (no text) | 0.663 | 0.550 | 0.538 | 0.562 | 0.281 | 0.537 | 0.000 |
| text only | 0.711 | 0.575 | 0.624 | 0.532 | 0.443 | 0.601 | 0.079 |
| text + context | 0.716 | 0.571 | 0.640 | 0.515 | 0.387 | 0.589 | 0.072 |
| text + context + smoothing | 0.714 | 0.538 | 0.660 | 0.454 | 0.264 | 0.438 | 0.098 |
| text + position (diagnostic) | 0.725 | 0.606 | 0.638 | 0.578 | 0.424 | 0.613 | 0.098 |

## Best model by generator

| generator | role | docs | sent F1(AI) | bound F1 ±1 |
|---|---|---|---|---|
| deepseek_v3 | seen | 302 | 0.657 | 0.471 |
| gemini_2_5_pro | held_out | 305 | 0.538 | 0.438 |
| gpt_4o | seen | 311 | 0.733 | 0.565 |

## Best model by construction type

| construction | docs | sent F1(AI) | bound F1 exact | bound F1 ±1 |
|---|---|---|---|---|
| type1_single_boundary | 368 | 0.821 | 0.391 | 0.661 |
| type2_single_internal_segment | 299 | 0.474 | 0.350 | 0.532 |
| type3_multiple_internal_segments | 251 | 0.432 | 0.256 | 0.385 |

Selected hyperparameters: `{'ngram': '(2, 4)', 'C': '4.0', 'context': '0'}`
