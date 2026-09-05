# Boundary detector results

Dataset: **2312 documents**, 21357 labelled sentences (train 1477 / dev 402 / test 433 documents).

## Protocol

- Trained on `train`, which contains seen generators only (the held-out generator is barred from train by construction).
- Hyperparameters tuned on `dev` **restricted to seen generators**. The held-out generator appears in dev as well as test, so tuning on all of dev would select settings using the generator whose novelty is then being measured.
- Reported on `test`, split into seen vs held-out. The gap between them is the real result: it distinguishes a detector that learned *AI text* from one that learned *these two models' text*.

Read **F1 on the AI class**, not accuracy: the AI class is a minority, so `all-human` scores well on accuracy while detecting nothing.

## What these numbers say

**1. The position-only baseline is the bar, and the text models barely clear it.** Knowing nothing but where a sentence sits in the document scores 0.578 F1(AI) and 0.293 exact-boundary F1. The best text model reaches 0.678 F1(AI) but only 0.419 on exact boundaries - *worse* than reading no text at all. Character n-grams are picking up some signal about which sentences are machine-written, but not about where authorship changes.

**2. Held-out generalisation gap is real.** The best model scores 0.708 F1(AI) on the seen generators and 0.511 on the held-out one, a drop of 0.196. On the held-out generator it is level with the position baseline (0.519), i.e. it has learned these two models' habits rather than machine text in general.

**3. Continuation is easy; span replacement is hard.** Type 1 reaches 0.840 F1(AI), while Type 2 gets 0.402 and Type 3 0.422. A single trailing AI block is detectable; short rewritten spans surrounded by human text largely are not. This is the dataset working as intended - Types 2 and 3 exist precisely because they are the hard case.

**4. Run smoothing hurt, and the reason is informative.** Merging author runs shorter than two sentences cut exact-boundary F1 further. Type 3 spans are 1-2 sentences by design, so the smoother deletes genuine single-sentence AI spans along with the noise. Any sequence model here must be able to emit one-sentence spans.

**5. Read exact-boundary F1, not ±1.** With ±1 tolerance even the random baseline scores 0.548, because scattering boundaries liberally puts one near almost every true change. The tolerant metric rewards over-prediction; the exact one does not.

### Implication

A bag-of-character-n-grams classifier scoring each sentence independently is the wrong shape for this task: it cannot represent *discontinuity* between neighbours, which is the signal boundary detection actually rests on. These results are a floor to beat, not a solution. A sequence model over the whole document - fine-tuned multilingual encoder with per-sentence outputs - is the natural next step.

## Overall test set

| model | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| baseline: all-human | 0.628 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| baseline: all-AI | 0.372 | 0.542 | 0.372 | 1.000 | 0.000 | 0.000 | 0.000 |
| baseline: random (train prior) | 0.548 | 0.387 | 0.390 | 0.384 | 0.360 | 0.548 | 0.009 |
| baseline: position only (no text) | 0.681 | 0.578 | 0.569 | 0.587 | 0.293 | 0.572 | 0.000 |
| text only | 0.762 | 0.680 | 0.681 | 0.679 | 0.468 | 0.643 | 0.122 |
| text + context | 0.759 | 0.678 | 0.674 | 0.682 | 0.419 | 0.600 | 0.134 |
| text + context + smoothing | 0.751 | 0.651 | 0.680 | 0.624 | 0.282 | 0.452 | 0.159 |
| text + position (diagnostic) | 0.767 | 0.691 | 0.682 | 0.700 | 0.465 | 0.639 | 0.166 |

## Seen generators (DeepSeek V3, GPT-4o)

| model | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| baseline: all-human | 0.625 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| baseline: all-AI | 0.375 | 0.546 | 0.375 | 1.000 | 0.000 | 0.000 | 0.000 |
| baseline: random (train prior) | 0.526 | 0.360 | 0.365 | 0.356 | 0.335 | 0.533 | 0.003 |
| baseline: position only (no text) | 0.689 | 0.590 | 0.583 | 0.598 | 0.295 | 0.582 | 0.000 |
| text only | 0.778 | 0.711 | 0.695 | 0.727 | 0.495 | 0.668 | 0.142 |
| text + context | 0.774 | 0.708 | 0.686 | 0.730 | 0.447 | 0.623 | 0.154 |
| text + context + smoothing | 0.763 | 0.680 | 0.691 | 0.670 | 0.304 | 0.477 | 0.179 |
| text + position (diagnostic) | 0.782 | 0.720 | 0.695 | 0.746 | 0.494 | 0.668 | 0.187 |

## Held-out generator (Gemini 2.5 Pro)

| model | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| baseline: all-human | 0.644 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| baseline: all-AI | 0.356 | 0.525 | 0.356 | 1.000 | 0.000 | 0.000 | 0.000 |
| baseline: random (train prior) | 0.542 | 0.352 | 0.355 | 0.349 | 0.372 | 0.569 | 0.000 |
| baseline: position only (no text) | 0.645 | 0.519 | 0.502 | 0.537 | 0.282 | 0.529 | 0.000 |
| text only | 0.690 | 0.507 | 0.585 | 0.447 | 0.346 | 0.531 | 0.027 |
| text + context | 0.693 | 0.511 | 0.590 | 0.451 | 0.281 | 0.487 | 0.040 |
| text + context + smoothing | 0.693 | 0.484 | 0.602 | 0.404 | 0.183 | 0.337 | 0.067 |
| text + position (diagnostic) | 0.700 | 0.532 | 0.598 | 0.478 | 0.336 | 0.514 | 0.067 |

## Best model by generator

| generator | role | docs | sent F1(AI) | bound F1 ±1 |
|---|---|---|---|---|
| deepseek_v3 | seen | 282 | 0.657 | 0.457 |
| gemini_2_5_pro | held_out | 75 | 0.484 | 0.337 |
| gpt_4o | seen | 76 | 0.750 | 0.569 |

## Best model by construction type

| construction | docs | sent F1(AI) | bound F1 exact | bound F1 ±1 |
|---|---|---|---|---|
| type1_single_boundary | 171 | 0.840 | 0.359 | 0.674 |
| type2_single_internal_segment | 146 | 0.402 | 0.292 | 0.470 |
| type3_multiple_internal_segments | 116 | 0.422 | 0.238 | 0.332 |

Selected hyperparameters: `{'ngram': '(2, 5)', 'C': '4.0', 'context': '0'}`
