# Transformer detector — XLM-RoBERTa sentence tagger

`xlm-roberta-base` fine-tuned for per-sentence human/AI tagging: 8 epochs, lr 3e-05, effective batch 4, best epoch 1, 147.3 min on GPU. Trained on 2490 documents.

The whole document is encoded in one pass with a marker token before each sentence, so each sentence representation is built with its neighbours in attention range.

**Structured decoding.** Decoding is a Viterbi pass over the sentence scores with a single transition cost for changing author, tuned on dev-seen for **exact-boundary F1**. This is a linear-chain CRF decode with a constant transition. It matters because the previous setup tuned a per-sentence threshold for *sentence* F1 and then reported *boundary* F1 - optimising a different objective from the one reported. Unlike run-length smoothing it never forbids a one-sentence span; it only makes one cost two transitions.

Threshold 0.250 (sentence F1) and boundary bias +0.50 (exact-boundary F1), both tuned on dev restricted to seen generators.

## Test results

| slice | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| overall | 0.538 | 0.413 | 0.391 | 0.437 | 0.459 | 0.465 | 0.004 |
| seen generators | 0.535 | 0.411 | 0.391 | 0.432 | 0.459 | 0.464 | 0.008 |
| held-out (Gemini) | 0.548 | 0.423 | 0.397 | 0.452 | 0.462 | 0.467 | 0.003 |

### Decoder comparison (overall test)

| slice | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| threshold (per-sentence) | 0.373 | 0.542 | 0.372 | 0.999 | 0.000 | 0.000 | 0.000 |
| **viterbi (pair head)** | 0.538 | 0.413 | 0.391 | 0.437 | 0.459 | 0.465 | 0.004 |

## All-human twins

Each test document is also scored as its all-human source window: same topic, length and positions, no machine text. *False alarm* is the share of twins with any sentence flagged; *win* is how often an AI sentence scores above the human sentence it replaced; *flip* is how often it is flagged while that human sentence is not; *ctx stable* is how often an untouched human sentence gets the same label in both documents; *bE + twins* is exact-boundary F1 over the mixed documents and the twins together.

Trained with counterfactual twins: margin weight 1.0, consistency weight 1.0, margin 2.0.

| decoder | slice | twins | false alarm | boundaries/twin | sentence FPR | win | flip | ctx stable | bE + twins |
|---|---|---|---|---|---|---|---|---|---|
| threshold | overall | 578 | 1.000 | 0.00 | 1.000 | 0.030 | 0.000 | 0.999 | 0.000 |
| threshold | seen | 377 | 1.000 | 0.00 | 0.999 | 0.035 | 0.000 | 0.999 | 0.000 |
| threshold | held_out | 289 | 1.000 | 0.00 | 1.000 | 0.019 | 0.000 | 1.000 | 0.000 |
| viterbi | overall | 578 | 0.998 | 7.72 | 0.416 | 0.030 | 0.008 | 0.989 | 0.310 |
| viterbi | seen | 377 | 0.997 | 7.63 | 0.414 | 0.035 | 0.010 | 0.984 | 0.312 |
| viterbi | held_out | 289 | 1.000 | 7.76 | 0.417 | 0.019 | 0.003 | 0.998 | 0.268 |

## Versus the linear baselines

| model | sent F1(AI) | bound F1 exact | held-out F1(AI) |
|---|---|---|---|
| baseline: all-human | 0.000 | 0.000 | 0.000 |
| baseline: all-AI | 0.542 | 0.000 | 0.536 |
| baseline: random (train prior) | 0.367 | 0.357 | 0.341 |
| baseline: position only (no text) | 0.567 | 0.284 | 0.550 |
| text only | 0.675 | 0.482 | 0.575 |
| text + context | 0.678 | 0.438 | 0.571 |
| text + context + smoothing | 0.651 | 0.314 | 0.538 |
| text + position (diagnostic) | 0.696 | 0.474 | 0.606 |
| **xlm-roberta-base** | **0.413** | **0.459** | **0.423** |

## By generator

| generator | role | docs | sent F1(AI) | bound F1 exact |
|---|---|---|---|---|
| deepseek_v3 | seen | 302 | 0.542 | 0.000 |
| gemini_2_5_pro | held_out | 305 | 0.536 | 0.000 |
| gpt_4o | seen | 311 | 0.548 | 0.000 |

## By construction type

| construction | docs | sent F1(AI) | bound F1 exact |
|---|---|---|---|
| type1_single_boundary | 368 | 0.673 | 0.000 |
| type2_single_internal_segment | 299 | 0.393 | 0.000 |
| type3_multiple_internal_segments | 251 | 0.494 | 0.000 |
