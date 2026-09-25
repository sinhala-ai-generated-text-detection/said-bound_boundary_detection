# Transformer detector — XLM-RoBERTa sentence tagger

`xlm-roberta-base` fine-tuned for per-sentence human/AI tagging: 8 epochs, lr 3e-05, effective batch 4, best epoch 6, 158.6 min on GPU. Trained on 2490 documents.

The whole document is encoded in one pass with a marker token before each sentence, so each sentence representation is built with its neighbours in attention range.

**Structured decoding.** Decoding is a Viterbi pass over the sentence scores with a single transition cost for changing author, tuned on dev-seen for **exact-boundary F1**. This is a linear-chain CRF decode with a constant transition. It matters because the previous setup tuned a per-sentence threshold for *sentence* F1 and then reported *boundary* F1 - optimising a different objective from the one reported. Unlike run-length smoothing it never forbids a one-sentence span; it only makes one cost two transitions.

Threshold 0.200 (sentence F1) and boundary bias +3.50 (exact-boundary F1), both tuned on dev restricted to seen generators.

## Test results

| slice | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| overall | 0.821 | 0.770 | 0.737 | 0.806 | 0.595 | 0.725 | 0.256 |
| seen generators | 0.843 | 0.804 | 0.758 | 0.856 | 0.632 | 0.754 | 0.316 |
| held-out (Gemini) | 0.776 | 0.697 | 0.690 | 0.704 | 0.527 | 0.671 | 0.134 |

### Decoder comparison (overall test)

| slice | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| threshold (per-sentence) | 0.839 | 0.785 | 0.778 | 0.793 | 0.553 | 0.726 | 0.317 |
| **viterbi (pair head)** | 0.821 | 0.770 | 0.737 | 0.806 | 0.595 | 0.725 | 0.256 |

## All-human twins

Each test document is also scored as its all-human source window: same topic, length and positions, no machine text. *False alarm* is the share of twins with any sentence flagged; *win* is how often an AI sentence scores above the human sentence it replaced; *flip* is how often it is flagged while that human sentence is not; *ctx stable* is how often an untouched human sentence gets the same label in both documents; *bE + twins* is exact-boundary F1 over the mixed documents and the twins together.

Trained with counterfactual twins: margin weight 0.0, consistency weight 0.0, margin 2.0.

| decoder | slice | twins | false alarm | boundaries/twin | sentence FPR | win | flip | ctx stable | bE + twins |
|---|---|---|---|---|---|---|---|---|---|
| threshold | overall | 578 | 0.587 | 1.72 | 0.157 | 0.921 | 0.634 | 0.931 | 0.448 |
| threshold | seen | 377 | 0.578 | 1.71 | 0.156 | 0.944 | 0.678 | 0.925 | 0.481 |
| threshold | held_out | 289 | 0.581 | 1.65 | 0.150 | 0.876 | 0.545 | 0.943 | 0.357 |
| viterbi | overall | 578 | 0.801 | 3.59 | 0.226 | 0.921 | 0.582 | 0.906 | 0.434 |
| viterbi | seen | 377 | 0.796 | 3.66 | 0.230 | 0.944 | 0.615 | 0.904 | 0.458 |
| viterbi | held_out | 289 | 0.806 | 3.46 | 0.218 | 0.876 | 0.514 | 0.908 | 0.350 |

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
| **xlm-roberta-base** | **0.770** | **0.595** | **0.697** |

## By generator

| generator | role | docs | sent F1(AI) | bound F1 exact |
|---|---|---|---|---|
| deepseek_v3 | seen | 302 | 0.802 | 0.584 |
| gemini_2_5_pro | held_out | 305 | 0.711 | 0.477 |
| gpt_4o | seen | 311 | 0.835 | 0.597 |

## By construction type

| construction | docs | sent F1(AI) | bound F1 exact |
|---|---|---|---|
| type1_single_boundary | 368 | 0.916 | 0.588 |
| type2_single_internal_segment | 299 | 0.640 | 0.496 |
| type3_multiple_internal_segments | 251 | 0.639 | 0.574 |
