# Transformer detector — XLM-RoBERTa sentence tagger

`xlm-roberta-base` fine-tuned for per-sentence human/AI tagging: 3 epochs, lr 1e-05, effective batch 4, best epoch 2, 30.2 min on GPU. Trained on 2490 documents.

The whole document is encoded in one pass with a marker token before each sentence, so each sentence representation is built with its neighbours in attention range.

**Structured decoding.** Decoding is a Viterbi pass over the sentence scores with a single transition cost for changing author, tuned on dev-seen for **exact-boundary F1**. This is a linear-chain CRF decode with a constant transition. It matters because the previous setup tuned a per-sentence threshold for *sentence* F1 and then reported *boundary* F1 - optimising a different objective from the one reported. Unlike run-length smoothing it never forbids a one-sentence span; it only makes one cost two transitions.

Threshold 0.550 (sentence F1) and boundary bias +3.50 (exact-boundary F1), both tuned on dev restricted to seen generators.

## Test results

| slice | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| overall | 0.835 | 0.789 | 0.751 | 0.832 | 0.616 | 0.751 | 0.319 |
| seen generators | 0.857 | 0.821 | 0.776 | 0.871 | 0.650 | 0.778 | 0.383 |
| held-out (Gemini) | 0.790 | 0.724 | 0.698 | 0.751 | 0.555 | 0.703 | 0.190 |

### Decoder comparison (overall test)

| slice | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| threshold (per-sentence) | 0.849 | 0.803 | 0.781 | 0.827 | 0.608 | 0.777 | 0.368 |
| **viterbi (pair head)** | 0.835 | 0.789 | 0.751 | 0.832 | 0.616 | 0.751 | 0.319 |

## All-human twins

Each test document is also scored as its all-human source window: same topic, length and positions, no machine text. *False alarm* is the share of twins with any sentence flagged; *win* is how often an AI sentence scores above the human sentence it replaced; *flip* is how often it is flagged while that human sentence is not; *ctx stable* is how often an untouched human sentence gets the same label in both documents; *bE + twins* is exact-boundary F1 over the mixed documents and the twins together.

| decoder | slice | twins | false alarm | boundaries/twin | sentence FPR | win | flip | ctx stable | bE + twins |
|---|---|---|---|---|---|---|---|---|---|
| threshold | overall | 578 | 0.941 | 2.98 | 0.246 | 0.873 | 0.564 | 0.891 | 0.441 |
| threshold | seen | 377 | 0.931 | 2.96 | 0.246 | 0.904 | 0.602 | 0.889 | 0.462 |
| threshold | held_out | 289 | 0.955 | 2.99 | 0.248 | 0.812 | 0.487 | 0.895 | 0.359 |
| viterbi | overall | 578 | 0.998 | 4.78 | 0.325 | 0.873 | 0.500 | 0.840 | 0.405 |
| viterbi | seen | 377 | 0.997 | 4.84 | 0.327 | 0.904 | 0.532 | 0.828 | 0.423 |
| viterbi | held_out | 289 | 1.000 | 4.74 | 0.323 | 0.812 | 0.437 | 0.866 | 0.322 |

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
| likelihood only (no text) | 0.606 | 0.323 | 0.588 |
| text + context + likelihood | 0.704 | 0.450 | 0.615 |
| **xlm-roberta-base** | **0.789** | **0.616** | **0.724** |

## By generator

| generator | role | docs | sent F1(AI) | bound F1 exact |
|---|---|---|---|---|
| deepseek_v3 | seen | 302 | 0.814 | 0.620 |
| gemini_2_5_pro | held_out | 305 | 0.746 | 0.556 |
| gpt_4o | seen | 311 | 0.847 | 0.652 |

## By construction type

| construction | docs | sent F1(AI) | bound F1 exact |
|---|---|---|---|
| type1_single_boundary | 368 | 0.926 | 0.639 |
| type2_single_internal_segment | 299 | 0.696 | 0.554 |
| type3_multiple_internal_segments | 251 | 0.656 | 0.631 |
