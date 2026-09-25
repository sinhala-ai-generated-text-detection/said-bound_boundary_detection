# Transformer detector — XLM-RoBERTa sentence tagger

`xlm-roberta-base` fine-tuned for per-sentence human/AI tagging: 8 epochs, lr 3e-05, effective batch 4, best epoch 5, 72.0 min on GPU. Trained on 2490 documents.

The whole document is encoded in one pass with a marker token before each sentence, so each sentence representation is built with its neighbours in attention range.

**Structured decoding.** Decoding is a Viterbi pass over the sentence scores with a single transition cost for changing author, tuned on dev-seen for **exact-boundary F1**. This is a linear-chain CRF decode with a constant transition. It matters because the previous setup tuned a per-sentence threshold for *sentence* F1 and then reported *boundary* F1 - optimising a different objective from the one reported. Unlike run-length smoothing it never forbids a one-sentence span; it only makes one cost two transitions.

Threshold 0.650 (sentence F1) and boundary bias +1.25 (exact-boundary F1), both tuned on dev restricted to seen generators.

## Test results

| slice | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| overall | 0.822 | 0.771 | 0.742 | 0.802 | 0.580 | 0.717 | 0.255 |
| seen generators | 0.841 | 0.801 | 0.754 | 0.854 | 0.609 | 0.733 | 0.312 |
| held-out (Gemini) | 0.786 | 0.704 | 0.712 | 0.696 | 0.524 | 0.687 | 0.141 |

### Decoder comparison (overall test)

| slice | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| threshold (per-sentence) | 0.840 | 0.776 | 0.808 | 0.747 | 0.543 | 0.705 | 0.313 |
| **viterbi (pair head)** | 0.822 | 0.771 | 0.742 | 0.802 | 0.580 | 0.717 | 0.255 |

## All-human twins

Each test document is also scored as its all-human source window: same topic, length and positions, no machine text. *False alarm* is the share of twins with any sentence flagged; *win* is how often an AI sentence scores above the human sentence it replaced; *flip* is how often it is flagged while that human sentence is not; *ctx stable* is how often an untouched human sentence gets the same label in both documents; *bE + twins* is exact-boundary F1 over the mixed documents and the twins together.

Trained with counterfactual twins: margin weight 1.0, consistency weight 1.0, margin 2.0.

| decoder | slice | twins | false alarm | boundaries/twin | sentence FPR | win | flip | ctx stable | bE + twins |
|---|---|---|---|---|---|---|---|---|---|
| threshold | overall | 578 | 0.490 | 1.19 | 0.100 | 0.921 | 0.635 | 0.956 | 0.464 |
| threshold | seen | 377 | 0.491 | 1.21 | 0.099 | 0.944 | 0.694 | 0.950 | 0.492 |
| threshold | held_out | 289 | 0.491 | 1.15 | 0.097 | 0.875 | 0.516 | 0.966 | 0.381 |
| viterbi | overall | 578 | 0.803 | 2.97 | 0.199 | 0.921 | 0.599 | 0.949 | 0.435 |
| viterbi | seen | 377 | 0.806 | 3.05 | 0.201 | 0.944 | 0.642 | 0.947 | 0.454 |
| viterbi | held_out | 289 | 0.792 | 2.80 | 0.190 | 0.875 | 0.509 | 0.953 | 0.361 |

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
| **xlm-roberta-base** | **0.771** | **0.580** | **0.704** |

## By generator

| generator | role | docs | sent F1(AI) | bound F1 exact |
|---|---|---|---|---|
| deepseek_v3 | seen | 302 | 0.786 | 0.553 |
| gemini_2_5_pro | held_out | 305 | 0.698 | 0.476 |
| gpt_4o | seen | 311 | 0.834 | 0.600 |

## By construction type

| construction | docs | sent F1(AI) | bound F1 exact |
|---|---|---|---|
| type1_single_boundary | 368 | 0.908 | 0.597 |
| type2_single_internal_segment | 299 | 0.651 | 0.519 |
| type3_multiple_internal_segments | 251 | 0.596 | 0.534 |
