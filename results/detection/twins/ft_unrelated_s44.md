# Transformer detector — XLM-RoBERTa sentence tagger

`xlm-roberta-base` fine-tuned for per-sentence human/AI tagging: 3 epochs, lr 1e-05, effective batch 4, best epoch 1, 11.9 min on GPU. Trained on 2490 documents.

The whole document is encoded in one pass with a marker token before each sentence, so each sentence representation is built with its neighbours in attention range.

**Structured decoding.** Decoding is a Viterbi pass over the sentence scores with a single transition cost for changing author, tuned on dev-seen for **exact-boundary F1**. This is a linear-chain CRF decode with a constant transition. It matters because the previous setup tuned a per-sentence threshold for *sentence* F1 and then reported *boundary* F1 - optimising a different objective from the one reported. Unlike run-length smoothing it never forbids a one-sentence span; it only makes one cost two transitions.

Threshold 0.300 (sentence F1) and boundary bias +2.75 (exact-boundary F1), both tuned on dev restricted to seen generators.

## Test results

| slice | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| overall | 0.824 | 0.767 | 0.754 | 0.781 | 0.627 | 0.723 | 0.290 |
| seen generators | 0.849 | 0.806 | 0.779 | 0.835 | 0.661 | 0.756 | 0.359 |
| held-out (Gemini) | 0.773 | 0.684 | 0.697 | 0.671 | 0.567 | 0.665 | 0.151 |

### Decoder comparison (overall test)

| slice | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| threshold (per-sentence) | 0.847 | 0.794 | 0.797 | 0.791 | 0.579 | 0.745 | 0.339 |
| **viterbi (pair head)** | 0.824 | 0.767 | 0.754 | 0.781 | 0.627 | 0.723 | 0.290 |

## All-human twins

Each test document is also scored as its all-human source window: same topic, length and positions, no machine text. *False alarm* is the share of twins with any sentence flagged; *win* is how often an AI sentence scores above the human sentence it replaced; *flip* is how often it is flagged while that human sentence is not; *ctx stable* is how often an untouched human sentence gets the same label in both documents; *bE + twins* is exact-boundary F1 over the mixed documents and the twins together.

Trained with unrelated same-length human windows: margin weight 0.0, consistency weight 0.0, margin 2.0.

| decoder | slice | twins | false alarm | boundaries/twin | sentence FPR | win | flip | ctx stable | bE + twins |
|---|---|---|---|---|---|---|---|---|---|
| threshold | overall | 578 | 0.685 | 1.84 | 0.151 | 0.906 | 0.629 | 0.931 | 0.463 |
| threshold | seen | 377 | 0.706 | 1.88 | 0.154 | 0.935 | 0.671 | 0.929 | 0.480 |
| threshold | held_out | 289 | 0.661 | 1.76 | 0.145 | 0.847 | 0.544 | 0.934 | 0.397 |
| viterbi | overall | 578 | 0.955 | 4.40 | 0.256 | 0.906 | 0.526 | 0.878 | 0.435 |
| viterbi | seen | 377 | 0.950 | 4.42 | 0.258 | 0.935 | 0.571 | 0.873 | 0.455 |
| viterbi | held_out | 289 | 0.965 | 4.42 | 0.258 | 0.847 | 0.435 | 0.886 | 0.351 |

## Unrelated human documents

All 618 source windows of test-split articles that no mixed document uses: human text with no content link to any training or test document.

| decoder | docs | false alarm | boundaries/doc | sentence FPR |
|---|---|---|---|---|
| threshold | 618 | 0.647 | 1.61 | 0.135 |
| viterbi | 618 | 0.947 | 4.14 | 0.248 |

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
| **xlm-roberta-base** | **0.767** | **0.627** | **0.684** |

## By generator

| generator | role | docs | sent F1(AI) | bound F1 exact |
|---|---|---|---|---|
| deepseek_v3 | seen | 302 | 0.808 | 0.595 |
| gemini_2_5_pro | held_out | 305 | 0.731 | 0.537 |
| gpt_4o | seen | 311 | 0.838 | 0.608 |

## By construction type

| construction | docs | sent F1(AI) | bound F1 exact |
|---|---|---|---|
| type1_single_boundary | 368 | 0.918 | 0.624 |
| type2_single_internal_segment | 299 | 0.672 | 0.526 |
| type3_multiple_internal_segments | 251 | 0.644 | 0.595 |
