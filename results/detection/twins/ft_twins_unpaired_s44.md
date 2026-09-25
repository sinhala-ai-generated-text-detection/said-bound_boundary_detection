# Transformer detector — XLM-RoBERTa sentence tagger

`xlm-roberta-base` fine-tuned for per-sentence human/AI tagging: 3 epochs, lr 1e-05, effective batch 4, best epoch 1, 11.7 min on GPU. Trained on 2490 documents.

The whole document is encoded in one pass with a marker token before each sentence, so each sentence representation is built with its neighbours in attention range.

**Structured decoding.** Decoding is a Viterbi pass over the sentence scores with a single transition cost for changing author, tuned on dev-seen for **exact-boundary F1**. This is a linear-chain CRF decode with a constant transition. It matters because the previous setup tuned a per-sentence threshold for *sentence* F1 and then reported *boundary* F1 - optimising a different objective from the one reported. Unlike run-length smoothing it never forbids a one-sentence span; it only makes one cost two transitions.

Threshold 0.275 (sentence F1) and boundary bias +2.00 (exact-boundary F1), both tuned on dev restricted to seen generators.

## Test results

| slice | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| overall | 0.837 | 0.788 | 0.765 | 0.812 | 0.614 | 0.748 | 0.316 |
| seen generators | 0.856 | 0.817 | 0.782 | 0.854 | 0.643 | 0.771 | 0.385 |
| held-out (Gemini) | 0.800 | 0.727 | 0.728 | 0.726 | 0.562 | 0.705 | 0.177 |

### Decoder comparison (overall test)

| slice | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| threshold (per-sentence) | 0.847 | 0.797 | 0.788 | 0.807 | 0.576 | 0.747 | 0.331 |
| **viterbi (pair head)** | 0.837 | 0.788 | 0.765 | 0.812 | 0.614 | 0.748 | 0.316 |

## All-human twins

Each test document is also scored as its all-human source window: same topic, length and positions, no machine text. *False alarm* is the share of twins with any sentence flagged; *win* is how often an AI sentence scores above the human sentence it replaced; *flip* is how often it is flagged while that human sentence is not; *ctx stable* is how often an untouched human sentence gets the same label in both documents; *bE + twins* is exact-boundary F1 over the mixed documents and the twins together.

Trained with counterfactual twins: margin weight 0.0, consistency weight 0.0, margin 2.0.

| decoder | slice | twins | false alarm | boundaries/twin | sentence FPR | win | flip | ctx stable | bE + twins |
|---|---|---|---|---|---|---|---|---|---|
| threshold | overall | 578 | 0.678 | 1.89 | 0.156 | 0.921 | 0.637 | 0.940 | 0.460 |
| threshold | seen | 377 | 0.679 | 1.86 | 0.154 | 0.947 | 0.676 | 0.936 | 0.480 |
| threshold | held_out | 289 | 0.657 | 1.84 | 0.153 | 0.870 | 0.556 | 0.948 | 0.391 |
| viterbi | overall | 578 | 0.903 | 3.66 | 0.225 | 0.921 | 0.582 | 0.904 | 0.437 |
| viterbi | seen | 377 | 0.899 | 3.67 | 0.227 | 0.947 | 0.617 | 0.901 | 0.456 |
| viterbi | held_out | 289 | 0.920 | 3.68 | 0.227 | 0.870 | 0.510 | 0.910 | 0.356 |

## Unrelated human documents

All 618 source windows of test-split articles that no mixed document uses: human text with no content link to any training or test document.

| decoder | docs | false alarm | boundaries/doc | sentence FPR |
|---|---|---|---|---|
| threshold | 618 | 0.650 | 1.68 | 0.140 |
| viterbi | 618 | 0.887 | 3.26 | 0.207 |

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
| **xlm-roberta-base** | **0.788** | **0.614** | **0.727** |

## By generator

| generator | role | docs | sent F1(AI) | bound F1 exact |
|---|---|---|---|---|
| deepseek_v3 | seen | 302 | 0.803 | 0.584 |
| gemini_2_5_pro | held_out | 305 | 0.741 | 0.532 |
| gpt_4o | seen | 311 | 0.842 | 0.613 |

## By construction type

| construction | docs | sent F1(AI) | bound F1 exact |
|---|---|---|---|
| type1_single_boundary | 368 | 0.914 | 0.606 |
| type2_single_internal_segment | 299 | 0.683 | 0.519 |
| type3_multiple_internal_segments | 251 | 0.656 | 0.599 |
