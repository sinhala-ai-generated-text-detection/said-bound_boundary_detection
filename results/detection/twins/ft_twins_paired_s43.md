# Transformer detector — XLM-RoBERTa sentence tagger

`xlm-roberta-base` fine-tuned for per-sentence human/AI tagging: 3 epochs, lr 1e-05, effective batch 4, best epoch 1, 11.7 min on GPU. Trained on 2490 documents.

The whole document is encoded in one pass with a marker token before each sentence, so each sentence representation is built with its neighbours in attention range.

**Structured decoding.** Decoding is a Viterbi pass over the sentence scores with a single transition cost for changing author, tuned on dev-seen for **exact-boundary F1**. This is a linear-chain CRF decode with a constant transition. It matters because the previous setup tuned a per-sentence threshold for *sentence* F1 and then reported *boundary* F1 - optimising a different objective from the one reported. Unlike run-length smoothing it never forbids a one-sentence span; it only makes one cost two transitions.

Threshold 0.525 (sentence F1) and boundary bias +2.00 (exact-boundary F1), both tuned on dev restricted to seen generators.

## Test results

| slice | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| overall | 0.835 | 0.787 | 0.758 | 0.818 | 0.603 | 0.748 | 0.279 |
| seen generators | 0.853 | 0.815 | 0.774 | 0.860 | 0.627 | 0.766 | 0.328 |
| held-out (Gemini) | 0.800 | 0.729 | 0.723 | 0.734 | 0.559 | 0.715 | 0.180 |

### Decoder comparison (overall test)

| slice | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| threshold (per-sentence) | 0.843 | 0.790 | 0.787 | 0.794 | 0.556 | 0.729 | 0.310 |
| **viterbi (pair head)** | 0.835 | 0.787 | 0.758 | 0.818 | 0.603 | 0.748 | 0.279 |

## All-human twins

Each test document is also scored as its all-human source window: same topic, length and positions, no machine text. *False alarm* is the share of twins with any sentence flagged; *win* is how often an AI sentence scores above the human sentence it replaced; *flip* is how often it is flagged while that human sentence is not; *ctx stable* is how often an untouched human sentence gets the same label in both documents; *bE + twins* is exact-boundary F1 over the mixed documents and the twins together.

Trained with counterfactual twins: margin weight 1.0, consistency weight 1.0, margin 2.0.

| decoder | slice | twins | false alarm | boundaries/twin | sentence FPR | win | flip | ctx stable | bE + twins |
|---|---|---|---|---|---|---|---|---|---|
| threshold | overall | 578 | 0.580 | 1.51 | 0.134 | 0.935 | 0.653 | 0.949 | 0.460 |
| threshold | seen | 377 | 0.581 | 1.51 | 0.132 | 0.955 | 0.696 | 0.950 | 0.477 |
| threshold | held_out | 289 | 0.571 | 1.44 | 0.131 | 0.894 | 0.565 | 0.948 | 0.403 |
| viterbi | overall | 578 | 0.829 | 3.01 | 0.198 | 0.935 | 0.607 | 0.931 | 0.450 |
| viterbi | seen | 377 | 0.830 | 3.07 | 0.200 | 0.955 | 0.645 | 0.929 | 0.465 |
| viterbi | held_out | 289 | 0.820 | 2.83 | 0.193 | 0.894 | 0.529 | 0.934 | 0.383 |

## Unrelated human documents

All 618 source windows of test-split articles that no mixed document uses: human text with no content link to any training or test document.

| decoder | docs | false alarm | boundaries/doc | sentence FPR |
|---|---|---|---|---|
| threshold | 618 | 0.573 | 1.41 | 0.120 |
| viterbi | 618 | 0.799 | 2.68 | 0.180 |

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
| **xlm-roberta-base** | **0.787** | **0.603** | **0.729** |

## By generator

| generator | role | docs | sent F1(AI) | bound F1 exact |
|---|---|---|---|---|
| deepseek_v3 | seen | 302 | 0.800 | 0.557 |
| gemini_2_5_pro | held_out | 305 | 0.738 | 0.520 |
| gpt_4o | seen | 311 | 0.828 | 0.593 |

## By construction type

| construction | docs | sent F1(AI) | bound F1 exact |
|---|---|---|---|
| type1_single_boundary | 368 | 0.913 | 0.579 |
| type2_single_internal_segment | 299 | 0.675 | 0.515 |
| type3_multiple_internal_segments | 251 | 0.628 | 0.572 |
