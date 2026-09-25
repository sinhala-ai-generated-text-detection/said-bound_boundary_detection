# Transformer detector — XLM-RoBERTa sentence tagger

`xlm-roberta-base` fine-tuned for per-sentence human/AI tagging: 12 epochs, lr 3e-05, effective batch 4, best epoch -1, 0.0 min on GPU. Trained on 2490 documents.

The whole document is encoded in one pass with a marker token before each sentence, so each sentence representation is built with its neighbours in attention range.

**Structured decoding.** Decoding is a Viterbi pass over the sentence scores with a single transition cost for changing author, tuned on dev-seen for **exact-boundary F1**. This is a linear-chain CRF decode with a constant transition. It matters because the previous setup tuned a per-sentence threshold for *sentence* F1 and then reported *boundary* F1 - optimising a different objective from the one reported. Unlike run-length smoothing it never forbids a one-sentence span; it only makes one cost two transitions.

Threshold 0.200 (sentence F1) and boundary bias +2.75 (exact-boundary F1), both tuned on dev restricted to seen generators.

**Surface-normalized.** Every sentence of every document (both classes, dev and test, twins and human documents) had whitespace before punctuation, zero-width characters other than ZWJ and soft hyphens removed, curly quotes straightened and a final full stop added where missing (`data.normalize_surface`). Parentheses and Latin text are content and were left alone. Threshold and bias were re-tuned on the normalized dev set.

## Test results

| slice | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| overall | 0.841 | 0.787 | 0.787 | 0.787 | 0.603 | 0.753 | 0.294 |
| seen generators | 0.861 | 0.819 | 0.800 | 0.838 | 0.637 | 0.775 | 0.364 |
| held-out (Gemini) | 0.803 | 0.717 | 0.756 | 0.682 | 0.539 | 0.711 | 0.154 |

### Decoder comparison (overall test)

| slice | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| threshold (per-sentence) | 0.847 | 0.791 | 0.803 | 0.779 | 0.571 | 0.739 | 0.326 |
| **viterbi (pair head)** | 0.841 | 0.787 | 0.787 | 0.787 | 0.603 | 0.753 | 0.294 |

## All-human twins

Each test document is also scored as its all-human source window: same topic, length and positions, no machine text. *False alarm* is the share of twins with any sentence flagged; *win* is how often an AI sentence scores above the human sentence it replaced; *flip* is how often it is flagged while that human sentence is not; *ctx stable* is how often an untouched human sentence gets the same label in both documents; *bE + twins* is exact-boundary F1 over the mixed documents and the twins together.

| decoder | slice | twins | false alarm | boundaries/twin | sentence FPR | win | flip | ctx stable | bE + twins |
|---|---|---|---|---|---|---|---|---|---|
| threshold | overall | 578 | 0.561 | 1.57 | 0.129 | 0.921 | 0.644 | 0.944 | 0.471 |
| threshold | seen | 377 | 0.573 | 1.60 | 0.130 | 0.943 | 0.690 | 0.939 | 0.498 |
| threshold | held_out | 289 | 0.540 | 1.48 | 0.122 | 0.878 | 0.551 | 0.952 | 0.389 |
| viterbi | overall | 578 | 0.758 | 2.63 | 0.174 | 0.921 | 0.607 | 0.936 | 0.460 |
| viterbi | seen | 377 | 0.775 | 2.69 | 0.175 | 0.943 | 0.651 | 0.930 | 0.484 |
| viterbi | held_out | 289 | 0.723 | 2.45 | 0.164 | 0.878 | 0.516 | 0.948 | 0.379 |

## Unrelated human documents

All 618 source windows of test-split articles that no mixed document uses: human text with no content link to any training or test document.

| decoder | docs | false alarm | boundaries/doc | sentence FPR |
|---|---|---|---|---|
| threshold | 618 | 0.561 | 1.42 | 0.118 |
| viterbi | 618 | 0.756 | 2.39 | 0.161 |

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
| **xlm-roberta-base** | **0.787** | **0.603** | **0.717** |

## By generator

| generator | role | docs | sent F1(AI) | bound F1 exact |
|---|---|---|---|---|
| deepseek_v3 | seen | 302 | 0.809 | 0.596 |
| gemini_2_5_pro | held_out | 305 | 0.717 | 0.505 |
| gpt_4o | seen | 311 | 0.839 | 0.614 |

## By construction type

| construction | docs | sent F1(AI) | bound F1 exact |
|---|---|---|---|
| type1_single_boundary | 368 | 0.913 | 0.608 |
| type2_single_internal_segment | 299 | 0.678 | 0.511 |
| type3_multiple_internal_segments | 251 | 0.631 | 0.593 |
