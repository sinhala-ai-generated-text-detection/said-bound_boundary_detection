# Transformer detector — XLM-RoBERTa sentence tagger

`xlm-roberta-base` fine-tuned for per-sentence human/AI tagging: 3 epochs, lr 1e-05, effective batch 4, best epoch 1, 6.3 min on GPU. Trained on 2490 documents.

The whole document is encoded in one pass with a marker token before each sentence, so each sentence representation is built with its neighbours in attention range.

**Structured decoding.** Decoding is a Viterbi pass over the sentence scores with a single transition cost for changing author, tuned on dev-seen for **exact-boundary F1**. This is a linear-chain CRF decode with a constant transition. It matters because the previous setup tuned a per-sentence threshold for *sentence* F1 and then reported *boundary* F1 - optimising a different objective from the one reported. Unlike run-length smoothing it never forbids a one-sentence span; it only makes one cost two transitions.

Threshold 0.375 (sentence F1) and boundary bias +3.00 (exact-boundary F1), both tuned on dev restricted to seen generators.

## Test results

| slice | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| overall | 0.832 | 0.787 | 0.747 | 0.831 | 0.614 | 0.748 | 0.304 |
| seen generators | 0.854 | 0.816 | 0.772 | 0.866 | 0.647 | 0.774 | 0.377 |
| held-out (Gemini) | 0.789 | 0.725 | 0.695 | 0.759 | 0.555 | 0.702 | 0.157 |

### Decoder comparison (overall test)

| slice | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| threshold (per-sentence) | 0.844 | 0.798 | 0.769 | 0.829 | 0.605 | 0.769 | 0.341 |
| **viterbi (pair head)** | 0.832 | 0.787 | 0.747 | 0.831 | 0.614 | 0.748 | 0.304 |

## All-human twins

Each test document is also scored as its all-human source window: same topic, length and positions, no machine text. *False alarm* is the share of twins with any sentence flagged; *win* is how often an AI sentence scores above the human sentence it replaced; *flip* is how often it is flagged while that human sentence is not; *ctx stable* is how often an untouched human sentence gets the same label in both documents; *bE + twins* is exact-boundary F1 over the mixed documents and the twins together.

| decoder | slice | twins | false alarm | boundaries/twin | sentence FPR | win | flip | ctx stable | bE + twins |
|---|---|---|---|---|---|---|---|---|---|
| threshold | overall | 578 | 0.960 | 3.28 | 0.279 | 0.861 | 0.534 | 0.875 | 0.430 |
| threshold | seen | 377 | 0.955 | 3.24 | 0.280 | 0.891 | 0.569 | 0.869 | 0.451 |
| threshold | held_out | 289 | 0.969 | 3.36 | 0.280 | 0.800 | 0.464 | 0.887 | 0.347 |
| viterbi | overall | 578 | 0.993 | 4.78 | 0.329 | 0.861 | 0.495 | 0.838 | 0.406 |
| viterbi | seen | 377 | 0.989 | 4.80 | 0.330 | 0.891 | 0.524 | 0.824 | 0.424 |
| viterbi | held_out | 289 | 1.000 | 4.73 | 0.331 | 0.800 | 0.434 | 0.866 | 0.325 |

## Unrelated human documents

All 618 source windows of test-split articles that no mixed document uses: human text with no content link to any training or test document.

| decoder | docs | false alarm | boundaries/doc | sentence FPR |
|---|---|---|---|---|
| threshold | 618 | 0.950 | 2.97 | 0.253 |
| viterbi | 618 | 0.992 | 4.46 | 0.314 |

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
| **xlm-roberta-base** | **0.787** | **0.614** | **0.725** |

## By generator

| generator | role | docs | sent F1(AI) | bound F1 exact |
|---|---|---|---|---|
| deepseek_v3 | seen | 302 | 0.815 | 0.634 |
| gemini_2_5_pro | held_out | 305 | 0.736 | 0.554 |
| gpt_4o | seen | 311 | 0.840 | 0.632 |

## By construction type

| construction | docs | sent F1(AI) | bound F1 exact |
|---|---|---|---|
| type1_single_boundary | 368 | 0.922 | 0.627 |
| type2_single_internal_segment | 299 | 0.687 | 0.534 |
| type3_multiple_internal_segments | 251 | 0.655 | 0.643 |
