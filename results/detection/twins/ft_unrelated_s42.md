# Transformer detector — XLM-RoBERTa sentence tagger

`xlm-roberta-base` fine-tuned for per-sentence human/AI tagging: 3 epochs, lr 1e-05, effective batch 4, best epoch 3, 12.4 min on GPU. Trained on 2490 documents.

The whole document is encoded in one pass with a marker token before each sentence, so each sentence representation is built with its neighbours in attention range.

**Structured decoding.** Decoding is a Viterbi pass over the sentence scores with a single transition cost for changing author, tuned on dev-seen for **exact-boundary F1**. This is a linear-chain CRF decode with a constant transition. It matters because the previous setup tuned a per-sentence threshold for *sentence* F1 and then reported *boundary* F1 - optimising a different objective from the one reported. Unlike run-length smoothing it never forbids a one-sentence span; it only makes one cost two transitions.

Threshold 0.225 (sentence F1) and boundary bias +4.00 (exact-boundary F1), both tuned on dev restricted to seen generators.

## Test results

| slice | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| overall | 0.835 | 0.787 | 0.757 | 0.819 | 0.612 | 0.745 | 0.313 |
| seen generators | 0.855 | 0.817 | 0.777 | 0.861 | 0.647 | 0.775 | 0.380 |
| held-out (Gemini) | 0.794 | 0.723 | 0.712 | 0.734 | 0.548 | 0.690 | 0.177 |

### Decoder comparison (overall test)

| slice | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| threshold (per-sentence) | 0.842 | 0.785 | 0.795 | 0.776 | 0.556 | 0.721 | 0.325 |
| **viterbi (pair head)** | 0.835 | 0.787 | 0.757 | 0.819 | 0.612 | 0.745 | 0.313 |

## All-human twins

Each test document is also scored as its all-human source window: same topic, length and positions, no machine text. *False alarm* is the share of twins with any sentence flagged; *win* is how often an AI sentence scores above the human sentence it replaced; *flip* is how often it is flagged while that human sentence is not; *ctx stable* is how often an untouched human sentence gets the same label in both documents; *bE + twins* is exact-boundary F1 over the mixed documents and the twins together.

Trained with unrelated same-length human windows: margin weight 0.0, consistency weight 0.0, margin 2.0.

| decoder | slice | twins | false alarm | boundaries/twin | sentence FPR | win | flip | ctx stable | bE + twins |
|---|---|---|---|---|---|---|---|---|---|
| threshold | overall | 578 | 0.526 | 1.48 | 0.126 | 0.908 | 0.646 | 0.926 | 0.461 |
| threshold | seen | 377 | 0.528 | 1.46 | 0.124 | 0.936 | 0.691 | 0.924 | 0.484 |
| threshold | held_out | 289 | 0.509 | 1.46 | 0.126 | 0.853 | 0.554 | 0.930 | 0.388 |
| viterbi | overall | 578 | 0.917 | 4.03 | 0.245 | 0.908 | 0.575 | 0.874 | 0.426 |
| viterbi | seen | 377 | 0.902 | 4.00 | 0.242 | 0.936 | 0.611 | 0.872 | 0.450 |
| viterbi | held_out | 289 | 0.941 | 4.08 | 0.250 | 0.853 | 0.503 | 0.876 | 0.338 |

## Unrelated human documents

All 618 source windows of test-split articles that no mixed document uses: human text with no content link to any training or test document.

| decoder | docs | false alarm | boundaries/doc | sentence FPR |
|---|---|---|---|---|
| threshold | 618 | 0.505 | 1.30 | 0.112 |
| viterbi | 618 | 0.893 | 3.61 | 0.228 |

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
| **xlm-roberta-base** | **0.787** | **0.612** | **0.723** |

## By generator

| generator | role | docs | sent F1(AI) | bound F1 exact |
|---|---|---|---|---|
| deepseek_v3 | seen | 302 | 0.801 | 0.573 |
| gemini_2_5_pro | held_out | 305 | 0.723 | 0.507 |
| gpt_4o | seen | 311 | 0.825 | 0.587 |

## By construction type

| construction | docs | sent F1(AI) | bound F1 exact |
|---|---|---|---|
| type1_single_boundary | 368 | 0.917 | 0.604 |
| type2_single_internal_segment | 299 | 0.659 | 0.516 |
| type3_multiple_internal_segments | 251 | 0.611 | 0.561 |
