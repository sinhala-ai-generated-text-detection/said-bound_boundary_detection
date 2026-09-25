# Transformer detector — XLM-RoBERTa sentence tagger

`xlm-roberta-base` fine-tuned for per-sentence human/AI tagging: 12 epochs, lr 3e-05, effective batch 4, best epoch -1, 0.0 min on GPU. Trained on 2490 documents.

The whole document is encoded in one pass with a marker token before each sentence, so each sentence representation is built with its neighbours in attention range.

**Structured decoding.** Decoding is a Viterbi pass over the sentence scores with a single transition cost for changing author, tuned on dev-seen for **exact-boundary F1**. This is a linear-chain CRF decode with a constant transition. It matters because the previous setup tuned a per-sentence threshold for *sentence* F1 and then reported *boundary* F1 - optimising a different objective from the one reported. Unlike run-length smoothing it never forbids a one-sentence span; it only makes one cost two transitions.

Threshold 0.325 (sentence F1) and boundary bias +3.50 (exact-boundary F1), both tuned on dev restricted to seen generators.

**Surface-normalized.** Every sentence of every document (both classes, dev and test, twins and human documents) had whitespace before punctuation, zero-width characters other than ZWJ and soft hyphens removed, curly quotes straightened and a final full stop added where missing (`data.normalize_surface`). Parentheses and Latin text are content and were left alone. Threshold and bias were re-tuned on the normalized dev set.

## Test results

| slice | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| overall | 0.821 | 0.775 | 0.728 | 0.828 | 0.609 | 0.727 | 0.288 |
| seen generators | 0.844 | 0.807 | 0.754 | 0.868 | 0.642 | 0.753 | 0.357 |
| held-out (Gemini) | 0.774 | 0.708 | 0.673 | 0.747 | 0.547 | 0.680 | 0.148 |

### Decoder comparison (overall test)

| slice | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| threshold (per-sentence) | 0.841 | 0.796 | 0.763 | 0.832 | 0.601 | 0.766 | 0.337 |
| **viterbi (pair head)** | 0.821 | 0.775 | 0.728 | 0.828 | 0.609 | 0.727 | 0.288 |

## All-human twins

Each test document is also scored as its all-human source window: same topic, length and positions, no machine text. *False alarm* is the share of twins with any sentence flagged; *win* is how often an AI sentence scores above the human sentence it replaced; *flip* is how often it is flagged while that human sentence is not; *ctx stable* is how often an untouched human sentence gets the same label in both documents; *bE + twins* is exact-boundary F1 over the mixed documents and the twins together.

| decoder | slice | twins | false alarm | boundaries/twin | sentence FPR | win | flip | ctx stable | bE + twins |
|---|---|---|---|---|---|---|---|---|---|
| threshold | overall | 578 | 0.965 | 3.34 | 0.287 | 0.861 | 0.528 | 0.873 | 0.425 |
| threshold | seen | 377 | 0.960 | 3.30 | 0.286 | 0.890 | 0.561 | 0.866 | 0.445 |
| threshold | held_out | 289 | 0.976 | 3.42 | 0.290 | 0.802 | 0.462 | 0.887 | 0.344 |
| viterbi | overall | 578 | 0.998 | 5.31 | 0.352 | 0.861 | 0.482 | 0.822 | 0.395 |
| viterbi | seen | 377 | 0.997 | 5.35 | 0.352 | 0.890 | 0.513 | 0.808 | 0.414 |
| viterbi | held_out | 289 | 1.000 | 5.23 | 0.352 | 0.802 | 0.420 | 0.850 | 0.314 |

## Unrelated human documents

All 618 source windows of test-split articles that no mixed document uses: human text with no content link to any training or test document.

| decoder | docs | false alarm | boundaries/doc | sentence FPR |
|---|---|---|---|---|
| threshold | 618 | 0.961 | 3.03 | 0.265 |
| viterbi | 618 | 1.000 | 5.05 | 0.341 |

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
| **xlm-roberta-base** | **0.775** | **0.609** | **0.708** |

## By generator

| generator | role | docs | sent F1(AI) | bound F1 exact |
|---|---|---|---|---|
| deepseek_v3 | seen | 302 | 0.810 | 0.624 |
| gemini_2_5_pro | held_out | 305 | 0.734 | 0.552 |
| gpt_4o | seen | 311 | 0.839 | 0.632 |

## By construction type

| construction | docs | sent F1(AI) | bound F1 exact |
|---|---|---|---|
| type1_single_boundary | 368 | 0.921 | 0.626 |
| type2_single_internal_segment | 299 | 0.677 | 0.517 |
| type3_multiple_internal_segments | 251 | 0.658 | 0.647 |
