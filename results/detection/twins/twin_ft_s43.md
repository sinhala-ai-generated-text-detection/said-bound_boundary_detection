# Transformer detector — XLM-RoBERTa sentence tagger

`xlm-roberta-base` fine-tuned for per-sentence human/AI tagging: 3 epochs, lr 1e-05, effective batch 4, best epoch 2, 40.3 min on GPU. Trained on 2490 documents.

The whole document is encoded in one pass with a marker token before each sentence, so each sentence representation is built with its neighbours in attention range.

**Structured decoding.** Decoding is a Viterbi pass over the sentence scores with a single transition cost for changing author, tuned on dev-seen for **exact-boundary F1**. This is a linear-chain CRF decode with a constant transition. It matters because the previous setup tuned a per-sentence threshold for *sentence* F1 and then reported *boundary* F1 - optimising a different objective from the one reported. Unlike run-length smoothing it never forbids a one-sentence span; it only makes one cost two transitions.

Threshold 0.375 (sentence F1) and boundary bias +2.75 (exact-boundary F1), both tuned on dev restricted to seen generators.

## Test results

| slice | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| overall | 0.835 | 0.787 | 0.758 | 0.818 | 0.599 | 0.742 | 0.278 |
| seen generators | 0.853 | 0.815 | 0.774 | 0.861 | 0.621 | 0.756 | 0.330 |
| held-out (Gemini) | 0.799 | 0.727 | 0.724 | 0.731 | 0.558 | 0.717 | 0.174 |

### Decoder comparison (overall test)

| slice | sent acc | sent F1(AI) | P(AI) | R(AI) | bound F1 exact | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| threshold (per-sentence) | 0.846 | 0.796 | 0.786 | 0.806 | 0.570 | 0.742 | 0.330 |
| **viterbi (pair head)** | 0.835 | 0.787 | 0.758 | 0.818 | 0.599 | 0.742 | 0.278 |

## All-human twins

Each test document is also scored as its all-human source window: same topic, length and positions, no machine text. *False alarm* is the share of twins with any sentence flagged; *win* is how often an AI sentence scores above the human sentence it replaced; *flip* is how often it is flagged while that human sentence is not; *ctx stable* is how often an untouched human sentence gets the same label in both documents; *bE + twins* is exact-boundary F1 over the mixed documents and the twins together.

Trained with counterfactual twins: margin weight 1.0, consistency weight 1.0, margin 2.0.

| decoder | slice | twins | false alarm | boundaries/twin | sentence FPR | win | flip | ctx stable | bE + twins |
|---|---|---|---|---|---|---|---|---|---|
| threshold | overall | 578 | 0.590 | 1.61 | 0.142 | 0.933 | 0.657 | 0.938 | 0.468 |
| threshold | seen | 377 | 0.599 | 1.63 | 0.145 | 0.951 | 0.696 | 0.936 | 0.488 |
| threshold | held_out | 289 | 0.561 | 1.52 | 0.135 | 0.896 | 0.579 | 0.940 | 0.403 |
| viterbi | overall | 578 | 0.818 | 3.01 | 0.198 | 0.933 | 0.612 | 0.928 | 0.446 |
| viterbi | seen | 377 | 0.841 | 3.09 | 0.201 | 0.951 | 0.649 | 0.927 | 0.459 |
| viterbi | held_out | 289 | 0.799 | 2.91 | 0.196 | 0.896 | 0.537 | 0.930 | 0.378 |

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
| **xlm-roberta-base** | **0.787** | **0.599** | **0.727** |

## By generator

| generator | role | docs | sent F1(AI) | bound F1 exact |
|---|---|---|---|---|
| deepseek_v3 | seen | 302 | 0.807 | 0.585 |
| gemini_2_5_pro | held_out | 305 | 0.735 | 0.524 |
| gpt_4o | seen | 311 | 0.841 | 0.604 |

## By construction type

| construction | docs | sent F1(AI) | bound F1 exact |
|---|---|---|---|
| type1_single_boundary | 368 | 0.916 | 0.605 |
| type2_single_internal_segment | 299 | 0.686 | 0.512 |
| type3_multiple_internal_segments | 251 | 0.647 | 0.592 |
