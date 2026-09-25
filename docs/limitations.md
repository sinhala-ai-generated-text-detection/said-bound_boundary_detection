# Limitations and threats to validity

[README](../README.md) · [Dataset](dataset.md) · [Detectors](detectors.md) · [Likelihood](likelihood.md) · [Counterfactual twins](counterfactual-twins.md) · [Limitations](limitations.md) · [Reproducing](reproducing.md)

**Surface noise in the human text ("cleaner text = AI").** Sinhala Wikipedia
carries typographical noise that generator output lacks. Measured over all
39,358 sentences:

| feature | human sentences | machine sentences |
|---|---|---|
| space before punctuation | 3.55% | 0.01% |
| parentheses | 12.0% | 2.6% |
| no final punctuation | 3.33% | 0.17% |
| Latin characters | 12.1% | 5.6% |
| zero-width non-joiner | 0.52% | 0.01% |

Any sentence with one of the first three is almost certainly human, so a
detector can score partly on formatting. Unicode normalization is **not** the
issue: neither class contains non-NFC text in meaningful amounts. How much of
each detector's score comes from these cues has not yet been measured (for
example, by re-scoring after normalizing them on both sides).

**Positional prior.** Boundaries fall at 20–80% of a document, and spans never
touch the first or last sentence. A position-only baseline reaches 0.284
exact-boundary F1. That is measured and weak, but it is not zero, and the twin
results show the tagger does use position ([What the baseline does on twins](counterfactual-twins.md#what-the-baseline-does-on-twins)).

**Length prior.** The ±25% length gate bounds AI spans, but all generators
still under-write by 3–8 words on average, so a residual length cue may remain.

**Selection protocol.** Epochs and thresholds are chosen on dev *mixed*
documents, which cannot see false alarms. The threshold grid in
`run_transformer.py` (0.20–0.80) was hit at an edge by two runs. Both issues
are quantified in [Operating points: is the cost real?](counterfactual-twins.md#operating-points-is-the-cost-real), but the reported headline numbers still use the
original protocol.

**Seeds.** Twin fine-tuning and its control have three seeds each ([Three seeds](counterfactual-twins.md#three-seeds)).
Every other transformer result, including the baseline and the from-scratch
twin models, is a single training run.

**Single domain, single held-out generator.** Wikipedia only. Generalisation is
measured against one unseen model, which cannot separate "generalises to
unseen models" from "generalises to Gemini".

**No human ceiling.** No annotation study establishes how well Sinhala readers
do on this task.
