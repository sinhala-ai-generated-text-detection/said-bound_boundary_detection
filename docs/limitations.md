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
issue: neither class contains non-NFC text in meaningful amounts.

**Measured: the XLM-R taggers barely use these cues.**
`run_transformer.py --eval-only --normalize-surface` re-scores a saved model
after applying `data.normalize_surface` to every sentence of both classes (dev,
test, twins and human documents): whitespace before `. , ? ! : ;` removed,
U+200B, ZWNJ, U+2060, U+FEFF and soft hyphens removed, curly quotes
straightened, and a full stop added where a sentence has no final `. ? ! ෴ …`.
ZWJ is kept, as Sinhala conjuncts need it. Threshold and bias are re-tuned on
the normalized dev set. On the test set this takes the targeted cues to zero
in both classes (space before punctuation 2.81% → 0% of human sentences, no
final punctuation 3.22% → 0%, ZWNJ 0.32% → 0%)
([`surface_noise.md`](../results/detection/surface_noise.md)).

| model (seed 42) | exact-boundary F1, mixed (Viterbi) | sentence F1 (threshold) | unrelated-human false alarm |
|---|---|---|---|
| baseline | 0.604 → 0.602 | 0.802 → 0.800 | 93.5% → 95.0% |
| control fine-tune | 0.614 → 0.609 | 0.798 → 0.796 | 95.0% → 96.1% |
| unrelated human | 0.612 → 0.609 | 0.785 → 0.781 | 50.5% → 49.7% |
| twins, CE only | 0.621 → 0.620 | 0.797 → 0.794 | 57.1% → 57.9% |
| twins + paired terms | 0.603 → 0.603 | 0.794 → 0.791 | 55.5% → 56.1% |

Across the five models, no F1 moves by more than 1.2 points, no document
false-alarm rate by more than 1.5 points, and no sentence false-positive rate
by more than 2.7 points (the baseline). Nearly every change is a small loss,
so the cues account for about a point of these models' scores at most. Three limits
remain: parentheses (12.0% of human vs 2.6% of machine sentences) and Latin
text (12.1% vs 5.6%) are content and were left in, so their imbalance is
untested; the models were trained on raw text, so this measures reliance at
inference rather than what a model trained on normalized text would learn; and
it is one seed per condition. The linear character n-gram models, which see
punctuation directly, were not re-scored.

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

**Seeds.** Twin fine-tuning and its control have three seeds each ([Three seeds](counterfactual-twins.md#three-seeds)),
as do the four fine-tuning conditions that separate human text, content
matching and the paired terms ([Is it the twin, or any human text?](counterfactual-twins.md#is-it-the-twin-or-any-human-text)).
Every other transformer result, including the baseline and the from-scratch
twin models, is a single training run.

**Single domain, single held-out generator.** Wikipedia only. Generalisation is
measured against one unseen model, which cannot separate "generalises to
unseen models" from "generalises to Gemini".

**No human ceiling.** No annotation study establishes how well Sinhala readers
do on this task.
