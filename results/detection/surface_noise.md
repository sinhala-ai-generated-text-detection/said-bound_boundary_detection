# Surface noise: scores with formatting cues removed

Each model is scored twice on the test set: as is, and after `data.normalize_surface` has been applied to every sentence of every document (both classes; dev, test, twins and unrelated human documents). Normalization removes whitespace before `. , ? ! : ;`, removes U+200B, U+200C (ZWNJ), U+2060, U+FEFF and U+00AD, maps curly quotes to straight ones, and appends `.` to a sentence that does not end in `. ? ! ෴ …`. U+200D (ZWJ) is kept because Sinhala conjuncts are spelled with it. **Parentheses and Latin text are left alone**: they are content, not formatting, so their imbalance remains. The threshold and Viterbi bias are re-tuned on the normalized dev set; the models are not retrained.

## The cues before and after normalization

Test set, 5,309 human and 3,146 machine sentences.

| feature | human, raw | machine, raw | human, normalized | machine, normalized |
|---|---|---|---|---|
| space before punctuation | 2.81% | 0.03% | 0.00% | 0.00% |
| parentheses (not normalized) | 11.38% | 3.34% | 11.38% | 3.34% |
| no final punctuation | 3.22% | 0.03% | 0.00% | 0.00% |
| Latin characters (not normalized) | 11.89% | 5.79% | 11.89% | 5.79% |
| zero-width non-joiner | 0.32% | 0.00% | 0.00% | 0.00% |
| curly quotes | 1.39% | 0.41% | 0.00% | 0.00% |

## Scores

Each cell is raw → normalized (change). Twin and unrelated-human false alarms use threshold decoding.

### Mixed documents

| model | sentence F1 (Viterbi) | sentence F1 (threshold) | exact-boundary F1, mixed (Viterbi) | exact-boundary F1, mixed (threshold) | exact-boundary F1, held-out (Viterbi) |
|---|---|---|---|---|---|
| base | 0.792 → 0.790 (-0.001) | 0.802 → 0.800 (-0.002) | 0.604 → 0.602 (-0.002) | 0.603 → 0.593 (-0.010) | 0.549 → 0.544 (-0.005) |
| control | 0.787 → 0.775 (-0.012) | 0.798 → 0.796 (-0.002) | 0.614 → 0.609 (-0.006) | 0.605 → 0.601 (-0.004) | 0.555 → 0.547 (-0.008) |
| unrelated | 0.787 → 0.785 (-0.002) | 0.785 → 0.781 (-0.004) | 0.612 → 0.609 (-0.004) | 0.556 → 0.551 (-0.005) | 0.548 → 0.552 (+0.003) |
| twins_unpaired | 0.792 → 0.784 (-0.008) | 0.797 → 0.794 (-0.003) | 0.621 → 0.620 (-0.001) | 0.582 → 0.576 (-0.006) | 0.556 → 0.553 (-0.003) |
| twins_paired | 0.789 → 0.787 (-0.003) | 0.794 → 0.791 (-0.003) | 0.603 → 0.603 (+0.000) | 0.574 → 0.571 (-0.003) | 0.540 → 0.539 (-0.002) |

### Human documents

| model | twin false-alarm rate | unrelated-human false-alarm rate | sentence FPR, unrelated human |
|---|---|---|---|
| base | 0.953 → 0.965 (+0.012) | 0.935 → 0.950 (+0.015) | 0.241 → 0.268 (+0.027) |
| control | 0.960 → 0.965 (+0.005) | 0.950 → 0.961 (+0.011) | 0.253 → 0.265 (+0.012) |
| unrelated | 0.526 → 0.526 (+0.000) | 0.505 → 0.497 (-0.008) | 0.112 → 0.113 (+0.001) |
| twins_unpaired | 0.562 → 0.562 (+0.000) | 0.571 → 0.579 (+0.008) | 0.122 → 0.125 (+0.003) |
| twins_paired | 0.555 → 0.561 (+0.005) | 0.555 → 0.561 (+0.006) | 0.113 → 0.118 (+0.005) |

Runs: base: `xlmr_rescored_spark` / `xlmr_normsurface`; control: `ft_control_s42` / `ft_control_s42_normsurface`; unrelated: `ft_unrelated_s42` / `ft_unrelated_s42_normsurface`; twins_unpaired: `ft_twins_unpaired_s42` / `ft_twins_unpaired_s42_normsurface`; twins_paired: `ft_twins_paired_s42` / `ft_twins_paired_s42_normsurface`
