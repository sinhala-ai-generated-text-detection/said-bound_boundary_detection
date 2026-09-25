# Results across seeds

Test set. Mean ± sample standard deviation across seeds. Metrics on human documents (twins, unrelated human documents) use threshold decoding unless labelled Viterbi; mixed-only metrics are labelled by decoder.

- **control**: `ft_control_s42` (seed 42), `ft_control_s43` (seed 43), `ft_control_s44` (seed 44)
- **unrelated**: `ft_unrelated_s42` (seed 42), `ft_unrelated_s43` (seed 43), `ft_unrelated_s44` (seed 44)
- **twins_unpaired**: `ft_twins_unpaired_s42` (seed 42), `ft_twins_unpaired_s43` (seed 43), `ft_twins_unpaired_s44` (seed 44)
- **twins_paired**: `ft_twins_paired_s42` (seed 42), `ft_twins_paired_s43` (seed 43), `ft_twins_paired_s44` (seed 44)

| metric | control | unrelated | twins_unpaired | twins_paired |
|---|---|---|---|---|
| exact-boundary F1, mixed (Viterbi) | 0.617 ± 0.004 | 0.618 ± 0.008 | 0.615 ± 0.004 | 0.604 ± 0.001 |
| exact-boundary F1, mixed, held-out (Viterbi) | 0.557 ± 0.007 | 0.554 ± 0.011 | 0.557 ± 0.004 | 0.549 ± 0.009 |
| exact-boundary F1, mixed (threshold) | 0.602 ± 0.004 | 0.566 ± 0.012 | 0.580 ± 0.003 | 0.569 ± 0.011 |
| sentence F1 (Viterbi) | 0.785 ± 0.003 | 0.778 ± 0.010 | 0.792 ± 0.004 | 0.787 ± 0.002 |
| twin false-alarm rate | 0.949 ± 0.020 | 0.632 ± 0.092 | 0.633 ± 0.062 | 0.584 ± 0.030 |
| boundaries per twin | 3.110 ± 0.164 | 1.701 ± 0.194 | 1.741 ± 0.161 | 1.563 ± 0.057 |
| sentence FPR on twins | 0.268 ± 0.013 | 0.140 ± 0.013 | 0.144 ± 0.012 | 0.130 ± 0.004 |
| exact-boundary F1, mixed + twins | 0.432 ± 0.002 | 0.459 ± 0.006 | 0.470 ± 0.010 | 0.469 ± 0.008 |
| exact-boundary F1, mixed + twins, held-out | 0.351 ± 0.005 | 0.390 ± 0.007 | 0.398 ± 0.006 | 0.399 ± 0.007 |
| unrelated-human false-alarm rate | 0.931 ± 0.025 | 0.597 ± 0.080 | 0.615 ± 0.040 | 0.577 ± 0.024 |
| boundaries per unrelated human doc | 2.846 ± 0.147 | 1.494 ± 0.170 | 1.580 ± 0.112 | 1.439 ± 0.049 |
| twin false-alarm rate (Viterbi) | 0.997 ± 0.003 | 0.927 ± 0.025 | 0.863 ± 0.040 | 0.792 ± 0.035 |
| unrelated-human false-alarm rate (Viterbi) | 0.993 ± 0.001 | 0.906 ± 0.036 | 0.839 ± 0.041 | 0.759 ± 0.035 |

*Better in* counts the seeds where the first method beats the second on that metric (lower is better for false alarms, boundaries per twin and FPR).
