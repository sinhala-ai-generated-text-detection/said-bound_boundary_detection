# Results across seeds

Test set. Mean ± sample standard deviation across seeds. Metrics on human documents (twins, unrelated human documents) use threshold decoding unless labelled Viterbi; mixed-only metrics are labelled by decoder.

- **twins_paired**: `ft_twins_paired_s42` (seed 42), `ft_twins_paired_s43` (seed 43), `ft_twins_paired_s44` (seed 44)
- **unrelated**: `ft_unrelated_s42` (seed 42), `ft_unrelated_s43` (seed 43), `ft_unrelated_s44` (seed 44)

| metric | twins_paired | unrelated | paired diff (twins_paired − unrelated) | better in |
|---|---|---|---|---|
| exact-boundary F1, mixed (Viterbi) | 0.604 ± 0.001 | 0.618 ± 0.008 | -0.015 ± 0.006 | 0/3 seeds |
| exact-boundary F1, mixed, held-out (Viterbi) | 0.549 ± 0.009 | 0.554 ± 0.011 | -0.005 ± 0.016 | 1/3 seeds |
| exact-boundary F1, mixed (threshold) | 0.569 ± 0.011 | 0.566 ± 0.012 | 0.002 ± 0.014 | 1/3 seeds |
| sentence F1 (Viterbi) | 0.787 ± 0.002 | 0.778 ± 0.010 | 0.009 ± 0.008 | 3/3 seeds |
| twin false-alarm rate | 0.584 ± 0.030 | 0.632 ± 0.092 | -0.048 ± 0.070 | 2/3 seeds |
| boundaries per twin | 1.563 ± 0.057 | 1.701 ± 0.194 | -0.138 ± 0.183 | 2/3 seeds |
| sentence FPR on twins | 0.130 ± 0.004 | 0.140 ± 0.013 | -0.009 ± 0.010 | 2/3 seeds |
| exact-boundary F1, mixed + twins | 0.469 ± 0.008 | 0.459 ± 0.006 | 0.010 ± 0.003 | 3/3 seeds |
| exact-boundary F1, mixed + twins, held-out | 0.399 ± 0.007 | 0.390 ± 0.007 | 0.009 ± 0.008 | 3/3 seeds |
| unrelated-human false-alarm rate | 0.577 ± 0.024 | 0.597 ± 0.080 | -0.020 ± 0.062 | 2/3 seeds |
| boundaries per unrelated human doc | 1.439 ± 0.049 | 1.494 ± 0.170 | -0.054 ± 0.142 | 2/3 seeds |
| twin false-alarm rate (Viterbi) | 0.792 ± 0.035 | 0.927 ± 0.025 | -0.134 ± 0.048 | 3/3 seeds |
| unrelated-human false-alarm rate (Viterbi) | 0.759 ± 0.035 | 0.906 ± 0.036 | -0.147 ± 0.066 | 3/3 seeds |

*Better in* counts the seeds where the first method beats the second on that metric (lower is better for false alarms, boundaries per twin and FPR).
