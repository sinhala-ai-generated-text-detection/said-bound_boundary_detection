# Results across seeds

Test set. Mean ± sample standard deviation across seeds. Metrics on human documents (twins, unrelated human documents) use threshold decoding unless labelled Viterbi; mixed-only metrics are labelled by decoder.

- **twins_unpaired**: `ft_twins_unpaired_s42` (seed 42), `ft_twins_unpaired_s43` (seed 43), `ft_twins_unpaired_s44` (seed 44)
- **unrelated**: `ft_unrelated_s42` (seed 42), `ft_unrelated_s43` (seed 43), `ft_unrelated_s44` (seed 44)

| metric | twins_unpaired | unrelated | paired diff (twins_unpaired − unrelated) | better in |
|---|---|---|---|---|
| exact-boundary F1, mixed (Viterbi) | 0.615 ± 0.004 | 0.618 ± 0.008 | -0.003 ± 0.011 | 1/3 seeds |
| exact-boundary F1, mixed, held-out (Viterbi) | 0.557 ± 0.004 | 0.554 ± 0.011 | 0.003 ± 0.008 | 2/3 seeds |
| exact-boundary F1, mixed (threshold) | 0.580 ± 0.003 | 0.566 ± 0.012 | 0.013 ± 0.015 | 2/3 seeds |
| sentence F1 (Viterbi) | 0.792 ± 0.004 | 0.778 ± 0.010 | 0.014 ± 0.008 | 3/3 seeds |
| twin false-alarm rate | 0.633 ± 0.062 | 0.632 ± 0.092 | 0.001 ± 0.032 | 2/3 seeds |
| boundaries per twin | 1.741 ± 0.161 | 1.701 ± 0.194 | 0.040 ± 0.056 | 1/3 seeds |
| sentence FPR on twins | 0.144 ± 0.012 | 0.140 ± 0.013 | 0.004 ± 0.002 | 0/3 seeds |
| exact-boundary F1, mixed + twins | 0.470 ± 0.010 | 0.459 ± 0.006 | 0.011 ± 0.012 | 2/3 seeds |
| exact-boundary F1, mixed + twins, held-out | 0.398 ± 0.006 | 0.390 ± 0.007 | 0.008 ± 0.012 | 2/3 seeds |
| unrelated-human false-alarm rate | 0.615 ± 0.040 | 0.597 ± 0.080 | 0.018 ± 0.043 | 1/3 seeds |
| boundaries per unrelated human doc | 1.580 ± 0.112 | 1.494 ± 0.170 | 0.087 ± 0.063 | 0/3 seeds |
| twin false-alarm rate (Viterbi) | 0.863 ± 0.040 | 0.927 ± 0.025 | -0.064 ± 0.026 | 3/3 seeds |
| unrelated-human false-alarm rate (Viterbi) | 0.839 ± 0.041 | 0.906 ± 0.036 | -0.067 ± 0.009 | 3/3 seeds |

*Better in* counts the seeds where the first method beats the second on that metric (lower is better for false alarms, boundaries per twin and FPR).
