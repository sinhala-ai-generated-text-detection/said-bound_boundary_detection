# Results across seeds

Test set. Mean ± sample standard deviation across seeds. Metrics on human documents (twins, unrelated human documents) use threshold decoding unless labelled Viterbi; mixed-only metrics are labelled by decoder.

- **unrelated**: `ft_unrelated_s42` (seed 42), `ft_unrelated_s43` (seed 43), `ft_unrelated_s44` (seed 44)
- **control**: `ft_control_s42` (seed 42), `ft_control_s43` (seed 43), `ft_control_s44` (seed 44)

| metric | unrelated | control | paired diff (unrelated − control) | better in |
|---|---|---|---|---|
| exact-boundary F1, mixed (Viterbi) | 0.618 ± 0.008 | 0.617 ± 0.004 | 0.001 ± 0.009 | 1/3 seeds |
| exact-boundary F1, mixed, held-out (Viterbi) | 0.554 ± 0.011 | 0.557 ± 0.007 | -0.003 ± 0.017 | 1/3 seeds |
| exact-boundary F1, mixed (threshold) | 0.566 ± 0.012 | 0.602 ± 0.004 | -0.036 ± 0.016 | 0/3 seeds |
| sentence F1 (Viterbi) | 0.778 ± 0.010 | 0.785 ± 0.003 | -0.007 ± 0.011 | 0/3 seeds |
| twin false-alarm rate | 0.632 ± 0.092 | 0.949 ± 0.020 | -0.317 ± 0.103 | 3/3 seeds |
| boundaries per twin | 1.701 ± 0.194 | 3.110 ± 0.164 | -1.408 ± 0.354 | 3/3 seeds |
| sentence FPR on twins | 0.140 ± 0.013 | 0.268 ± 0.013 | -0.128 ± 0.025 | 3/3 seeds |
| exact-boundary F1, mixed + twins | 0.459 ± 0.006 | 0.432 ± 0.002 | 0.027 ± 0.006 | 3/3 seeds |
| exact-boundary F1, mixed + twins, held-out | 0.390 ± 0.007 | 0.351 ± 0.005 | 0.039 ± 0.009 | 3/3 seeds |
| unrelated-human false-alarm rate | 0.597 ± 0.080 | 0.931 ± 0.025 | -0.334 ± 0.099 | 3/3 seeds |
| boundaries per unrelated human doc | 1.494 ± 0.170 | 2.846 ± 0.147 | -1.352 ± 0.303 | 3/3 seeds |
| twin false-alarm rate (Viterbi) | 0.927 ± 0.025 | 0.997 ± 0.003 | -0.070 ± 0.024 | 3/3 seeds |
| unrelated-human false-alarm rate (Viterbi) | 0.906 ± 0.036 | 0.993 ± 0.001 | -0.087 ± 0.036 | 3/3 seeds |

*Better in* counts the seeds where the first method beats the second on that metric (lower is better for false alarms, boundaries per twin and FPR).
