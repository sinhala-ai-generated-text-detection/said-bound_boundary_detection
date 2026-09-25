# Results across seeds

Test set. Mean ± sample standard deviation across seeds. Metrics on human documents (twins, unrelated human documents) use threshold decoding unless labelled Viterbi; mixed-only metrics are labelled by decoder.

- **twins_paired**: `ft_twins_paired_s42` (seed 42), `ft_twins_paired_s43` (seed 43), `ft_twins_paired_s44` (seed 44)
- **twins_unpaired**: `ft_twins_unpaired_s42` (seed 42), `ft_twins_unpaired_s43` (seed 43), `ft_twins_unpaired_s44` (seed 44)

| metric | twins_paired | twins_unpaired | paired diff (twins_paired − twins_unpaired) | better in |
|---|---|---|---|---|
| exact-boundary F1, mixed (Viterbi) | 0.604 ± 0.001 | 0.615 ± 0.004 | -0.012 ± 0.005 | 0/3 seeds |
| exact-boundary F1, mixed, held-out (Viterbi) | 0.549 ± 0.009 | 0.557 ± 0.004 | -0.008 ± 0.011 | 1/3 seeds |
| exact-boundary F1, mixed (threshold) | 0.569 ± 0.011 | 0.580 ± 0.003 | -0.011 ± 0.014 | 1/3 seeds |
| sentence F1 (Viterbi) | 0.787 ± 0.002 | 0.792 ± 0.004 | -0.005 ± 0.003 | 0/3 seeds |
| twin false-alarm rate | 0.584 ± 0.030 | 0.633 ± 0.062 | -0.050 ± 0.038 | 3/3 seeds |
| boundaries per twin | 1.563 ± 0.057 | 1.741 ± 0.161 | -0.178 ± 0.137 | 3/3 seeds |
| sentence FPR on twins | 0.130 ± 0.004 | 0.144 ± 0.012 | -0.013 ± 0.011 | 3/3 seeds |
| exact-boundary F1, mixed + twins | 0.469 ± 0.008 | 0.470 ± 0.010 | -0.001 ± 0.011 | 1/3 seeds |
| exact-boundary F1, mixed + twins, held-out | 0.399 ± 0.007 | 0.398 ± 0.006 | 0.001 ± 0.011 | 2/3 seeds |
| unrelated-human false-alarm rate | 0.577 ± 0.024 | 0.615 ± 0.040 | -0.039 ± 0.020 | 3/3 seeds |
| boundaries per unrelated human doc | 1.439 ± 0.049 | 1.580 ± 0.112 | -0.141 ± 0.079 | 3/3 seeds |
| twin false-alarm rate (Viterbi) | 0.792 ± 0.035 | 0.863 ± 0.040 | -0.070 ± 0.040 | 3/3 seeds |
| unrelated-human false-alarm rate (Viterbi) | 0.759 ± 0.035 | 0.839 ± 0.041 | -0.080 ± 0.068 | 3/3 seeds |

*Better in* counts the seeds where the first method beats the second on that metric (lower is better for false alarms, boundaries per twin and FPR).
