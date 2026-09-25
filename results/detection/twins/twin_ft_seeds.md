# Results across seeds

Test set. Mean ± sample standard deviation across seeds. Twin metrics use threshold decoding; mixed-only metrics are labelled by decoder.

- **twin_ft**: `twin_ft` (seed 42), `twin_ft_s43` (seed 43), `twin_ft_s44` (seed 44)
- **control**: `xlmr_ft` (seed 42), `xlmr_ft_s43` (seed 43), `xlmr_ft_s44` (seed 44)

| metric | twin_ft | control | paired diff (twin_ft − control) | better in |
|---|---|---|---|---|
| exact-boundary F1, mixed (Viterbi) | 0.596 ± 0.002 | 0.607 ± 0.012 | -0.010 ± 0.010 | 1/3 seeds |
| exact-boundary F1, mixed, held-out (Viterbi) | 0.554 ± 0.005 | 0.548 ± 0.006 | 0.006 ± 0.007 | 2/3 seeds |
| exact-boundary F1, mixed (threshold) | 0.574 ± 0.004 | 0.599 ± 0.010 | -0.025 ± 0.010 | 0/3 seeds |
| sentence F1 (Viterbi) | 0.791 ± 0.004 | 0.789 ± 0.000 | 0.002 ± 0.004 | 2/3 seeds |
| twin false-alarm rate | 0.583 ± 0.011 | 0.954 ± 0.013 | -0.371 ± 0.007 | 3/3 seeds |
| boundaries per twin | 1.603 ± 0.014 | 3.107 ± 0.114 | -1.505 ± 0.101 | 3/3 seeds |
| sentence FPR on twins | 0.141 ± 0.004 | 0.264 ± 0.016 | -0.123 ± 0.012 | 3/3 seeds |
| exact-boundary F1, mixed + twins | 0.472 ± 0.004 | 0.430 ± 0.010 | 0.042 ± 0.009 | 3/3 seeds |
| exact-boundary F1, mixed + twins, held-out | 0.404 ± 0.003 | 0.348 ± 0.011 | 0.056 ± 0.008 | 3/3 seeds |

*Better in* counts the seeds where the first method beats the second on that metric (lower is better for false alarms, boundaries per twin and FPR).
