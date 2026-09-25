# Operating-point analysis: counterfactual twins

Every model's decision threshold is swept over 0.05-0.95 (threshold decoding). *Mixed bE* is exact-boundary F1 on the mixed test documents; *false alarm* is the share of all-human twins with any sentence flagged; *bE + twins* scores the mixed documents and the twins together, where every boundary predicted on a twin is a false positive.

## Selected operating points

*Mixed-only* picks the threshold with the best dev-seen exact-boundary F1 on mixed documents (the original protocol, on a wider grid). *Deployment* picks it on dev-seen mixed documents plus their twins. Neither selection sees test or the held-out generator.

| model | selection | thr | mixed bE | held-out bE | false alarm | boundaries/twin | bE + twins | held-out bE + twins |
|---|---|---|---|---|---|---|---|---|
| base | mixed-only | 0.950 | 0.600 | 0.555 | 0.931 | 2.88 | 0.438 | 0.359 |
| base | deployment | 0.950 | 0.600 | 0.555 | 0.931 | 2.88 | 0.438 | 0.359 |
| twin_plain | mixed-only | 0.050 | 0.548 | 0.480 | 0.630 | 1.93 | 0.436 | 0.349 |
| twin_plain | deployment | 0.925 | 0.545 | 0.450 | 0.453 | 1.20 | 0.465 | 0.364 |
| twin_warm | mixed-only | 0.650 | 0.543 | 0.475 | 0.490 | 1.19 | 0.464 | 0.381 |
| twin_warm | deployment | 0.650 | 0.543 | 0.475 | 0.490 | 1.19 | 0.464 | 0.381 |

## Mixed-document accuracy at matched false-alarm rates

Best test mixed exact-boundary F1 among thresholds whose twin false-alarm rate is at most the given level. This is read off the test curve, so it describes the trade-off each model offers rather than a tuned result.

| model | alarm <= 0.3 | alarm <= 0.5 | alarm <= 0.7 |
|---|---|---|---|
| base | unreachable | unreachable | unreachable |
| twin_plain | unreachable | 0.550 (thr 0.80) | 0.554 (thr 0.33) |
| twin_warm | 0.502 (thr 0.93) | 0.543 (thr 0.65) | 0.554 (thr 0.40) |
