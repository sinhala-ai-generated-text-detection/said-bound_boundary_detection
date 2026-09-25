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
| base_ft | mixed-only | 0.900 | 0.592 | 0.538 | 0.955 | 3.05 | 0.427 | 0.344 |
| base_ft | deployment | 0.925 | 0.595 | 0.543 | 0.950 | 3.01 | 0.430 | 0.348 |
| base_ft_s43 | mixed-only | 0.825 | 0.603 | 0.550 | 0.941 | 3.02 | 0.437 | 0.354 |
| base_ft_s43 | deployment | 0.950 | 0.605 | 0.550 | 0.929 | 2.89 | 0.442 | 0.358 |
| base_ft_s44 | mixed-only | 0.850 | 0.606 | 0.556 | 0.931 | 2.88 | 0.442 | 0.365 |
| base_ft_s44 | deployment | 0.950 | 0.608 | 0.558 | 0.917 | 2.74 | 0.449 | 0.369 |
| twin_ft | mixed-only | 0.900 | 0.574 | 0.522 | 0.561 | 1.50 | 0.477 | 0.403 |
| twin_ft | deployment | 0.925 | 0.577 | 0.525 | 0.545 | 1.46 | 0.481 | 0.408 |
| twin_ft_s43 | mixed-only | 0.100 | 0.578 | 0.538 | 0.687 | 1.94 | 0.461 | 0.399 |
| twin_ft_s43 | deployment | 0.925 | 0.545 | 0.480 | 0.439 | 1.12 | 0.470 | 0.392 |
| twin_ft_s44 | mixed-only | 0.750 | 0.578 | 0.528 | 0.571 | 1.59 | 0.476 | 0.408 |
| twin_ft_s44 | deployment | 0.775 | 0.578 | 0.528 | 0.569 | 1.58 | 0.476 | 0.407 |

## Mixed-document accuracy at matched false-alarm rates

Best test mixed exact-boundary F1 among thresholds whose twin false-alarm rate is at most the given level. This is read off the test curve, so it describes the trade-off each model offers rather than a tuned result.

| model | alarm <= 0.3 | alarm <= 0.5 | alarm <= 0.7 |
|---|---|---|---|
| base | unreachable | unreachable | unreachable |
| twin_plain | unreachable | 0.550 (thr 0.80) | 0.554 (thr 0.33) |
| twin_warm | 0.502 (thr 0.93) | 0.543 (thr 0.65) | 0.554 (thr 0.40) |
| base_ft | unreachable | unreachable | unreachable |
| base_ft_s43 | unreachable | unreachable | unreachable |
| base_ft_s44 | unreachable | unreachable | unreachable |
| twin_ft | unreachable | unreachable | 0.578 (thr 0.53) |
| twin_ft_s43 | unreachable | 0.559 (thr 0.80) | 0.580 (thr 0.12) |
| twin_ft_s44 | unreachable | 0.579 (thr 0.95) | 0.580 (thr 0.90) |
