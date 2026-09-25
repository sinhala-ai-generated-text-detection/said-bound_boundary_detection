# Operating-point analysis: counterfactual twins

Every model's decision threshold is swept over 0.05-0.95 (threshold decoding). *Mixed bE* is exact-boundary F1 on the mixed test documents; *false alarm* is the share of all-human twins with any sentence flagged; *bE + twins* scores the mixed documents and the twins together, where every boundary predicted on a twin is a false positive.

## Selected operating points

*Mixed-only* picks the threshold with the best dev-seen exact-boundary F1 on mixed documents (the original protocol, on a wider grid). *Deployment* picks it on dev-seen mixed documents plus their twins. Neither selection sees test or the held-out generator.

| model | selection | thr | mixed bE | held-out bE | false alarm | boundaries/twin | bE + twins | held-out bE + twins |
|---|---|---|---|---|---|---|---|---|
| base | mixed-only | 0.950 | 0.600 | 0.555 | 0.933 | 2.88 | 0.438 | 0.359 |
| base | deployment | 0.950 | 0.600 | 0.555 | 0.933 | 2.88 | 0.438 | 0.359 |
| control_s42 | mixed-only | 0.375 | 0.605 | 0.554 | 0.960 | 3.28 | 0.430 | 0.347 |
| control_s42 | deployment | 0.950 | 0.604 | 0.559 | 0.919 | 2.86 | 0.443 | 0.364 |
| control_s43 | mixed-only | 0.700 | 0.605 | 0.565 | 0.958 | 3.08 | 0.435 | 0.358 |
| control_s43 | deployment | 0.950 | 0.607 | 0.572 | 0.945 | 2.93 | 0.441 | 0.369 |
| control_s44 | mixed-only | 0.950 | 0.596 | 0.546 | 0.898 | 2.83 | 0.436 | 0.359 |
| control_s44 | deployment | 0.950 | 0.596 | 0.546 | 0.898 | 2.83 | 0.436 | 0.359 |
| unrelated_s42 | mixed-only | 0.250 | 0.556 | 0.508 | 0.522 | 1.46 | 0.462 | 0.391 |
| unrelated_s42 | deployment | 0.825 | 0.547 | 0.480 | 0.446 | 1.18 | 0.468 | 0.383 |
| unrelated_s43 | mixed-only | 0.400 | 0.548 | 0.489 | 0.567 | 1.39 | 0.457 | 0.381 |
| unrelated_s43 | deployment | 0.400 | 0.548 | 0.489 | 0.567 | 1.39 | 0.457 | 0.381 |
| unrelated_s44 | mixed-only | 0.450 | 0.571 | 0.519 | 0.606 | 1.55 | 0.470 | 0.397 |
| unrelated_s44 | deployment | 0.800 | 0.535 | 0.478 | 0.391 | 0.89 | 0.472 | 0.402 |
| twins_unpaired_s42 | mixed-only | 0.275 | 0.582 | 0.521 | 0.562 | 1.57 | 0.479 | 0.402 |
| twins_unpaired_s42 | deployment | 0.625 | 0.580 | 0.523 | 0.497 | 1.33 | 0.490 | 0.415 |
| twins_unpaired_s43 | mixed-only | 0.450 | 0.583 | 0.531 | 0.668 | 1.79 | 0.469 | 0.398 |
| twins_unpaired_s43 | deployment | 0.925 | 0.557 | 0.496 | 0.434 | 1.03 | 0.484 | 0.408 |
| twins_unpaired_s44 | mixed-only | 0.500 | 0.576 | 0.528 | 0.592 | 1.54 | 0.477 | 0.409 |
| twins_unpaired_s44 | deployment | 0.575 | 0.575 | 0.516 | 0.559 | 1.42 | 0.481 | 0.406 |
| twins_paired_s42 | mixed-only | 0.050 | 0.584 | 0.529 | 0.611 | 1.72 | 0.475 | 0.402 |
| twins_paired_s42 | deployment | 0.950 | 0.557 | 0.476 | 0.386 | 0.97 | 0.489 | 0.398 |
| twins_paired_s43 | mixed-only | 0.500 | 0.556 | 0.516 | 0.587 | 1.55 | 0.459 | 0.397 |
| twins_paired_s43 | deployment | 0.950 | 0.525 | 0.455 | 0.355 | 0.81 | 0.467 | 0.386 |
| twins_paired_s44 | mixed-only | 0.100 | 0.569 | 0.524 | 0.676 | 1.84 | 0.458 | 0.394 |
| twins_paired_s44 | deployment | 0.775 | 0.548 | 0.484 | 0.400 | 0.89 | 0.484 | 0.407 |

## Mixed-document accuracy at matched false-alarm rates

Best test mixed exact-boundary F1 among thresholds whose twin false-alarm rate is at most the given level. This is read off the test curve, so it describes the trade-off each model offers rather than a tuned result.

| model | alarm <= 0.3 | alarm <= 0.5 | alarm <= 0.7 |
|---|---|---|---|
| base | unreachable | unreachable | unreachable |
| control_s42 | unreachable | unreachable | unreachable |
| control_s43 | unreachable | unreachable | unreachable |
| control_s44 | unreachable | unreachable | unreachable |
| unrelated_s42 | unreachable | 0.554 (thr 0.70) | 0.563 (thr 0.10) |
| unrelated_s43 | 0.495 (thr 0.80) | 0.542 (thr 0.50) | 0.566 (thr 0.20) |
| unrelated_s44 | 0.507 (thr 0.90) | 0.555 (thr 0.65) | 0.579 (thr 0.30) |
| twins_unpaired_s42 | unreachable | 0.580 (thr 0.62) | 0.585 (thr 0.07) |
| twins_unpaired_s43 | unreachable | 0.566 (thr 0.88) | 0.585 (thr 0.42) |
| twins_unpaired_s44 | 0.524 (thr 0.95) | 0.575 (thr 0.75) | 0.579 (thr 0.45) |
| twins_paired_s42 | unreachable | 0.571 (thr 0.57) | 0.584 (thr 0.05) |
| twins_paired_s43 | unreachable | 0.547 (thr 0.80) | 0.561 (thr 0.23) |
| twins_paired_s44 | 0.498 (thr 0.93) | 0.567 (thr 0.62) | 0.577 (thr 0.15) |

## The same, against unrelated human documents

As above, but the false-alarm rate is measured on the 618 test-split windows that no mixed document uses, instead of the twins. These have no content link to any document a model was trained or tested on.

| model | lowest false alarm | alarm <= 0.3 | alarm <= 0.5 | alarm <= 0.7 |
|---|---|---|---|---|
| base | 0.913 | unreachable | unreachable | unreachable |
| control_s42 | 0.888 | unreachable | unreachable | unreachable |
| control_s43 | 0.921 | unreachable | unreachable | unreachable |
| control_s44 | 0.882 | unreachable | unreachable | unreachable |
| unrelated_s42 | 0.354 | unreachable | 0.557 (thr 0.28) | 0.563 (thr 0.10) |
| unrelated_s43 | 0.092 | 0.495 (thr 0.80) | 0.544 (thr 0.45) | 0.576 (thr 0.15) |
| unrelated_s44 | 0.189 | 0.528 (thr 0.85) | 0.560 (thr 0.57) | 0.579 (thr 0.30) |
| twins_unpaired_s42 | 0.377 | unreachable | 0.580 (thr 0.62) | 0.585 (thr 0.07) |
| twins_unpaired_s43 | 0.379 | unreachable | 0.572 (thr 0.82) | 0.585 (thr 0.42) |
| twins_unpaired_s44 | 0.293 | 0.524 (thr 0.95) | 0.577 (thr 0.70) | 0.579 (thr 0.45) |
| twins_paired_s42 | 0.393 | unreachable | 0.571 (thr 0.68) | 0.584 (thr 0.05) |
| twins_paired_s43 | 0.316 | unreachable | 0.554 (thr 0.70) | 0.561 (thr 0.23) |
| twins_paired_s44 | 0.204 | 0.510 (thr 0.90) | 0.567 (thr 0.62) | 0.577 (thr 0.15) |
