# Subtask C: results across seeds

Test set, mean ± sample standard deviation over seeds. The first-word rule at threshold 0.5 unless labelled *change point*. False alarms and flagged words in percent.

- **A**: `deberta_control_s42`, `deberta_control_s43`, `deberta_control_s44`
- **B**: `deberta_human_s42`, `deberta_human_s43`, `deberta_human_s44`
- **C**: `deberta_twins_s42`, `deberta_twins_s43`, `deberta_twins_s44`

| metric | A | B | C |
|---|---|---|---|
| test MAE | 17.064 ± 0.643 | 20.086 ± 2.279 | 19.627 ± 0.387 |
| test exact-boundary accuracy | 0.213 ± 0.002 | 0.193 ± 0.005 | 0.204 ± 0.006 |
| test MAE, PeerRead | 12.226 ± 0.345 | 19.579 ± 3.212 | 18.146 ± 0.975 |
| test MAE, OUTFOX | 25.678 ± 1.175 | 20.989 ± 2.367 | 22.263 ± 0.742 |
| false alarm, ASAP reviews (%) | 73.767 ± 4.936 | 2.800 ± 0.529 | 4.600 ± 1.480 |
| false alarm, OUTFOX essays (%) | 89.200 ± 4.258 | 13.633 ± 2.101 | 13.867 ± 5.802 |
| false alarm, test twins PeerRead (%) | 77.867 ± 7.765 | 5.712 ± 1.154 | 3.436 ± 1.419 |
| false alarm, test twins OUTFOX (%) | 90.806 ± 3.801 | 16.048 ± 1.312 | 15.413 ± 6.270 |
| words flagged, ASAP reviews (%) | 21.491 ± 2.697 | 0.953 ± 0.177 | 1.651 ± 0.444 |
| words flagged, OUTFOX essays (%) | 23.583 ± 2.633 | 2.688 ± 0.918 | 2.899 ± 1.781 |
| presence accuracy, mixed + human (balanced) | 0.588 ± 0.022 | 0.864 ± 0.007 | 0.865 ± 0.010 |
| change point: test MAE | 14.606 ± 0.612 | 22.504 ± 3.225 | 22.449 ± 0.279 |
| change point: false alarm, ASAP (%) | 64.700 ± 6.402 | 2.300 ± 0.436 | 3.500 ± 0.889 |
| change point: false alarm, OUTFOX (%) | 82.633 ± 6.352 | 10.800 ± 1.400 | 11.133 ± 4.450 |
| change point: false alarm, twins PeerRead (%) | 69.612 ± 9.417 | 4.596 ± 1.082 | 2.900 ± 1.115 |

## Seed-paired differences

Each cell: mean ± sd of the per-seed difference, and the seeds on which the first condition is better.

| metric | B − A | C − A | C − B |
|---|---|---|---|
| test MAE | 3.022 ± 1.898 (0/3) | 2.563 ± 0.972 (0/3) | -0.459 ± 2.663 (2/3) |
| test exact-boundary accuracy | -0.020 ± 0.005 (0/3) | -0.008 ± 0.008 (1/3) | 0.011 ± 0.010 (3/3) |
| test MAE, PeerRead | 7.353 ± 3.083 (0/3) | 5.920 ± 1.300 (0/3) | -1.433 ± 3.980 (2/3) |
| test MAE, OUTFOX | -4.689 ± 1.276 (3/3) | -3.415 ± 0.446 (3/3) | 1.275 ± 1.703 (1/3) |
| false alarm, ASAP reviews (%) | -70.967 ± 4.441 (3/3) | -69.167 ± 6.414 (3/3) | 1.800 ± 1.992 (1/3) |
| false alarm, OUTFOX essays (%) | -75.567 ± 3.275 (3/3) | -75.333 ± 9.758 (3/3) | 0.233 ± 7.808 (1/3) |
| false alarm, test twins PeerRead (%) | -72.155 ± 6.628 (3/3) | -74.431 ± 9.171 (3/3) | -2.276 ± 2.544 (2/3) |
| false alarm, test twins OUTFOX (%) | -74.758 ± 3.281 (3/3) | -75.393 ± 10.031 (3/3) | -0.635 ± 7.236 (1/3) |
| words flagged, ASAP reviews (%) | -20.538 ± 2.520 (3/3) | -19.840 ± 3.051 (3/3) | 0.698 ± 0.596 (0/3) |
| words flagged, OUTFOX essays (%) | -20.895 ± 1.862 (3/3) | -20.684 ± 4.382 (3/3) | 0.211 ± 2.684 (1/3) |
| presence accuracy, mixed + human (balanced) | 0.276 ± 0.029 (3/3) | 0.277 ± 0.030 (3/3) | 0.001 ± 0.006 (2/3) |
| change point: test MAE | 7.898 ± 2.718 (0/3) | 7.843 ± 0.891 (0/3) | -0.055 ± 3.464 (2/3) |
| change point: false alarm, ASAP (%) | -62.400 ± 6.000 (3/3) | -61.200 ± 7.093 (3/3) | 1.200 ± 1.308 (1/3) |
| change point: false alarm, OUTFOX (%) | -71.833 ± 5.615 (3/3) | -71.500 ± 10.548 (3/3) | 0.333 ± 5.729 (1/3) |
| change point: false alarm, twins PeerRead (%) | -65.016 ± 8.370 (3/3) | -66.711 ± 10.515 (3/3) | -1.696 ± 2.147 (2/3) |
