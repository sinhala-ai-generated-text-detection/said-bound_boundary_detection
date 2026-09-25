# Threshold sweep: Subtask C MAE against false alarms

First-word rule, threshold swept over 0.05–0.95. *False alarm*: share of human-only documents with any word at or above the threshold. Read off the test curves, so these describe the trade-off each model offers rather than a tuned result.

## Lowest false-alarm rate reachable at any threshold

| run | best test MAE | ASAP reviews | OUTFOX essays | test twins, PeerRead | test twins, OUTFOX |
|---|---|---|---|---|---|
| deberta_control_s42 | 16.96 | 0.494 (thr 0.95, MAE 17.3) | 0.660 (thr 0.95, MAE 17.3) | 0.479 (thr 0.95, MAE 17.3) | 0.699 (thr 0.95, MAE 17.3) |
| deberta_control_s43 | 15.81 | 0.554 (thr 0.95, MAE 15.9) | 0.728 (thr 0.95, MAE 15.9) | 0.606 (thr 0.95, MAE 15.9) | 0.754 (thr 0.95, MAE 15.9) |
| deberta_control_s44 | 15.48 | 0.652 (thr 0.95, MAE 15.5) | 0.824 (thr 0.95, MAE 15.5) | 0.689 (thr 0.95, MAE 15.5) | 0.865 (thr 0.95, MAE 15.5) |
| deberta_human_s42 | 20.95 | 0.007 (thr 0.95, MAE 29.2) | 0.055 (thr 0.95, MAE 29.2) | 0.009 (thr 0.95, MAE 29.2) | 0.087 (thr 0.95, MAE 29.2) |
| deberta_human_s43 | 18.92 | 0.016 (thr 0.95, MAE 24.3) | 0.083 (thr 0.95, MAE 24.3) | 0.029 (thr 0.95, MAE 24.3) | 0.101 (thr 0.95, MAE 24.3) |
| deberta_human_s44 | 17.06 | 0.024 (thr 0.95, MAE 20.1) | 0.106 (thr 0.95, MAE 20.1) | 0.035 (thr 0.95, MAE 20.1) | 0.134 (thr 0.95, MAE 20.1) |
| deberta_twins_s42 | 18.25 | 0.035 (thr 0.95, MAE 21.7) | 0.112 (thr 0.95, MAE 21.7) | 0.024 (thr 0.95, MAE 21.7) | 0.135 (thr 0.95, MAE 21.7) |
| deberta_twins_s43 | 18.48 | 0.039 (thr 0.95, MAE 23.2) | 0.138 (thr 0.95, MAE 23.2) | 0.020 (thr 0.95, MAE 23.2) | 0.151 (thr 0.95, MAE 23.2) |
| deberta_twins_s44 | 18.39 | 0.019 (thr 0.95, MAE 24.2) | 0.040 (thr 0.95, MAE 24.2) | 0.009 (thr 0.93, MAE 23.5) | 0.057 (thr 0.95, MAE 24.2) |
| xlmr_control_s42 | 19.06 | 0.572 (thr 0.95, MAE 20.4) | 0.396 (thr 0.95, MAE 20.4) | 0.598 (thr 0.95, MAE 20.4) | 0.463 (thr 0.95, MAE 20.4) |
| xlmr_human_s42 | 23.40 | 0.038 (thr 0.95, MAE 29.0) | 0.079 (thr 0.95, MAE 29.0) | 0.074 (thr 0.95, MAE 29.0) | 0.111 (thr 0.95, MAE 29.0) |
| xlmr_twins_s42 | 21.30 | 0.075 (thr 0.95, MAE 25.1) | 0.088 (thr 0.95, MAE 25.1) | 0.068 (thr 0.95, MAE 25.1) | 0.115 (thr 0.95, MAE 25.1) |

## Beyond the grid

The same rule at thresholds above 0.95: test MAE, then false alarms on ASAP reviews / OUTFOX essays.

| run | thr 0.975 | thr 0.99 | thr 0.995 | thr 0.999 |
|---|---|---|---|---|
| deberta_control_s42 | 17.8; 0.44 / 0.60 | 18.8; 0.38 / 0.53 | 19.8; 0.35 / 0.47 | 22.9; 0.25 / 0.34 |
| deberta_control_s43 | 16.3; 0.52 / 0.68 | 17.0; 0.47 / 0.62 | 17.6; 0.43 / 0.57 | 19.7; 0.35 / 0.45 |
| deberta_control_s44 | 15.6; 0.62 / 0.79 | 16.0; 0.57 / 0.74 | 16.2; 0.53 / 0.70 | 17.7; 0.44 / 0.62 |
| deberta_human_s42 | 31.8; 0.01 / 0.04 | 36.2; 0.00 / 0.03 | 39.4; 0.00 / 0.02 | 46.9; 0.00 / 0.01 |
| deberta_human_s43 | 25.8; 0.01 / 0.08 | 27.8; 0.01 / 0.07 | 29.2; 0.01 / 0.07 | 32.6; 0.01 / 0.04 |
| deberta_human_s44 | 20.8; 0.02 / 0.10 | 21.8; 0.02 / 0.09 | 22.9; 0.02 / 0.08 | 25.0; 0.01 / 0.07 |
| deberta_twins_s42 | 22.3; 0.03 / 0.10 | 23.6; 0.03 / 0.09 | 24.8; 0.03 / 0.08 | 27.7; 0.02 / 0.06 |
| deberta_twins_s43 | 24.2; 0.04 / 0.12 | 25.3; 0.03 / 0.12 | 26.2; 0.03 / 0.11 | 28.8; 0.02 / 0.08 |
| deberta_twins_s44 | 25.3; 0.02 / 0.04 | 26.6; 0.01 / 0.03 | 27.6; 0.01 / 0.03 | 30.2; 0.01 / 0.02 |
| xlmr_control_s42 | 21.5; 0.51 / 0.36 | 24.6; 0.42 / 0.32 | 27.6; 0.35 / 0.28 | 36.6; 0.21 / 0.18 |
| xlmr_human_s42 | 30.2; 0.03 / 0.07 | 31.9; 0.03 / 0.07 | 33.4; 0.02 / 0.06 | 37.2; 0.02 / 0.05 |
| xlmr_twins_s42 | 26.6; 0.07 / 0.07 | 28.9; 0.06 / 0.07 | 30.6; 0.05 / 0.06 | 35.1; 0.04 / 0.04 |

## Lowest false alarm at a given test MAE

Over every threshold (grid and beyond) whose test MAE is at most the level: false alarms on ASAP reviews / OUTFOX essays. This compares models at equal accuracy on the official metric.

| run | MAE ≤ 18 | MAE ≤ 20 | MAE ≤ 22 | MAE ≤ 25 |
|---|---|---|---|---|
| deberta_control_s42 | 0.44 / 0.60 | 0.35 / 0.47 | 0.35 / 0.47 | 0.25 / 0.34 |
| deberta_control_s43 | 0.43 / 0.57 | 0.35 / 0.45 | 0.35 / 0.45 | 0.35 / 0.45 |
| deberta_control_s44 | 0.44 / 0.62 | 0.44 / 0.62 | 0.44 / 0.62 | 0.44 / 0.62 |
| deberta_human_s42 | unreachable | unreachable | 0.03 / 0.14 | 0.01 / 0.09 |
| deberta_human_s43 | unreachable | 0.03 / 0.12 | 0.02 / 0.10 | 0.02 / 0.08 |
| deberta_human_s44 | 0.03 / 0.15 | 0.03 / 0.11 | 0.02 / 0.09 | 0.01 / 0.07 |
| deberta_twins_s42 | unreachable | 0.05 / 0.15 | 0.04 / 0.11 | 0.03 / 0.08 |
| deberta_twins_s43 | unreachable | 0.05 / 0.17 | 0.05 / 0.15 | 0.04 / 0.12 |
| deberta_twins_s44 | unreachable | 0.03 / 0.07 | 0.03 / 0.05 | 0.02 / 0.04 |
| xlmr_control_s42 | unreachable | 0.60 / 0.42 | 0.51 / 0.36 | 0.42 / 0.32 |
| xlmr_human_s42 | unreachable | unreachable | unreachable | 0.06 / 0.11 |
| xlmr_twins_s42 | unreachable | unreachable | 0.10 / 0.11 | 0.08 / 0.09 |

## Best test MAE at a false-alarm rate on ASAP reviews

| run | ≤ 10% | ≤ 30% | ≤ 50% |
|---|---|---|---|
| deberta_control_s42 | unreachable | unreachable | 17.3 (thr 0.95) |
| deberta_control_s43 | unreachable | unreachable | unreachable |
| deberta_control_s44 | unreachable | unreachable | unreachable |
| deberta_human_s42 | 21.0 (thr 0.28) | 21.0 (thr 0.28) | 21.0 (thr 0.28) |
| deberta_human_s43 | 18.9 (thr 0.05) | 18.9 (thr 0.05) | 18.9 (thr 0.05) |
| deberta_human_s44 | 17.1 (thr 0.20) | 17.1 (thr 0.20) | 17.1 (thr 0.20) |
| deberta_twins_s42 | 18.2 (thr 0.07) | 18.2 (thr 0.07) | 18.2 (thr 0.07) |
| deberta_twins_s43 | 18.5 (thr 0.05) | 18.5 (thr 0.05) | 18.5 (thr 0.05) |
| deberta_twins_s44 | 18.4 (thr 0.05) | 18.4 (thr 0.05) | 18.4 (thr 0.05) |
| xlmr_control_s42 | unreachable | unreachable | unreachable |
| xlmr_human_s42 | 23.4 (thr 0.05) | 23.4 (thr 0.05) | 23.4 (thr 0.05) |
| xlmr_twins_s42 | 22.0 (thr 0.65) | 21.3 (thr 0.20) | 21.3 (thr 0.20) |

## Best test MAE at a false-alarm rate on OUTFOX essays

| run | ≤ 10% | ≤ 30% | ≤ 50% |
|---|---|---|---|
| deberta_control_s42 | unreachable | unreachable | unreachable |
| deberta_control_s43 | unreachable | unreachable | unreachable |
| deberta_control_s44 | unreachable | unreachable | unreachable |
| deberta_human_s42 | 24.4 (thr 0.80) | 21.0 (thr 0.28) | 21.0 (thr 0.28) |
| deberta_human_s43 | 22.0 (thr 0.80) | 18.9 (thr 0.05) | 18.9 (thr 0.05) |
| deberta_human_s44 | unreachable | 17.1 (thr 0.20) | 17.1 (thr 0.20) |
| deberta_twins_s42 | unreachable | 18.2 (thr 0.07) | 18.2 (thr 0.07) |
| deberta_twins_s43 | unreachable | 18.5 (thr 0.05) | 18.5 (thr 0.05) |
| deberta_twins_s44 | 18.5 (thr 0.07) | 18.4 (thr 0.05) | 18.4 (thr 0.05) |
| xlmr_control_s42 | unreachable | unreachable | 19.1 (thr 0.80) |
| xlmr_human_s42 | 26.0 (thr 0.70) | 23.4 (thr 0.05) | 23.4 (thr 0.05) |
| xlmr_twins_s42 | 23.9 (thr 0.90) | 21.3 (thr 0.20) | 21.3 (thr 0.20) |
