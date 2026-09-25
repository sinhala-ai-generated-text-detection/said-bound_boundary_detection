# Token lengths under `microsoft/deberta-v3-base`

Tokens per document (no special tokens). *Over window*: share of documents longer than 510 tokens, which are split into chunks on word boundaries.

| set | docs | median words | p5 | p50 | p95 | p99 | max | tokens/word | over window |
|---|---|---|---|---|---|---|---|---|---|
| train | 3649 | 201 | 77 | 241 | 631 | 819 | 1521 | 1.20 | 11.5% |
| dev | 505 | 191 | 82 | 234 | 600 | 799 | 1055 | 1.22 | 9.7% |
| test | 11123 | 212 | 75 | 256 | 646 | 863 | 1911 | 1.19 | 15.1% |
| human_train_asap | 1831 | 191 | 35 | 238 | 832 | 1274 | 3138 | 1.27 | 17.0% |
| human_train_twins | 1831 | 190 | 32 | 240 | 846 | 1281 | 2073 | 1.27 | 17.7% |
| eval_asap | 1000 | 158 | 62 | 198 | 549 | 725 | 1213 | 1.27 | 6.3% |
| eval_outfox | 1000 | 382 | 129 | 424 | 740 | 1094 | 1376 | 1.11 | 30.7% |
| eval_twins_peerread | 747 | 191 | 32 | 241 | 775 | 1327 | 2148 | 1.28 | 18.6% |
| eval_twins_outfox | 997 | 363 | 197 | 421 | 871 | 1119 | 1578 | 1.14 | 34.1% |
