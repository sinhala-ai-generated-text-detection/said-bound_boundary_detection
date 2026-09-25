# Subtask C tagger: deberta_control_s43

`microsoft/deberta-v3-base`, condition **control**, seed 43: 3649 mixed training documents (864,153 words) plus 0 human-only documents (0 words). 5 epochs, lr 2e-05, batch 8; best epoch 5 on dev MAE (2.92); 17.4 min training.

## Subtask C test

| decoder | slice | docs | MAE | exact |
|---|---|---|---|---|
| first_word | overall | 11123 | 16.63 | 0.214 |
| first_word | outfox | 4000 | 24.84 | 0.148 |
| first_word | peerread | 7123 | 12.01 | 0.251 |
| first_word | outfox/gpt4 | 1000 | 10.11 | 0.427 |
| first_word | outfox/llama13b_chat | 1000 | 31.52 | 0.046 |
| first_word | outfox/llama70b_chat | 1000 | 27.59 | 0.063 |
| first_word | outfox/llama7b_chat | 1000 | 30.16 | 0.057 |
| first_word | peerread/chatgpt | 1522 | 3.24 | 0.536 |
| first_word | peerread/llama | 1035 | 14.17 | 0.302 |
| first_word | peerread/llama2_13B_chat | 1522 | 13.15 | 0.162 |
| first_word | peerread/llama2_70B_chat | 1522 | 15.90 | 0.124 |
| first_word | peerread/llama2_7B_chat | 1522 | 14.29 | 0.147 |
| change_point | overall | 11123 | 14.31 | 0.216 |
| change_point | outfox | 4000 | 19.75 | 0.149 |
| change_point | peerread | 7123 | 11.26 | 0.254 |
| change_point | outfox/gpt4 | 1000 | 7.20 | 0.432 |
| change_point | outfox/llama13b_chat | 1000 | 23.66 | 0.053 |
| change_point | outfox/llama70b_chat | 1000 | 24.17 | 0.061 |
| change_point | outfox/llama7b_chat | 1000 | 23.99 | 0.049 |
| change_point | peerread/chatgpt | 1522 | 2.54 | 0.539 |
| change_point | peerread/llama | 1035 | 17.75 | 0.299 |
| change_point | peerread/llama2_13B_chat | 1522 | 10.85 | 0.166 |
| change_point | peerread/llama2_70B_chat | 1522 | 15.27 | 0.127 |
| change_point | peerread/llama2_7B_chat | 1522 | 11.95 | 0.154 |

## Human-only documents

*False alarm*: share of documents with a predicted boundary (any machine word, for the first-word rule). *Flagged*: mean share of words labelled machine.

| decoder | set | docs | false alarm | flagged |
|---|---|---|---|---|
| first_word | ASAP reviews | 1000 | 0.717 | 0.210 |
| first_word | OUTFOX essays | 1000 | 0.881 | 0.209 |
| first_word | test twins, PeerRead | 747 | 0.778 | 0.208 |
| first_word | test twins, OUTFOX | 997 | 0.889 | 0.231 |
| change_point | ASAP reviews | 1000 | 0.645 | 0.205 |
| change_point | OUTFOX essays | 1000 | 0.817 | 0.194 |
| change_point | test twins, PeerRead | 747 | 0.707 | 0.205 |
| change_point | test twins, OUTFOX | 997 | 0.824 | 0.220 |

## Deployment view

Mixed test documents plus the ASAP and OUTFOX human documents: is there machine text at all?

| decoder | accuracy | balanced | mixed detected | human quiet | MAE, mixed |
|---|---|---|---|---|---|
| first_word | 0.870 | 0.596 | 0.992 | 0.201 | 16.63 |
| change_point | 0.877 | 0.629 | 0.988 | 0.269 | 14.31 |

## Training

| epoch | loss | dev MAE | min |
|---|---|---|---|
| 1 | 0.1715 | 4.87 | 3.5 |
| 2 | 0.0344 | 3.23 | 3.5 |
| 3 | 0.0190 | 3.55 | 3.4 |
| 4 | 0.0127 | 3.05 | 3.5 |
| 5 | 0.0091 | 2.92 | 3.5 |
