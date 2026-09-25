# Subtask C tagger: deberta_control_s42

`microsoft/deberta-v3-base`, condition **control**, seed 42: 3649 mixed training documents (864,153 words) plus 0 human-only documents (0 words). 5 epochs, lr 2e-05, batch 8; best epoch 3 on dev MAE (3.03); 17.2 min training.

## Subtask C test

| decoder | slice | docs | MAE | exact |
|---|---|---|---|---|
| first_word | overall | 11123 | 17.80 | 0.210 |
| first_word | outfox | 4000 | 27.02 | 0.163 |
| first_word | peerread | 7123 | 12.62 | 0.237 |
| first_word | outfox/gpt4 | 1000 | 10.66 | 0.430 |
| first_word | outfox/llama13b_chat | 1000 | 35.75 | 0.064 |
| first_word | outfox/llama70b_chat | 1000 | 28.51 | 0.090 |
| first_word | outfox/llama7b_chat | 1000 | 33.16 | 0.069 |
| first_word | peerread/chatgpt | 1522 | 3.33 | 0.523 |
| first_word | peerread/llama | 1035 | 12.36 | 0.298 |
| first_word | peerread/llama2_13B_chat | 1522 | 13.92 | 0.146 |
| first_word | peerread/llama2_70B_chat | 1522 | 16.85 | 0.121 |
| first_word | peerread/llama2_7B_chat | 1522 | 16.58 | 0.116 |
| change_point | overall | 11123 | 15.31 | 0.213 |
| change_point | outfox | 4000 | 21.98 | 0.162 |
| change_point | peerread | 7123 | 11.57 | 0.241 |
| change_point | outfox/gpt4 | 1000 | 8.61 | 0.435 |
| change_point | outfox/llama13b_chat | 1000 | 28.01 | 0.068 |
| change_point | outfox/llama70b_chat | 1000 | 25.67 | 0.083 |
| change_point | outfox/llama7b_chat | 1000 | 25.62 | 0.062 |
| change_point | peerread/chatgpt | 1522 | 2.61 | 0.526 |
| change_point | peerread/llama | 1035 | 14.54 | 0.297 |
| change_point | peerread/llama2_13B_chat | 1522 | 11.62 | 0.149 |
| change_point | peerread/llama2_70B_chat | 1522 | 16.40 | 0.128 |
| change_point | peerread/llama2_7B_chat | 1522 | 13.61 | 0.125 |

## Human-only documents

*False alarm*: share of documents with a predicted boundary (any machine word, for the first-word rule). *Flagged*: mean share of words labelled machine.

| decoder | set | docs | false alarm | flagged |
|---|---|---|---|---|
| first_word | ASAP reviews | 1000 | 0.702 | 0.191 |
| first_word | OUTFOX essays | 1000 | 0.856 | 0.237 |
| first_word | test twins, PeerRead | 747 | 0.701 | 0.161 |
| first_word | test twins, OUTFOX | 997 | 0.884 | 0.266 |
| change_point | ASAP reviews | 1000 | 0.584 | 0.183 |
| change_point | OUTFOX essays | 1000 | 0.768 | 0.234 |
| change_point | test twins, PeerRead | 747 | 0.597 | 0.153 |
| change_point | test twins, OUTFOX | 997 | 0.800 | 0.261 |

## Deployment view

Mixed test documents plus the ASAP and OUTFOX human documents: is there machine text at all?

| decoder | accuracy | balanced | mixed detected | human quiet | MAE, mixed |
|---|---|---|---|---|---|
| first_word | 0.869 | 0.604 | 0.987 | 0.221 | 17.80 |
| change_point | 0.882 | 0.654 | 0.983 | 0.324 | 15.31 |

## Training

| epoch | loss | dev MAE | min |
|---|---|---|---|
| 1 | 0.1888 | 3.46 | 3.4 |
| 2 | 0.0358 | 3.34 | 3.4 |
| 3 | 0.0196 | 3.03 | 3.4 |
| 4 | 0.0130 | 3.05 | 3.4 |
| 5 | 0.0090 | 3.08 | 3.4 |
