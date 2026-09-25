# Subtask C tagger: xlmr_twins_s42

`xlm-roberta-base`, condition **twins**, seed 42: 3649 mixed training documents (864,153 words) plus 1831 human-only documents (449,892 words). 5 epochs, lr 2e-05, batch 8; best epoch 3 on dev MAE (3.95); 15.6 min training.

## Subtask C test

| decoder | slice | docs | MAE | exact |
|---|---|---|---|---|
| first_word | overall | 11123 | 21.71 | 0.180 |
| first_word | outfox | 4000 | 25.69 | 0.133 |
| first_word | peerread | 7123 | 19.47 | 0.206 |
| first_word | outfox/gpt4 | 1000 | 10.38 | 0.388 |
| first_word | outfox/llama13b_chat | 1000 | 26.13 | 0.038 |
| first_word | outfox/llama70b_chat | 1000 | 39.47 | 0.063 |
| first_word | outfox/llama7b_chat | 1000 | 26.78 | 0.044 |
| first_word | peerread/chatgpt | 1522 | 4.83 | 0.483 |
| first_word | peerread/llama | 1035 | 33.05 | 0.220 |
| first_word | peerread/llama2_13B_chat | 1522 | 17.14 | 0.130 |
| first_word | peerread/llama2_70B_chat | 1522 | 25.84 | 0.102 |
| first_word | peerread/llama2_7B_chat | 1522 | 20.83 | 0.101 |
| change_point | overall | 11123 | 22.80 | 0.180 |
| change_point | outfox | 4000 | 26.51 | 0.131 |
| change_point | peerread | 7123 | 20.72 | 0.207 |
| change_point | outfox/gpt4 | 1000 | 8.40 | 0.396 |
| change_point | outfox/llama13b_chat | 1000 | 25.25 | 0.036 |
| change_point | outfox/llama70b_chat | 1000 | 46.05 | 0.056 |
| change_point | outfox/llama7b_chat | 1000 | 26.36 | 0.035 |
| change_point | peerread/chatgpt | 1522 | 4.25 | 0.486 |
| change_point | peerread/llama | 1035 | 39.59 | 0.215 |
| change_point | peerread/llama2_13B_chat | 1522 | 17.07 | 0.130 |
| change_point | peerread/llama2_70B_chat | 1522 | 27.67 | 0.104 |
| change_point | peerread/llama2_7B_chat | 1522 | 21.05 | 0.104 |

## Human-only documents

*False alarm*: share of documents with a predicted boundary (any machine word, for the first-word rule). *Flagged*: mean share of words labelled machine.

| decoder | set | docs | false alarm | flagged |
|---|---|---|---|---|
| first_word | ASAP reviews | 1000 | 0.108 | 0.043 |
| first_word | OUTFOX essays | 1000 | 0.126 | 0.027 |
| first_word | test twins, PeerRead | 747 | 0.092 | 0.026 |
| first_word | test twins, OUTFOX | 997 | 0.163 | 0.036 |
| change_point | ASAP reviews | 1000 | 0.086 | 0.042 |
| change_point | OUTFOX essays | 1000 | 0.092 | 0.025 |
| change_point | test twins, PeerRead | 747 | 0.071 | 0.026 |
| change_point | test twins, OUTFOX | 997 | 0.113 | 0.034 |

## Deployment view

Mixed test documents plus the ASAP and OUTFOX human documents: is there machine text at all?

| decoder | accuracy | balanced | mixed detected | human quiet | MAE, mixed |
|---|---|---|---|---|---|
| first_word | 0.826 | 0.849 | 0.815 | 0.883 | 21.71 |
| change_point | 0.819 | 0.856 | 0.802 | 0.911 | 22.80 |

## Training

| epoch | loss | dev MAE | min |
|---|---|---|---|
| 1 | 0.1513 | 4.46 | 3.1 |
| 2 | 0.0341 | 5.50 | 3.0 |
| 3 | 0.0177 | 3.95 | 3.2 |
| 4 | 0.0122 | 4.53 | 3.0 |
| 5 | 0.0079 | 4.11 | 3.0 |
