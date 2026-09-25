# Subtask C tagger: deberta_twins_s43

`microsoft/deberta-v3-base`, condition **twins**, seed 43: 3649 mixed training documents (864,153 words) plus 1831 human-only documents (449,892 words). 5 epochs, lr 2e-05, batch 8; best epoch 5 on dev MAE (2.38); 26.7 min training.

## Subtask C test

| decoder | slice | docs | MAE | exact |
|---|---|---|---|---|
| first_word | overall | 11123 | 19.60 | 0.202 |
| first_word | outfox | 4000 | 21.82 | 0.153 |
| first_word | peerread | 7123 | 18.35 | 0.230 |
| first_word | outfox/gpt4 | 1000 | 8.42 | 0.431 |
| first_word | outfox/llama13b_chat | 1000 | 23.80 | 0.053 |
| first_word | outfox/llama70b_chat | 1000 | 31.21 | 0.070 |
| first_word | outfox/llama7b_chat | 1000 | 23.84 | 0.057 |
| first_word | peerread/chatgpt | 1522 | 2.55 | 0.552 |
| first_word | peerread/llama | 1035 | 44.34 | 0.207 |
| first_word | peerread/llama2_13B_chat | 1522 | 14.53 | 0.146 |
| first_word | peerread/llama2_70B_chat | 1522 | 21.27 | 0.120 |
| first_word | peerread/llama2_7B_chat | 1522 | 17.39 | 0.120 |
| change_point | overall | 11123 | 22.59 | 0.201 |
| change_point | outfox | 4000 | 24.45 | 0.151 |
| change_point | peerread | 7123 | 21.54 | 0.229 |
| change_point | outfox/gpt4 | 1000 | 6.69 | 0.433 |
| change_point | outfox/llama13b_chat | 1000 | 23.48 | 0.052 |
| change_point | outfox/llama70b_chat | 1000 | 43.20 | 0.069 |
| change_point | outfox/llama7b_chat | 1000 | 24.45 | 0.051 |
| change_point | peerread/chatgpt | 1522 | 2.50 | 0.552 |
| change_point | peerread/llama | 1035 | 59.07 | 0.199 |
| change_point | peerread/llama2_13B_chat | 1522 | 15.43 | 0.148 |
| change_point | peerread/llama2_70B_chat | 1522 | 24.03 | 0.120 |
| change_point | peerread/llama2_7B_chat | 1522 | 18.69 | 0.118 |

## Human-only documents

*False alarm*: share of documents with a predicted boundary (any machine word, for the first-word rule). *Flagged*: mean share of words labelled machine.

| decoder | set | docs | false alarm | flagged |
|---|---|---|---|---|
| first_word | ASAP reviews | 1000 | 0.053 | 0.020 |
| first_word | OUTFOX essays | 1000 | 0.183 | 0.044 |
| first_word | test twins, PeerRead | 747 | 0.032 | 0.008 |
| first_word | test twins, OUTFOX | 997 | 0.198 | 0.057 |
| change_point | ASAP reviews | 1000 | 0.042 | 0.019 |
| change_point | OUTFOX essays | 1000 | 0.139 | 0.042 |
| change_point | test twins, PeerRead | 747 | 0.025 | 0.008 |
| change_point | test twins, OUTFOX | 997 | 0.154 | 0.057 |

## Deployment view

Mixed test documents plus the ASAP and OUTFOX human documents: is there machine text at all?

| decoder | accuracy | balanced | mixed detected | human quiet | MAE, mixed |
|---|---|---|---|---|---|
| first_word | 0.838 | 0.856 | 0.829 | 0.882 | 19.60 |
| change_point | 0.825 | 0.860 | 0.810 | 0.909 | 22.59 |

## Training

| epoch | loss | dev MAE | min |
|---|---|---|---|
| 1 | 0.1266 | 4.27 | 5.4 |
| 2 | 0.0248 | 5.73 | 5.2 |
| 3 | 0.0123 | 2.94 | 5.4 |
| 4 | 0.0084 | 2.44 | 5.3 |
| 5 | 0.0065 | 2.38 | 5.3 |
