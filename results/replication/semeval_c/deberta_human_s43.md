# Subtask C tagger: deberta_human_s43

`microsoft/deberta-v3-base`, condition **human**, seed 43: 3649 mixed training documents (864,153 words) plus 1831 human-only documents (448,594 words). 5 epochs, lr 2e-05, batch 8; best epoch 4 on dev MAE (2.75); 26.6 min training.

## Subtask C test

| decoder | slice | docs | MAE | exact |
|---|---|---|---|---|
| first_word | overall | 11123 | 20.57 | 0.189 |
| first_word | outfox | 4000 | 18.79 | 0.160 |
| first_word | peerread | 7123 | 21.57 | 0.205 |
| first_word | outfox/gpt4 | 1000 | 6.88 | 0.437 |
| first_word | outfox/llama13b_chat | 1000 | 19.72 | 0.052 |
| first_word | outfox/llama70b_chat | 1000 | 28.73 | 0.089 |
| first_word | outfox/llama7b_chat | 1000 | 19.83 | 0.063 |
| first_word | peerread/chatgpt | 1522 | 2.66 | 0.535 |
| first_word | peerread/llama | 1035 | 54.91 | 0.149 |
| first_word | peerread/llama2_13B_chat | 1522 | 16.47 | 0.132 |
| first_word | peerread/llama2_70B_chat | 1522 | 24.87 | 0.094 |
| first_word | peerread/llama2_7B_chat | 1522 | 19.60 | 0.098 |
| change_point | overall | 11123 | 22.91 | 0.186 |
| change_point | outfox | 4000 | 19.30 | 0.157 |
| change_point | peerread | 7123 | 24.94 | 0.203 |
| change_point | outfox/gpt4 | 1000 | 5.95 | 0.440 |
| change_point | outfox/llama13b_chat | 1000 | 19.76 | 0.052 |
| change_point | outfox/llama70b_chat | 1000 | 31.78 | 0.083 |
| change_point | outfox/llama7b_chat | 1000 | 19.73 | 0.051 |
| change_point | peerread/chatgpt | 1522 | 2.58 | 0.534 |
| change_point | peerread/llama | 1035 | 69.55 | 0.143 |
| change_point | peerread/llama2_13B_chat | 1522 | 17.69 | 0.125 |
| change_point | peerread/llama2_70B_chat | 1522 | 28.28 | 0.097 |
| change_point | peerread/llama2_7B_chat | 1522 | 20.84 | 0.095 |

## Human-only documents

*False alarm*: share of documents with a predicted boundary (any machine word, for the first-word rule). *Flagged*: mean share of words labelled machine.

| decoder | set | docs | false alarm | flagged |
|---|---|---|---|---|
| first_word | ASAP reviews | 1000 | 0.024 | 0.009 |
| first_word | OUTFOX essays | 1000 | 0.115 | 0.021 |
| first_word | test twins, PeerRead | 747 | 0.055 | 0.015 |
| first_word | test twins, OUTFOX | 997 | 0.145 | 0.027 |
| change_point | ASAP reviews | 1000 | 0.021 | 0.009 |
| change_point | OUTFOX essays | 1000 | 0.094 | 0.020 |
| change_point | test twins, PeerRead | 747 | 0.044 | 0.015 |
| change_point | test twins, OUTFOX | 997 | 0.103 | 0.025 |

## Deployment view

Mixed test documents plus the ASAP and OUTFOX human documents: is there machine text at all?

| decoder | accuracy | balanced | mixed detected | human quiet | MAE, mixed |
|---|---|---|---|---|---|
| first_word | 0.815 | 0.862 | 0.793 | 0.930 | 20.57 |
| change_point | 0.801 | 0.858 | 0.774 | 0.943 | 22.91 |

## Training

| epoch | loss | dev MAE | min |
|---|---|---|---|
| 1 | 0.1228 | 4.05 | 5.3 |
| 2 | 0.0228 | 2.98 | 5.3 |
| 3 | 0.0126 | 3.42 | 5.3 |
| 4 | 0.0089 | 2.75 | 5.3 |
| 5 | 0.0066 | 3.44 | 5.3 |
