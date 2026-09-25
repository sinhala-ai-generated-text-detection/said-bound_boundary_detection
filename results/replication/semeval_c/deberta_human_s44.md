# Subtask C tagger: deberta_human_s44

`microsoft/deberta-v3-base`, condition **human**, seed 44: 3649 mixed training documents (864,153 words) plus 1831 human-only documents (448,594 words). 5 epochs, lr 2e-05, batch 8; best epoch 5 on dev MAE (2.54); 26.5 min training.

## Subtask C test

| decoder | slice | docs | MAE | exact |
|---|---|---|---|---|
| first_word | overall | 11123 | 17.60 | 0.198 |
| first_word | outfox | 4000 | 20.69 | 0.162 |
| first_word | peerread | 7123 | 15.87 | 0.219 |
| first_word | outfox/gpt4 | 1000 | 8.53 | 0.433 |
| first_word | outfox/llama13b_chat | 1000 | 22.84 | 0.051 |
| first_word | outfox/llama70b_chat | 1000 | 28.38 | 0.096 |
| first_word | outfox/llama7b_chat | 1000 | 23.00 | 0.066 |
| first_word | peerread/chatgpt | 1522 | 2.44 | 0.534 |
| first_word | peerread/llama | 1035 | 32.45 | 0.219 |
| first_word | peerread/llama2_13B_chat | 1522 | 13.62 | 0.134 |
| first_word | peerread/llama2_70B_chat | 1522 | 19.81 | 0.112 |
| first_word | peerread/llama2_7B_chat | 1522 | 16.36 | 0.097 |
| change_point | overall | 11123 | 19.09 | 0.196 |
| change_point | outfox | 4000 | 21.64 | 0.157 |
| change_point | peerread | 7123 | 17.66 | 0.218 |
| change_point | outfox/gpt4 | 1000 | 7.05 | 0.436 |
| change_point | outfox/llama13b_chat | 1000 | 23.56 | 0.049 |
| change_point | outfox/llama70b_chat | 1000 | 32.74 | 0.091 |
| change_point | outfox/llama7b_chat | 1000 | 23.22 | 0.052 |
| change_point | peerread/chatgpt | 1522 | 2.33 | 0.535 |
| change_point | peerread/llama | 1035 | 39.92 | 0.214 |
| change_point | peerread/llama2_13B_chat | 1522 | 14.25 | 0.131 |
| change_point | peerread/llama2_70B_chat | 1522 | 21.56 | 0.115 |
| change_point | peerread/llama2_7B_chat | 1522 | 17.39 | 0.094 |

## Human-only documents

*False alarm*: share of documents with a predicted boundary (any machine word, for the first-word rule). *Flagged*: mean share of words labelled machine.

| decoder | set | docs | false alarm | flagged |
|---|---|---|---|---|
| first_word | ASAP reviews | 1000 | 0.034 | 0.011 |
| first_word | OUTFOX essays | 1000 | 0.157 | 0.037 |
| first_word | test twins, PeerRead | 747 | 0.070 | 0.021 |
| first_word | test twins, OUTFOX | 997 | 0.170 | 0.052 |
| change_point | ASAP reviews | 1000 | 0.028 | 0.012 |
| change_point | OUTFOX essays | 1000 | 0.122 | 0.037 |
| change_point | test twins, PeerRead | 747 | 0.058 | 0.021 |
| change_point | test twins, OUTFOX | 997 | 0.141 | 0.052 |

## Deployment view

Mixed test documents plus the ASAP and OUTFOX human documents: is there machine text at all?

| decoder | accuracy | balanced | mixed detected | human quiet | MAE, mixed |
|---|---|---|---|---|---|
| first_word | 0.850 | 0.872 | 0.839 | 0.904 | 17.60 |
| change_point | 0.842 | 0.876 | 0.826 | 0.925 | 19.09 |

## Training

| epoch | loss | dev MAE | min |
|---|---|---|---|
| 1 | 0.1461 | 3.64 | 5.3 |
| 2 | 0.0226 | 3.19 | 5.3 |
| 3 | 0.0125 | 2.79 | 5.3 |
| 4 | 0.0089 | 3.05 | 5.2 |
| 5 | 0.0066 | 2.54 | 5.3 |
