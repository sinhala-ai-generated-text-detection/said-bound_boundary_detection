# Subtask C tagger: deberta_twins_s42

`microsoft/deberta-v3-base`, condition **twins**, seed 42: 3649 mixed training documents (864,153 words) plus 1831 human-only documents (449,892 words). 5 epochs, lr 2e-05, batch 8; best epoch 5 on dev MAE (2.43); 26.5 min training.

## Subtask C test

| decoder | slice | docs | MAE | exact |
|---|---|---|---|---|
| first_word | overall | 11123 | 19.26 | 0.211 |
| first_word | outfox | 4000 | 23.12 | 0.159 |
| first_word | peerread | 7123 | 17.08 | 0.240 |
| first_word | outfox/gpt4 | 1000 | 8.32 | 0.415 |
| first_word | outfox/llama13b_chat | 1000 | 23.28 | 0.058 |
| first_word | outfox/llama70b_chat | 1000 | 35.16 | 0.088 |
| first_word | outfox/llama7b_chat | 1000 | 25.72 | 0.073 |
| first_word | peerread/chatgpt | 1522 | 2.39 | 0.535 |
| first_word | peerread/llama | 1035 | 38.57 | 0.254 |
| first_word | peerread/llama2_13B_chat | 1522 | 13.74 | 0.152 |
| first_word | peerread/llama2_70B_chat | 1522 | 20.70 | 0.132 |
| first_word | peerread/llama2_7B_chat | 1522 | 16.90 | 0.134 |
| change_point | overall | 11123 | 22.13 | 0.208 |
| change_point | outfox | 4000 | 27.32 | 0.152 |
| change_point | peerread | 7123 | 19.21 | 0.239 |
| change_point | outfox/gpt4 | 1000 | 6.77 | 0.417 |
| change_point | outfox/llama13b_chat | 1000 | 25.17 | 0.051 |
| change_point | outfox/llama70b_chat | 1000 | 46.51 | 0.081 |
| change_point | outfox/llama7b_chat | 1000 | 30.82 | 0.058 |
| change_point | peerread/chatgpt | 1522 | 2.34 | 0.535 |
| change_point | peerread/llama | 1035 | 48.14 | 0.243 |
| change_point | peerread/llama2_13B_chat | 1522 | 14.35 | 0.153 |
| change_point | peerread/llama2_70B_chat | 1522 | 22.45 | 0.133 |
| change_point | peerread/llama2_7B_chat | 1522 | 18.04 | 0.135 |

## Human-only documents

*False alarm*: share of documents with a predicted boundary (any machine word, for the first-word rule). *Flagged*: mean share of words labelled machine.

| decoder | set | docs | false alarm | flagged |
|---|---|---|---|---|
| first_word | ASAP reviews | 1000 | 0.056 | 0.017 |
| first_word | OUTFOX essays | 1000 | 0.160 | 0.033 |
| first_word | test twins, PeerRead | 747 | 0.050 | 0.010 |
| first_word | test twins, OUTFOX | 997 | 0.183 | 0.045 |
| change_point | ASAP reviews | 1000 | 0.038 | 0.017 |
| change_point | OUTFOX essays | 1000 | 0.135 | 0.033 |
| change_point | test twins, PeerRead | 747 | 0.041 | 0.009 |
| change_point | test twins, OUTFOX | 997 | 0.142 | 0.045 |

## Deployment view

Mixed test documents plus the ASAP and OUTFOX human documents: is there machine text at all?

| decoder | accuracy | balanced | mixed detected | human quiet | MAE, mixed |
|---|---|---|---|---|---|
| first_word | 0.844 | 0.864 | 0.835 | 0.892 | 19.26 |
| change_point | 0.836 | 0.867 | 0.821 | 0.913 | 22.13 |

## Training

| epoch | loss | dev MAE | min |
|---|---|---|---|
| 1 | 0.1359 | 4.40 | 5.3 |
| 2 | 0.0229 | 3.41 | 5.3 |
| 3 | 0.0133 | 2.48 | 5.3 |
| 4 | 0.0091 | 3.12 | 5.3 |
| 5 | 0.0067 | 2.43 | 5.3 |
