# Subtask C tagger: xlmr_control_s42

`xlm-roberta-base`, condition **control**, seed 42: 3649 mixed training documents (864,153 words) plus 0 human-only documents (0 words). 5 epochs, lr 2e-05, batch 8; best epoch 2 on dev MAE (5.86); 10.4 min training.

## Subtask C test

| decoder | slice | docs | MAE | exact |
|---|---|---|---|---|
| first_word | overall | 11123 | 19.83 | 0.181 |
| first_word | outfox | 4000 | 25.43 | 0.140 |
| first_word | peerread | 7123 | 16.69 | 0.204 |
| first_word | outfox/gpt4 | 1000 | 12.32 | 0.383 |
| first_word | outfox/llama13b_chat | 1000 | 27.87 | 0.056 |
| first_word | outfox/llama70b_chat | 1000 | 32.04 | 0.066 |
| first_word | outfox/llama7b_chat | 1000 | 29.48 | 0.055 |
| first_word | peerread/chatgpt | 1522 | 5.82 | 0.487 |
| first_word | peerread/llama | 1035 | 15.36 | 0.270 |
| first_word | peerread/llama2_13B_chat | 1522 | 19.82 | 0.103 |
| first_word | peerread/llama2_70B_chat | 1522 | 21.23 | 0.102 |
| first_word | peerread/llama2_7B_chat | 1522 | 20.81 | 0.081 |
| change_point | overall | 11123 | 17.73 | 0.184 |
| change_point | outfox | 4000 | 22.78 | 0.139 |
| change_point | peerread | 7123 | 14.89 | 0.209 |
| change_point | outfox/gpt4 | 1000 | 9.26 | 0.394 |
| change_point | outfox/llama13b_chat | 1000 | 23.25 | 0.054 |
| change_point | outfox/llama70b_chat | 1000 | 32.85 | 0.058 |
| change_point | outfox/llama7b_chat | 1000 | 25.76 | 0.051 |
| change_point | peerread/chatgpt | 1522 | 4.52 | 0.490 |
| change_point | peerread/llama | 1035 | 15.73 | 0.271 |
| change_point | peerread/llama2_13B_chat | 1522 | 16.49 | 0.111 |
| change_point | peerread/llama2_70B_chat | 1522 | 19.87 | 0.105 |
| change_point | peerread/llama2_7B_chat | 1522 | 18.12 | 0.090 |

## Human-only documents

*False alarm*: share of documents with a predicted boundary (any machine word, for the first-word rule). *Flagged*: mean share of words labelled machine.

| decoder | set | docs | false alarm | flagged |
|---|---|---|---|---|
| first_word | ASAP reviews | 1000 | 0.762 | 0.284 |
| first_word | OUTFOX essays | 1000 | 0.549 | 0.117 |
| first_word | test twins, PeerRead | 747 | 0.762 | 0.255 |
| first_word | test twins, OUTFOX | 997 | 0.622 | 0.147 |
| change_point | ASAP reviews | 1000 | 0.686 | 0.279 |
| change_point | OUTFOX essays | 1000 | 0.512 | 0.116 |
| change_point | test twins, PeerRead | 747 | 0.704 | 0.250 |
| change_point | test twins, OUTFOX | 997 | 0.557 | 0.140 |

## Deployment view

Mixed test documents plus the ASAP and OUTFOX human documents: is there machine text at all?

| decoder | accuracy | balanced | mixed detected | human quiet | MAE, mixed |
|---|---|---|---|---|---|
| first_word | 0.878 | 0.660 | 0.975 | 0.345 | 19.83 |
| change_point | 0.882 | 0.685 | 0.970 | 0.401 | 17.73 |

## Training

| epoch | loss | dev MAE | min |
|---|---|---|---|
| 1 | 0.1957 | 7.13 | 2.1 |
| 2 | 0.0470 | 5.86 | 2.1 |
| 3 | 0.0300 | 6.14 | 2.0 |
| 4 | 0.0163 | 7.07 | 2.0 |
| 5 | 0.0124 | 6.61 | 2.0 |
