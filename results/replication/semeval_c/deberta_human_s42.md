# Subtask C tagger: deberta_human_s42

`microsoft/deberta-v3-base`, condition **human**, seed 42: 3649 mixed training documents (864,153 words) plus 1831 human-only documents (448,594 words). 5 epochs, lr 2e-05, batch 8; best epoch 2 on dev MAE (2.55); 26.6 min training.

## Subtask C test

| decoder | slice | docs | MAE | exact |
|---|---|---|---|---|
| first_word | overall | 11123 | 22.08 | 0.192 |
| first_word | outfox | 4000 | 23.49 | 0.163 |
| first_word | peerread | 7123 | 21.29 | 0.208 |
| first_word | outfox/gpt4 | 1000 | 9.65 | 0.422 |
| first_word | outfox/llama13b_chat | 1000 | 24.71 | 0.059 |
| first_word | outfox/llama70b_chat | 1000 | 35.94 | 0.096 |
| first_word | outfox/llama7b_chat | 1000 | 23.67 | 0.073 |
| first_word | peerread/chatgpt | 1522 | 2.86 | 0.532 |
| first_word | peerread/llama | 1035 | 42.87 | 0.239 |
| first_word | peerread/llama2_13B_chat | 1522 | 18.67 | 0.112 |
| first_word | peerread/llama2_70B_chat | 1522 | 27.41 | 0.085 |
| first_word | peerread/llama2_7B_chat | 1522 | 21.56 | 0.082 |
| change_point | overall | 11123 | 25.51 | 0.189 |
| change_point | outfox | 4000 | 27.24 | 0.155 |
| change_point | peerread | 7123 | 24.53 | 0.208 |
| change_point | outfox/gpt4 | 1000 | 7.80 | 0.423 |
| change_point | outfox/llama13b_chat | 1000 | 26.37 | 0.051 |
| change_point | outfox/llama70b_chat | 1000 | 45.93 | 0.086 |
| change_point | outfox/llama7b_chat | 1000 | 28.88 | 0.061 |
| change_point | peerread/chatgpt | 1522 | 2.78 | 0.535 |
| change_point | peerread/llama | 1035 | 53.15 | 0.230 |
| change_point | peerread/llama2_13B_chat | 1522 | 19.89 | 0.113 |
| change_point | peerread/llama2_70B_chat | 1522 | 32.50 | 0.089 |
| change_point | peerread/llama2_7B_chat | 1522 | 23.49 | 0.078 |

## Human-only documents

*False alarm*: share of documents with a predicted boundary (any machine word, for the first-word rule). *Flagged*: mean share of words labelled machine.

| decoder | set | docs | false alarm | flagged |
|---|---|---|---|---|
| first_word | ASAP reviews | 1000 | 0.026 | 0.008 |
| first_word | OUTFOX essays | 1000 | 0.137 | 0.022 |
| first_word | test twins, PeerRead | 747 | 0.047 | 0.008 |
| first_word | test twins, OUTFOX | 997 | 0.166 | 0.034 |
| change_point | ASAP reviews | 1000 | 0.020 | 0.008 |
| change_point | OUTFOX essays | 1000 | 0.108 | 0.021 |
| change_point | test twins, PeerRead | 747 | 0.036 | 0.007 |
| change_point | test twins, OUTFOX | 997 | 0.135 | 0.034 |

## Deployment view

Mixed test documents plus the ASAP and OUTFOX human documents: is there machine text at all?

| decoder | accuracy | balanced | mixed detected | human quiet | MAE, mixed |
|---|---|---|---|---|---|
| first_word | 0.817 | 0.858 | 0.798 | 0.918 | 22.08 |
| change_point | 0.801 | 0.856 | 0.777 | 0.936 | 25.51 |

## Training

| epoch | loss | dev MAE | min |
|---|---|---|---|
| 1 | 0.1399 | 3.56 | 5.4 |
| 2 | 0.0230 | 2.55 | 5.4 |
| 3 | 0.0126 | 3.23 | 5.3 |
| 4 | 0.0086 | 2.93 | 5.3 |
| 5 | 0.0060 | 3.03 | 5.3 |
