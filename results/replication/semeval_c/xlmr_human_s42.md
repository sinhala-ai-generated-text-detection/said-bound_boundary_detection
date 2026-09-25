# Subtask C tagger: xlmr_human_s42

`xlm-roberta-base`, condition **human**, seed 42: 3649 mixed training documents (864,153 words) plus 1831 human-only documents (448,594 words). 5 epochs, lr 2e-05, batch 8; best epoch 4 on dev MAE (3.16); 15.7 min training.

## Subtask C test

| decoder | slice | docs | MAE | exact |
|---|---|---|---|---|
| first_word | overall | 11123 | 25.09 | 0.165 |
| first_word | outfox | 4000 | 27.95 | 0.133 |
| first_word | peerread | 7123 | 23.49 | 0.183 |
| first_word | outfox/gpt4 | 1000 | 9.19 | 0.399 |
| first_word | outfox/llama13b_chat | 1000 | 27.01 | 0.039 |
| first_word | outfox/llama70b_chat | 1000 | 46.16 | 0.049 |
| first_word | outfox/llama7b_chat | 1000 | 29.44 | 0.043 |
| first_word | peerread/chatgpt | 1522 | 3.63 | 0.502 |
| first_word | peerread/llama | 1035 | 53.38 | 0.138 |
| first_word | peerread/llama2_13B_chat | 1522 | 18.82 | 0.095 |
| first_word | peerread/llama2_70B_chat | 1522 | 27.78 | 0.085 |
| first_word | peerread/llama2_7B_chat | 1522 | 23.40 | 0.081 |
| change_point | overall | 11123 | 27.86 | 0.163 |
| change_point | outfox | 4000 | 30.95 | 0.130 |
| change_point | peerread | 7123 | 26.12 | 0.182 |
| change_point | outfox/gpt4 | 1000 | 7.89 | 0.400 |
| change_point | outfox/llama13b_chat | 1000 | 27.38 | 0.037 |
| change_point | outfox/llama70b_chat | 1000 | 56.55 | 0.049 |
| change_point | outfox/llama7b_chat | 1000 | 31.97 | 0.032 |
| change_point | peerread/chatgpt | 1522 | 3.47 | 0.503 |
| change_point | peerread/llama | 1035 | 65.29 | 0.125 |
| change_point | peerread/llama2_13B_chat | 1522 | 19.39 | 0.093 |
| change_point | peerread/llama2_70B_chat | 1522 | 30.71 | 0.085 |
| change_point | peerread/llama2_7B_chat | 1522 | 24.30 | 0.083 |

## Human-only documents

*False alarm*: share of documents with a predicted boundary (any machine word, for the first-word rule). *Flagged*: mean share of words labelled machine.

| decoder | set | docs | false alarm | flagged |
|---|---|---|---|---|
| first_word | ASAP reviews | 1000 | 0.062 | 0.021 |
| first_word | OUTFOX essays | 1000 | 0.109 | 0.024 |
| first_word | test twins, PeerRead | 747 | 0.098 | 0.035 |
| first_word | test twins, OUTFOX | 997 | 0.136 | 0.033 |
| change_point | ASAP reviews | 1000 | 0.052 | 0.022 |
| change_point | OUTFOX essays | 1000 | 0.088 | 0.023 |
| change_point | test twins, PeerRead | 747 | 0.083 | 0.034 |
| change_point | test twins, OUTFOX | 997 | 0.103 | 0.031 |

## Deployment view

Mixed test documents plus the ASAP and OUTFOX human documents: is there machine text at all?

| decoder | accuracy | balanced | mixed detected | human quiet | MAE, mixed |
|---|---|---|---|---|---|
| first_word | 0.805 | 0.850 | 0.785 | 0.914 | 25.09 |
| change_point | 0.794 | 0.849 | 0.769 | 0.930 | 27.86 |

## Training

| epoch | loss | dev MAE | min |
|---|---|---|---|
| 1 | 0.1521 | 4.19 | 3.2 |
| 2 | 0.0338 | 5.33 | 3.0 |
| 3 | 0.0163 | 3.27 | 3.2 |
| 4 | 0.0111 | 3.16 | 3.2 |
| 5 | 0.0079 | 3.48 | 3.0 |
