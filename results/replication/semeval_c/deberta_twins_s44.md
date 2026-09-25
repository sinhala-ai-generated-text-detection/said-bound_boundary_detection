# Subtask C tagger: deberta_twins_s44

`microsoft/deberta-v3-base`, condition **twins**, seed 44: 3649 mixed training documents (864,153 words) plus 1831 human-only documents (449,892 words). 5 epochs, lr 2e-05, batch 8; best epoch 4 on dev MAE (2.90); 26.5 min training.

## Subtask C test

| decoder | slice | docs | MAE | exact |
|---|---|---|---|---|
| first_word | overall | 11123 | 20.03 | 0.199 |
| first_word | outfox | 4000 | 21.85 | 0.149 |
| first_word | peerread | 7123 | 19.00 | 0.227 |
| first_word | outfox/gpt4 | 1000 | 7.11 | 0.414 |
| first_word | outfox/llama13b_chat | 1000 | 22.08 | 0.045 |
| first_word | outfox/llama70b_chat | 1000 | 37.52 | 0.078 |
| first_word | outfox/llama7b_chat | 1000 | 20.71 | 0.061 |
| first_word | peerread/chatgpt | 1522 | 2.49 | 0.525 |
| first_word | peerread/llama | 1035 | 44.66 | 0.229 |
| first_word | peerread/llama2_13B_chat | 1522 | 14.84 | 0.150 |
| first_word | peerread/llama2_70B_chat | 1522 | 23.12 | 0.119 |
| first_word | peerread/llama2_7B_chat | 1522 | 18.11 | 0.112 |
| change_point | overall | 11123 | 22.63 | 0.196 |
| change_point | outfox | 4000 | 25.31 | 0.147 |
| change_point | peerread | 7123 | 21.13 | 0.224 |
| change_point | outfox/gpt4 | 1000 | 6.45 | 0.417 |
| change_point | outfox/llama13b_chat | 1000 | 23.14 | 0.043 |
| change_point | outfox/llama70b_chat | 1000 | 47.54 | 0.074 |
| change_point | outfox/llama7b_chat | 1000 | 24.09 | 0.053 |
| change_point | peerread/chatgpt | 1522 | 2.45 | 0.525 |
| change_point | peerread/llama | 1035 | 52.78 | 0.223 |
| change_point | peerread/llama2_13B_chat | 1522 | 15.80 | 0.145 |
| change_point | peerread/llama2_70B_chat | 1522 | 25.10 | 0.116 |
| change_point | peerread/llama2_7B_chat | 1522 | 19.63 | 0.113 |

## Human-only documents

*False alarm*: share of documents with a predicted boundary (any machine word, for the first-word rule). *Flagged*: mean share of words labelled machine.

| decoder | set | docs | false alarm | flagged |
|---|---|---|---|---|
| first_word | ASAP reviews | 1000 | 0.029 | 0.012 |
| first_word | OUTFOX essays | 1000 | 0.073 | 0.009 |
| first_word | test twins, PeerRead | 747 | 0.021 | 0.004 |
| first_word | test twins, OUTFOX | 997 | 0.082 | 0.014 |
| change_point | ASAP reviews | 1000 | 0.025 | 0.011 |
| change_point | OUTFOX essays | 1000 | 0.060 | 0.009 |
| change_point | test twins, PeerRead | 747 | 0.020 | 0.004 |
| change_point | test twins, OUTFOX | 997 | 0.063 | 0.013 |

## Deployment view

Mixed test documents plus the ASAP and OUTFOX human documents: is there machine text at all?

| decoder | accuracy | balanced | mixed detected | human quiet | MAE, mixed |
|---|---|---|---|---|---|
| first_word | 0.824 | 0.875 | 0.801 | 0.949 | 20.03 |
| change_point | 0.813 | 0.872 | 0.786 | 0.958 | 22.63 |

## Training

| epoch | loss | dev MAE | min |
|---|---|---|---|
| 1 | 0.1451 | 3.14 | 5.3 |
| 2 | 0.0252 | 3.23 | 5.2 |
| 3 | 0.0154 | 2.95 | 5.3 |
| 4 | 0.0091 | 2.90 | 5.3 |
| 5 | 0.0065 | 3.15 | 5.3 |
