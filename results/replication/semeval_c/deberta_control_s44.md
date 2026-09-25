# Subtask C tagger: deberta_control_s44

`microsoft/deberta-v3-base`, condition **control**, seed 44: 3649 mixed training documents (864,153 words) plus 0 human-only documents (0 words). 5 epochs, lr 2e-05, batch 8; best epoch 4 on dev MAE (3.09); 17.3 min training.

## Subtask C test

| decoder | slice | docs | MAE | exact |
|---|---|---|---|---|
| first_word | overall | 11123 | 16.76 | 0.213 |
| first_word | outfox | 4000 | 25.17 | 0.156 |
| first_word | peerread | 7123 | 12.04 | 0.245 |
| first_word | outfox/gpt4 | 1000 | 9.99 | 0.429 |
| first_word | outfox/llama13b_chat | 1000 | 32.97 | 0.061 |
| first_word | outfox/llama70b_chat | 1000 | 26.55 | 0.082 |
| first_word | outfox/llama7b_chat | 1000 | 31.18 | 0.052 |
| first_word | peerread/chatgpt | 1522 | 3.52 | 0.539 |
| first_word | peerread/llama | 1035 | 11.36 | 0.304 |
| first_word | peerread/llama2_13B_chat | 1522 | 13.73 | 0.152 |
| first_word | peerread/llama2_70B_chat | 1522 | 15.95 | 0.131 |
| first_word | peerread/llama2_7B_chat | 1522 | 15.44 | 0.119 |
| change_point | overall | 11123 | 14.20 | 0.218 |
| change_point | outfox | 4000 | 20.32 | 0.157 |
| change_point | peerread | 7123 | 10.76 | 0.252 |
| change_point | outfox/gpt4 | 1000 | 7.65 | 0.434 |
| change_point | outfox/llama13b_chat | 1000 | 25.73 | 0.063 |
| change_point | outfox/llama70b_chat | 1000 | 23.33 | 0.080 |
| change_point | outfox/llama7b_chat | 1000 | 24.58 | 0.050 |
| change_point | peerread/chatgpt | 1522 | 2.46 | 0.543 |
| change_point | peerread/llama | 1035 | 12.03 | 0.305 |
| change_point | peerread/llama2_13B_chat | 1522 | 11.54 | 0.161 |
| change_point | peerread/llama2_70B_chat | 1522 | 15.36 | 0.136 |
| change_point | peerread/llama2_7B_chat | 1522 | 12.81 | 0.131 |

## Human-only documents

*False alarm*: share of documents with a predicted boundary (any machine word, for the first-word rule). *Flagged*: mean share of words labelled machine.

| decoder | set | docs | false alarm | flagged |
|---|---|---|---|---|
| first_word | ASAP reviews | 1000 | 0.794 | 0.244 |
| first_word | OUTFOX essays | 1000 | 0.939 | 0.262 |
| first_word | test twins, PeerRead | 747 | 0.857 | 0.241 |
| first_word | test twins, OUTFOX | 997 | 0.952 | 0.283 |
| change_point | ASAP reviews | 1000 | 0.712 | 0.238 |
| change_point | OUTFOX essays | 1000 | 0.894 | 0.256 |
| change_point | test twins, PeerRead | 747 | 0.784 | 0.233 |
| change_point | test twins, OUTFOX | 997 | 0.904 | 0.278 |

## Deployment view

Mixed test documents plus the ASAP and OUTFOX human documents: is there machine text at all?

| decoder | accuracy | balanced | mixed detected | human quiet | MAE, mixed |
|---|---|---|---|---|---|
| first_word | 0.861 | 0.564 | 0.994 | 0.133 | 16.76 |
| change_point | 0.868 | 0.594 | 0.991 | 0.197 | 14.20 |

## Training

| epoch | loss | dev MAE | min |
|---|---|---|---|
| 1 | 0.1982 | 8.91 | 3.4 |
| 2 | 0.0368 | 4.30 | 3.5 |
| 3 | 0.0189 | 3.44 | 3.5 |
| 4 | 0.0121 | 3.09 | 3.5 |
| 5 | 0.0090 | 3.61 | 3.4 |
