#!/usr/bin/env bash
# Download the English replication data into data/english/ (gitignored).
# Needs git and gdown (pip install gdown; a separate environment is fine).
#
# The official task repository, github.com/mbzuai-nlp/SemEval2024-task8, has
# been blocked by a DMCA notice since 2026-09-14 (a wikiHow claim over the
# Subtask A/B data). Subtask C is taken from the M4GT-Bench release by the
# same group; only the two Subtask C files are downloaded, never Subtask A/B.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
mkdir -p data/english/semeval_c
cd data/english

# SemEval-2024 Task 8 Subtask C, from the M4GT-Bench Drive folder
# https://drive.google.com/drive/folders/1xC-n5hHiXmaCLmA0q3p8Cbyd7Dk0dpDE
gdown -q "https://drive.google.com/uc?id=1n2fgT071bgnHJVJG4syP_ZSI6GaqmlBt" \
    -O semeval_c/subtaskC_train_dev.jsonl
gdown -q "https://drive.google.com/uc?id=1e6a5GgyFEiVMkTUx_K3lN1zGsMhF58uC" \
    -O semeval_c/subtaskC_test.jsonl
sha256sum -c - <<'EOF'
857d91a9f637fe033e1aebe1e7f7daa6d1f05912e62557cda9bd1d1dd6eade79  semeval_c/subtaskC_train_dev.jsonl
86e11f1e494a29156213525a1e2598322de27d8a38b14ca620d4407096a9e0ea  semeval_c/subtaskC_test.jsonl
EOF

# PeerRead reviews only (checked, then found unusable: see the data report)
git clone -q --filter=blob:none --no-checkout https://github.com/allenai/PeerRead.git
git -C PeerRead checkout -q 9bb37751781a900cee9e74ec3105997732c8e8e5 -- \
    'data/iclr_2017/*/reviews/*' 'data/acl_2017/*/reviews/*' \
    'data/conll_2016/*/reviews/*'

# OUTFOX human essays
git clone -q --filter=blob:none --no-checkout https://github.com/ryuryukke/OUTFOX.git
git -C OUTFOX checkout -q 8dd6bfdec8e24ff6aeecae3015ab663f257e0350 -- data/common

# ASAP-Review (neulab/ReviewAdvisor), Apache-2.0
mkdir -p asap
gdown -q "https://drive.google.com/uc?id=1nJdljy468roUcKLbVwWUhMs7teirah75" \
    -O asap/dataset.zip
echo "13c5e5caa0db6ae43a755048183f7facf1ce73eac5eb7bf7cfb654501633216c  asap/dataset.zip" \
    | sha256sum -c -
(cd asap && unzip -q dataset.zip && rm -rf dataset.zip __MACOSX)
