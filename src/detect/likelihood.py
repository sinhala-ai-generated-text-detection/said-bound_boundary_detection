"""Per-sentence likelihood features from a multilingual masked LM.

Motivation. Machine text is, by construction, high-probability text under a
language model; human text is more surprising. For *boundary* detection the
useful quantity is not a sentence's likelihood in isolation but the
**discontinuity** in likelihood across neighbouring sentences, since that is
what a change of author produces. This is the feature family SeqXGPT uses for
sentence-level detection, applied here to a low-resource language where it has
not been tested.

Why a masked LM rather than a causal one. The features must be computed in
Sinhala, and XLM-R is the one model in reach with guaranteed Sinhala coverage
(it is already used as the tagger's encoder, so no extra model is downloaded).
Causal multilingual models of comparable size do not reliably list Sinhala.

Scoring. Exact pseudo-log-likelihood needs one forward pass per token, which is
infeasible here (~1.2M passes). We use **strided masking**: every k-th token is
masked in the same pass, so k passes score the whole document. Tokens never see
their own identity, so the estimate stays honest, and the cost drops by the
tokenizer's sequence length.

Likelihoods are computed with the whole document in context, which is what makes
the cross-sentence delta features meaningful.

    python src/detect/likelihood.py --probe      # is there any signal at all?
    python src/detect/likelihood.py --build      # cache features for the corpus
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import load_dataset, role_map  # noqa: E402
from utils import force_utf8_stdout, load_config  # noqa: E402

CACHE = Path("cache/likelihood_features.jsonl")

# Order matters: it is the feature-vector layout used by consumers.
FEATURE_NAMES = [
    "mean_logp",      # how expected the sentence is overall
    "std_logp",       # how uneven that expectation is
    "min_logp",       # the single most surprising token
    "p10_logp",       # robust version of the same
    "mean_rank",      # log rank of the observed token among predictions
    "mean_entropy",   # how uncertain the model was at these positions
    "frac_top1",      # fraction of tokens the model would itself have chosen
    "n_tokens",
]
DELTA_NAMES = ["d_prev_mean_logp", "d_next_mean_logp", "d_prev_entropy",
               "abs_d_prev_mean_logp", "is_first", "is_last"]


class LikelihoodScorer:
    """Strided-mask pseudo-log-likelihood over a document, per sentence."""

    # batch_size is 1 by default on purpose. XLM-R's vocabulary is ~250k, so a
    # (batch, 512, 250002) logits tensor is ~4 GB in fp32 before any softmax --
    # it OOMs a 6 GB card at batch 8. We also slice to the masked positions
    # before computing log-softmax or entropy, so the wide tensor never exists
    # for the full sequence.
    def __init__(self, model_name="xlm-roberta-base", stride=8,
                 max_length=512, device=None, batch_size=1):
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        self.torch = torch
        self.stride = stride
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.eval().to(self.device)
        self.mask_id = self.tok.mask_token_id

    def _encode(self, sentences):
        """Document token ids plus the sentence index of every token."""
        ids, owner = [], []
        for si, s in enumerate(sentences):
            t = self.tok.encode(s, add_special_tokens=False)
            if not t:                      # keep alignment for empty strings
                t = [self.tok.unk_token_id]
            ids.extend(t)
            owner.extend([si] * len(t))
        budget = self.max_length - 2
        if len(ids) > budget:
            ids, owner = ids[:budget], owner[:budget]
        full = [self.tok.bos_token_id] + ids + [self.tok.eos_token_id]
        owner = [-1] + owner + [-1]
        return full, owner

    def score_document(self, sentences):
        torch = self.torch
        ids, owner = self._encode(sentences)
        n = len(ids)
        base = torch.tensor(ids, device=self.device)
        attn = torch.ones(n, dtype=torch.long, device=self.device)

        logp = np.full(n, np.nan)
        rank = np.full(n, np.nan)
        entropy = np.full(n, np.nan)
        top1 = np.zeros(n, dtype=bool)

        # All stride offsets go through the encoder in ONE batch. The vocabulary
        # projection is then applied only at the masked positions: running the
        # full lm_head over every position would build a (batch, 512, 250002)
        # tensor, which is both the memory blow-up and ~8x wasted compute, since
        # only about one token in `stride` is ever read.
        offsets = list(range(self.stride))
        positions = [[i for i in range(1, n - 1) if i % self.stride == off]
                     for off in offsets]
        batch = base.unsqueeze(0).repeat(len(offsets), 1).clone()
        for r, pos in enumerate(positions):
            if pos:
                batch[r, pos] = self.mask_id

        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.float16,
                                enabled=self.device == "cuda"):
                hidden = self.model.roberta(
                    input_ids=batch,
                    attention_mask=attn.unsqueeze(0).repeat(len(offsets), 1),
                ).last_hidden_state                       # (S, T, 768)

            rows, flat_pos = [], []
            for r, pos in enumerate(positions):
                if pos:
                    rows.append(hidden[r, torch.tensor(pos, device=self.device)])
                    flat_pos.extend(pos)
            if not flat_pos:
                return [[0.0] * len(FEATURE_NAMES) for _ in sentences]
            gathered = torch.cat(rows, dim=0)             # (n_masked, 768)

            # Project in slices so the (n_masked, 250002) matrix stays bounded.
            CH = 256
            for s0 in range(0, gathered.size(0), CH):
                sl = slice(s0, min(s0 + CH, gathered.size(0)))
                logits = self.model.lm_head(gathered[sl]).float()
                lsm = torch.log_softmax(logits, dim=-1)
                idx = flat_pos[sl]
                true = base[torch.tensor(idx, device=self.device)]
                lp = lsm.gather(1, true.unsqueeze(1)).squeeze(1)
                rk = (lsm > lp.unsqueeze(1)).sum(1)
                ent = -(lsm.exp() * lsm).sum(-1)
                logp[idx] = lp.cpu().numpy()
                rank[idx] = rk.cpu().numpy()
                entropy[idx] = ent.cpu().numpy()
                top1[idx] = (rk == 0).cpu().numpy()
                del logits, lsm, lp, rk, ent
            del hidden, gathered, rows

        feats = []
        for si in range(len(sentences)):
            idx = [i for i in range(n) if owner[i] == si and not np.isnan(logp[i])]
            if not idx:
                feats.append([0.0] * len(FEATURE_NAMES))
                continue
            lp = logp[idx]
            feats.append([
                float(lp.mean()), float(lp.std()), float(lp.min()),
                float(np.percentile(lp, 10)),
                float(np.log1p(rank[idx]).mean()),
                float(entropy[idx].mean()),
                float(top1[idx].mean()),
                float(len(idx)),
            ])
        return feats


def add_deltas(feats):
    """Neighbour-difference features: the discontinuity signal itself."""
    out = []
    n = len(feats)
    for i, f in enumerate(feats):
        prev = feats[i - 1] if i > 0 else f
        nxt = feats[i + 1] if i < n - 1 else f
        d_prev = f[0] - prev[0]
        out.append(list(f) + [
            d_prev, f[0] - nxt[0], f[5] - prev[5],
            abs(d_prev), 1.0 if i == 0 else 0.0, 1.0 if i == n - 1 else 0.0,
        ])
    return out


# ------------------------------------------------------------------ probe ---
def probe(cfg, n_docs=120):
    """Cheap decisive test: do human and machine sentences separate at all?"""
    ds = load_dataset(cfg)
    docs = ds.filter(split="dev").docs[:n_docs]
    sc = LikelihoodScorer()
    print(f"scoring {len(docs)} documents on {sc.device} ...")

    H, A, dsame, dchange = [], [], [], []
    for k, d in enumerate(docs):
        f = sc.score_document(d.sentences)
        for i, (row, lab) in enumerate(zip(f, d.labels)):
            (A if lab else H).append(row[0])
            if i > 0:
                delta = abs(row[0] - f[i - 1][0])
                (dchange if d.labels[i] != d.labels[i - 1] else dsame).append(delta)
        if (k + 1) % 25 == 0:
            print(f"  {k + 1}/{len(docs)}", flush=True)

    H, A = np.array(H), np.array(A)
    dsame, dchange = np.array(dsame), np.array(dchange)

    def coh(a, b):
        s = np.sqrt((a.var() + b.var()) / 2)
        return (b.mean() - a.mean()) / s if s else 0.0

    print("\n--- mean per-token log-likelihood ---")
    print(f"  human   n={len(H):5d}  mean={H.mean():.4f}  sd={H.std():.4f}")
    print(f"  machine n={len(A):5d}  mean={A.mean():.4f}  sd={A.std():.4f}")
    print(f"  Cohen's d = {coh(H, A):+.3f}   (machine minus human)")
    print("\n--- |Δ log-likelihood| across adjacent sentence pairs ---")
    print(f"  same author   n={len(dsame):5d}  mean={dsame.mean():.4f}")
    print(f"  author change n={len(dchange):5d}  mean={dchange.mean():.4f}")
    print(f"  Cohen's d = {coh(dsame, dchange):+.3f}   (change minus same)")
    print("\nVERDICT:", "signal present, worth building"
          if abs(coh(H, A)) > 0.2 or abs(coh(dsame, dchange)) > 0.15
          else "no usable separation - do not build on this")


def build(cfg, limit=None):
    ds = load_dataset(cfg)
    docs = ds.docs[:limit] if limit else ds.docs
    done = set()
    if CACHE.exists():
        for r in (json.loads(l) for l in CACHE.open(encoding="utf-8") if l.strip()):
            done.add(r["record_id"])
    print(f"{len(docs)} documents, {len(done)} already cached")
    sc = LikelihoodScorer()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with CACHE.open("a", encoding="utf-8") as fh:
        for k, d in enumerate(docs):
            if d.record_id in done:
                continue
            feats = add_deltas(sc.score_document(d.sentences))
            fh.write(json.dumps({"record_id": d.record_id,
                                 "features": feats}, ensure_ascii=False) + "\n")
            written += 1
            if written % 200 == 0:
                fh.flush()
                print(f"  {k + 1}/{len(docs)} (+{written})", flush=True)
    print(f"wrote {written} -> {CACHE}")


def load_features():
    """record_id -> list of per-sentence feature vectors."""
    out = {}
    if not CACHE.exists():
        return out
    for line in CACHE.open(encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            out[r["record_id"]] = r["features"]
    return out


def main():
    force_utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--n-docs", type=int, default=120)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config()
    if a.probe:
        probe(cfg, a.n_docs)
    elif a.build:
        build(cfg, a.limit)
    else:
        ap.error("pass --probe or --build")


if __name__ == "__main__":
    main()
