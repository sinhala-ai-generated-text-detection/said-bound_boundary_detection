"""Word-level human/machine tagger for Subtask C.

Each word gets a label from the hidden state of its first subword token.
Documents longer than the encoder window are packed into chunks on word
boundaries (as src/detect/transformer.py does on sentence boundaries), so a
word is never split across chunks and never dropped: an over-long word keeps
its first ``budget`` tokens, which still include the one that is scored.
Predictions are reassembled by (document, word index).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from corpus import gold_labels, words_of


def encode_doc(tok, words, labels, max_length: int):
    """Chunks of one document: (input_ids, first-token positions, word
    indices, labels). Words with no token are absent and are filled later."""
    enc = tok(words, is_split_into_words=True, add_special_tokens=False)
    per_word = [[] for _ in words]
    for t, w in zip(enc["input_ids"], enc.word_ids()):
        if w is not None:
            per_word[w].append(t)
    budget = max_length - 2
    chunks, ids, pos, widx, lab = [], [], [], [], []

    def flush():
        if pos:
            chunks.append(([tok.cls_token_id] + ids + [tok.sep_token_id],
                           [p + 1 for p in pos], list(widx), list(lab)))

    for i, toks in enumerate(per_word):
        if not toks:
            continue
        toks = toks[:budget]
        if len(ids) + len(toks) > budget and pos:
            flush()
            ids, pos, widx, lab = [], [], [], []
        pos.append(len(ids))
        ids.extend(toks)
        widx.append(i)
        lab.append(int(labels[i]))
    flush()
    return chunks


def collate(batch, pad_id: int):
    L = max(len(c[0]) for c in batch)
    M = max(len(c[1]) for c in batch)
    B = len(batch)
    ids = torch.full((B, L), pad_id, dtype=torch.long)
    attn = torch.zeros((B, L), dtype=torch.long)
    pos = torch.zeros((B, M), dtype=torch.long)
    lab = torch.full((B, M), -100, dtype=torch.long)
    meta = []
    for b, (x, p, w, y, di) in enumerate(batch):
        ids[b, :len(x)] = torch.tensor(x)
        attn[b, :len(x)] = 1
        pos[b, :len(p)] = torch.tensor(p)
        lab[b, :len(y)] = torch.tensor(y)
        meta.append((di, w))
    return ids, attn, pos, lab, meta


class WordTagger:
    def __init__(self, model_name="microsoft/deberta-v3-base", max_length=512,
                 batch_size=8, lr=2e-5, epochs=5, warmup_ratio=0.1,
                 weight_decay=0.01, seed=42, class_weight=None,
                 bf16=True, max_grad_norm=1.0):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.seed = seed
        self.class_weight = class_weight
        # bf16 rather than fp16: DeBERTa-v3 is known to overflow in fp16, and
        # bf16 needs no loss scaling.
        self.bf16 = bf16
        self.max_grad_norm = max_grad_norm
        self.best_epoch = self.best_score = None
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.history = []

    def chunks(self, docs, labelled=True):
        out = []
        for di, d in enumerate(docs):
            w = words_of(d["text"])
            y = gold_labels(len(w), d["boundary"]) if labelled else [0] * len(w)
            for c in encode_doc(self.tok, w, y, self.max_length):
                out.append((*c, di))
        return out

    def _loader(self, items, shuffle, gen=None):
        return DataLoader(items, batch_size=self.batch_size, shuffle=shuffle,
                          generator=gen,
                          collate_fn=lambda b: collate(b, self.tok.pad_token_id))

    def _autocast(self):
        return torch.autocast("cuda", dtype=torch.bfloat16,
                              enabled=self.bf16 and self.device.type == "cuda")

    def fit(self, docs, dev_docs, dev_score, log=print):
        """Train on docs; keep the epoch with the lowest dev_score(self)."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        gen = torch.Generator().manual_seed(self.seed)
        # transformers 5 loads a checkpoint in its stored dtype, and
        # DeBERTa-v3 is stored in fp16: AdamW's epsilon underflows there and
        # the first step turns every weight to NaN. Keep fp32 master weights.
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name, num_labels=2,
            dtype=torch.float32).to(self.device)
        items = self.chunks(docs)
        loader = self._loader(items, True, gen)
        steps = len(loader) * self.epochs
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        sched = get_linear_schedule_with_warmup(
            opt, int(steps * self.warmup_ratio), steps)
        w = (torch.tensor(self.class_weight, dtype=torch.float,
                          device=self.device)
             if self.class_weight is not None else None)
        lossf = nn.CrossEntropyLoss(weight=w, ignore_index=-100)
        log(f"  device={self.device} docs={len(docs)} chunks={len(items)} "
            f"steps={steps}")
        best, best_state = None, None
        for ep in range(self.epochs):
            t0 = time.time()
            self.model.train()
            tot, nb = 0.0, 0
            for ids, attn, pos, lab, _ in loader:
                ids, attn = ids.to(self.device), attn.to(self.device)
                pos, lab = pos.to(self.device), lab.to(self.device)
                with self._autocast():
                    logits = self.model(input_ids=ids,
                                        attention_mask=attn).logits
                g = logits.float().gather(
                    1, pos.unsqueeze(-1).expand(-1, -1, 2))
                loss = lossf(g.reshape(-1, 2), lab.reshape(-1))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.max_grad_norm)
                opt.step()
                sched.step()
                tot += loss.item()
                nb += 1
            score = dev_score(self)
            mark = ""
            if best is None or score < best:
                best, self.best_epoch = score, ep + 1
                best_state = {k: v.detach().cpu().clone()
                              for k, v in self.model.state_dict().items()}
                mark = "  *"
            self.history.append({"epoch": ep + 1, "loss": tot / max(1, nb),
                                 "dev_mae": score,
                                 "minutes": (time.time() - t0) / 60})
            log(f"  epoch {ep + 1}/{self.epochs} loss={tot / max(1, nb):.4f} "
                f"dev MAE={score:.3f} ({(time.time() - t0) / 60:.1f} min)"
                f"{mark}")
        self.best_score = best
        self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    @torch.no_grad()
    def predict_proba(self, docs) -> list[np.ndarray]:
        """P(machine) per word, NaN for words with no token."""
        self.model.eval()
        out = [np.full(len(words_of(d["text"])), np.nan, dtype=np.float32)
               for d in docs]
        for ids, attn, pos, _, meta in self._loader(
                self.chunks(docs, labelled=False), False):
            with self._autocast():
                logits = self.model(input_ids=ids.to(self.device),
                                    attention_mask=attn.to(self.device)).logits
            p = torch.softmax(logits.float(), -1)[..., 1]
            p = p.gather(1, pos.to(self.device)).cpu().numpy()
            for b, (di, widx) in enumerate(meta):
                out[di][widx] = p[b, :len(widx)]
        return out

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tok.save_pretrained(path)
