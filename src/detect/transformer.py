"""Document-level sentence tagger: XLM-RoBERTa with per-sentence heads.

The linear baseline failed at boundary localisation for a structural reason:
it scores each sentence independently, so it cannot represent *discontinuity*
between neighbours, which is the signal this task actually rests on. This model
fixes that by encoding the entire document in one pass, so every sentence
representation is built with its neighbours in attention range.

Encoding: a marker token is inserted before each sentence and that marker's
final hidden state becomes the sentence representation, which a linear head
maps to human/AI. This is the standard trick for sentence-level tagging inside
a token-level encoder, and it means the head sees a representation that has
already attended over the surrounding text.

XLM-R is used because it covers Sinhala; monolingual English encoders do not,
and its SentencePiece vocabulary handles Sinhala orthography without a
language-specific tokenizer.

Documents longer than the encoder window are handled by chunking on sentence
boundaries and encoding each chunk separately, so no sentence is ever silently
dropped - a truncated document would corrupt the label alignment.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent))


class DocEncoding:
    """One encoded chunk plus the sentence indices its markers correspond to."""

    def __init__(self, input_ids, attention_mask, marker_pos, sent_idx):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.marker_pos = marker_pos      # token positions of the markers
        self.sent_idx = sent_idx          # which sentence each marker is


class SentenceTaggingDataset(TorchDataset):
    """Turns Doc objects into encoder chunks with per-sentence label targets."""

    def __init__(self, docs, tokenizer, max_length: int = 512,
                 marker: str | None = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        # The marker must be a real token in the vocabulary. XLM-R's <s> is a
        # natural choice: it already means "segment start" to the model.
        self.marker = marker or tokenizer.cls_token
        self.marker_id = tokenizer.convert_tokens_to_ids(self.marker)
        self.items: list[tuple[DocEncoding, list[int], int]] = []
        self._build(docs)

    def _build(self, docs) -> None:
        tok = self.tokenizer
        budget = self.max_length - 2          # room for <s> ... </s>
        for di, d in enumerate(docs):
            # Pre-tokenize each sentence once; +1 for its marker.
            per_sent = [tok.encode(s, add_special_tokens=False)
                        for s in d.sentences]
            chunk_ids: list[int] = []
            chunk_markers: list[int] = []
            chunk_sidx: list[int] = []
            chunk_labels: list[int] = []

            def flush():
                if not chunk_markers:
                    return
                ids = [tok.bos_token_id] + chunk_ids + [tok.eos_token_id]
                # markers were recorded relative to chunk_ids; shift by the bos
                mk = [m + 1 for m in chunk_markers]
                enc = DocEncoding(ids, [1] * len(ids), mk, list(chunk_sidx))
                self.items.append((enc, list(chunk_labels), di))

            for si, ids in enumerate(per_sent):
                need = len(ids) + 1
                # A single sentence longer than the window is truncated rather
                # than dropped, so sentence/label alignment always holds.
                if need > budget:
                    ids = ids[: budget - 1]
                    need = len(ids) + 1
                if len(chunk_ids) + need > budget and chunk_markers:
                    flush()
                    chunk_ids, chunk_markers = [], []
                    chunk_sidx, chunk_labels = [], []
                chunk_markers.append(len(chunk_ids))
                chunk_ids.extend([self.marker_id] + ids)
                chunk_sidx.append(si)
                chunk_labels.append(int(d.labels[si]))
            flush()

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch, pad_id: int):
    maxlen = max(len(e.input_ids) for e, _, _ in batch)
    maxm = max(len(e.marker_pos) for e, _, _ in batch)
    B = len(batch)
    input_ids = torch.full((B, maxlen), pad_id, dtype=torch.long)
    attn = torch.zeros((B, maxlen), dtype=torch.long)
    marker = torch.zeros((B, maxm), dtype=torch.long)
    marker_mask = torch.zeros((B, maxm), dtype=torch.bool)
    labels = torch.full((B, maxm), -100, dtype=torch.long)
    meta = []
    for b, (e, lab, di) in enumerate(batch):
        n, m = len(e.input_ids), len(e.marker_pos)
        input_ids[b, :n] = torch.tensor(e.input_ids)
        attn[b, :n] = torch.tensor(e.attention_mask)
        marker[b, :m] = torch.tensor(e.marker_pos)
        marker_mask[b, :m] = True
        labels[b, :m] = torch.tensor(lab)
        meta.append((di, e.sent_idx))
    return input_ids, attn, marker, marker_mask, labels, meta


class SentenceTagger(nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        h = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(h, 2)

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask):
        out = self.encoder(input_ids=input_ids,
                           attention_mask=attention_mask).last_hidden_state
        # Gather the marker position for every sentence slot.
        idx = marker_pos.unsqueeze(-1).expand(-1, -1, out.size(-1))
        sent_repr = out.gather(1, idx)
        return self.head(self.dropout(sent_repr))


def _device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TransformerDetector:
    """Fine-tunes XLM-R for per-sentence human/AI tagging."""

    def __init__(self, model_name="xlm-roberta-base", max_length=512,
                 batch_size=4, grad_accum=4, lr=2e-5, epochs=4,
                 warmup_ratio=0.1, seed=42, fp16=True, class_weight=None,
                 max_grad_norm=1.0, verbose=True):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.grad_accum = grad_accum
        self.lr = lr
        self.epochs = epochs
        self.warmup_ratio = warmup_ratio
        self.seed = seed
        self.fp16 = fp16
        self.class_weight = class_weight
        self.max_grad_norm = max_grad_norm
        self.verbose = verbose
        self.tokenizer = None
        self.model = None
        self.device = _device()
        self.best_state = None
        self.best_score = -1.0
        self.best_epoch = -1

    @property
    def name(self) -> str:
        return f"{self.model_name} (ep={self.epochs}, lr={self.lr})"

    def _loader(self, docs, shuffle: bool):
        ds = SentenceTaggingDataset(docs, self.tokenizer, self.max_length)
        pad = self.tokenizer.pad_token_id
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                          collate_fn=lambda b: collate(b, pad))

    def fit(self, docs, dev_docs=None, eval_fn=None):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = SentenceTagger(self.model_name).to(self.device)

        loader = self._loader(docs, shuffle=True)
        steps = max(1, len(loader) // self.grad_accum) * self.epochs
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr,
                                weight_decay=0.01)
        sched = get_linear_schedule_with_warmup(
            opt, int(steps * self.warmup_ratio), steps)
        scaler = torch.amp.GradScaler("cuda", enabled=self.fp16 and
                                      self.device.type == "cuda")
        w = None
        if self.class_weight is not None:
            w = torch.tensor(self.class_weight, dtype=torch.float,
                             device=self.device)
        lossf = nn.CrossEntropyLoss(weight=w, ignore_index=-100)

        if self.verbose:
            print(f"  device={self.device} chunks={len(loader.dataset)} "
                  f"steps={steps}")
        self.model.train()
        for ep in range(self.epochs):
            tot, nb = 0.0, 0
            opt.zero_grad(set_to_none=True)
            for i, (ids, attn, mk, mkm, lab, _) in enumerate(loader):
                ids, attn = ids.to(self.device), attn.to(self.device)
                mk, lab = mk.to(self.device), lab.to(self.device)
                with torch.amp.autocast("cuda", enabled=self.fp16 and
                                        self.device.type == "cuda"):
                    logits = self.model(ids, attn, mk, mkm)
                    loss = lossf(logits.reshape(-1, 2), lab.reshape(-1))
                scaler.scale(loss / self.grad_accum).backward()
                tot += loss.item()
                nb += 1
                if (i + 1) % self.grad_accum == 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.max_grad_norm)
                    scaler.step(opt)
                    scaler.update()
                    sched.step()
                    opt.zero_grad(set_to_none=True)
            msg = f"  epoch {ep + 1}/{self.epochs} loss={tot / max(1, nb):.4f}"
            if dev_docs is not None and eval_fn is not None:
                self.model.eval()
                score = eval_fn(self)
                msg += f"  dev F1(AI)={score:.4f}"
                # Keep the best epoch. With only a few hundred training
                # documents the last epoch is often not the best one, and
                # XLM-R sits in a majority-class collapse for several epochs
                # before it starts separating the classes.
                if score > self.best_score:
                    self.best_score = score
                    self.best_epoch = ep + 1
                    self.best_state = {k: v.detach().cpu().clone()
                                       for k, v in self.model.state_dict().items()}
                    msg += "  *"
                self.model.train()
            if self.verbose:
                print(msg, flush=True)

        if self.best_state is not None:
            if self.verbose:
                print(f"  restoring best epoch {self.best_epoch} "
                      f"(dev F1(AI)={self.best_score:.4f})")
            self.model.load_state_dict(self.best_state)
        self.model.eval()
        return self

    @torch.no_grad()
    def predict(self, docs):
        """Flat per-sentence predictions, in the same order as sentence_view."""
        return self._infer(docs)[0]

    @torch.no_grad()
    def predict_proba(self, docs):
        return self._infer(docs)[1]

    @torch.no_grad()
    def _infer(self, docs):
        self.model.eval()
        loader = self._loader(docs, shuffle=False)
        # Chunking can split a document, so results are reassembled by
        # (doc index, sentence index) rather than by arrival order.
        probs: dict[tuple[int, int], float] = {}
        for ids, attn, mk, mkm, lab, meta in loader:
            ids, attn, mk = ids.to(self.device), attn.to(self.device), mk.to(self.device)
            with torch.amp.autocast("cuda", enabled=self.fp16 and
                                    self.device.type == "cuda"):
                logits = self.model(ids, attn, mk, mkm)
            p = torch.softmax(logits.float(), dim=-1)[:, :, 1].cpu().numpy()
            for b, (di, sidx) in enumerate(meta):
                for j, si in enumerate(sidx):
                    probs[(di, si)] = float(p[b, j])
        flat_p, flat_y = [], []
        for di, d in enumerate(docs):
            for si in range(d.n):
                pr = probs.get((di, si), 0.0)
                flat_p.append(pr)
                flat_y.append(int(pr >= 0.5))
        return flat_y, np.array(flat_p)

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "model.pt")
        self.tokenizer.save_pretrained(path)
