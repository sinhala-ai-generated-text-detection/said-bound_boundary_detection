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
                 marker: str | None = None, feat_map=None, feat_dim: int = 0):
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Optional per-sentence numeric features (e.g. LM likelihood stats),
        # keyed by record_id and aligned to sentence index.
        self.feat_map = feat_map or {}
        self.feat_dim = feat_dim
        # The marker must be a real token in the vocabulary. XLM-R's <s> is a
        # natural choice: it already means "segment start" to the model.
        self.marker = marker or tokenizer.cls_token
        self.marker_id = tokenizer.convert_tokens_to_ids(self.marker)
        self.items: list[tuple[DocEncoding, list[int], int, list]] = []
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
            doc_feats = self.feat_map.get(getattr(d, "record_id", None))

            def flush():
                if not chunk_markers:
                    return
                ids = [tok.bos_token_id] + chunk_ids + [tok.eos_token_id]
                # markers were recorded relative to chunk_ids; shift by the bos
                mk = [m + 1 for m in chunk_markers]
                enc = DocEncoding(ids, [1] * len(ids), mk, list(chunk_sidx))
                if self.feat_dim:
                    feats = [list(doc_feats[i]) if doc_feats and i < len(doc_feats)
                             else [0.0] * self.feat_dim for i in chunk_sidx]
                else:
                    feats = [[] for _ in chunk_sidx]
                self.items.append((enc, list(chunk_labels), di, feats))

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
    maxlen = max(len(e.input_ids) for e, _, _, _ in batch)
    maxm = max(len(e.marker_pos) for e, _, _, _ in batch)
    fdim = len(batch[0][3][0]) if batch[0][3] and batch[0][3][0] else 0
    B = len(batch)
    input_ids = torch.full((B, maxlen), pad_id, dtype=torch.long)
    attn = torch.zeros((B, maxlen), dtype=torch.long)
    marker = torch.zeros((B, maxm), dtype=torch.long)
    marker_mask = torch.zeros((B, maxm), dtype=torch.bool)
    labels = torch.full((B, maxm), -100, dtype=torch.long)
    extra = torch.zeros((B, maxm, fdim), dtype=torch.float) if fdim else None
    meta = []
    for b, (e, lab, di, feats) in enumerate(batch):
        n, m = len(e.input_ids), len(e.marker_pos)
        input_ids[b, :n] = torch.tensor(e.input_ids)
        attn[b, :n] = torch.tensor(e.attention_mask)
        marker[b, :m] = torch.tensor(e.marker_pos)
        marker_mask[b, :m] = True
        labels[b, :m] = torch.tensor(lab)
        if fdim:
            extra[b, :m] = torch.tensor(feats, dtype=torch.float)
        meta.append((di, e.sent_idx))
    return input_ids, attn, marker, marker_mask, labels, meta, extra


class TwinPairDataset(TorchDataset):
    """Each item is a mixed document together with its all-human twin.

    The two are encoded separately (a twin sentence can tokenize to a different
    length than the AI sentence it stands in for, so long documents may chunk
    differently) and matched up again per sentence in `collate_pairs`.
    """

    def __init__(self, docs, twins, weights, tokenizer, max_length: int = 512):
        def by_doc(ds, n):
            out = [[] for _ in range(n)]
            for item in ds.items:
                out[item[2]].append(item)
            return out
        self.mixed = by_doc(SentenceTaggingDataset(docs, tokenizer, max_length),
                            len(docs))
        self.twin = by_doc(SentenceTaggingDataset(twins, tokenizer, max_length),
                           len(twins))
        self.labels = [list(d.labels) for d in docs]
        self.weights = list(weights)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i):
        return self.mixed[i], self.twin[i], self.labels[i], self.weights[i]


def collate_pairs(batch, pad_id: int):
    """Stack every chunk of every pair into one encoder batch.

    Besides the usual tensors, returns for each sentence of each pair the flat
    index of its marker in the mixed document (`gm`) and in the twin (`gt`),
    so the loss can compare the two predictions for the same position.
    """
    chunks, owner = [], []
    for b, (cm, ct, _, _) in enumerate(batch):
        chunks += cm
        owner += [(b, 0)] * len(cm)
        chunks += ct
        owner += [(b, 1)] * len(ct)
    ids, attn, marker, marker_mask, _, meta, _ = collate(chunks, pad_id)
    width = marker.size(1)
    flat = {}
    for r, ((b, side), (_, sidx)) in enumerate(zip(owner, meta)):
        for j, si in enumerate(sidx):
            flat[(b, side, si)] = r * width + j
    gm, gt, y, tw = [], [], [], []
    for b, (_, _, labels, w) in enumerate(batch):
        for si, lab in enumerate(labels):
            gm.append(flat[(b, 0, si)])
            gt.append(flat[(b, 1, si)])
            y.append(lab)
            tw.append(w)
    return (ids, attn, marker, marker_mask, torch.tensor(gm),
            torch.tensor(gt), torch.tensor(y), torch.tensor(tw))


def twin_loss(logits, gm, gt, y, tw, class_weight=None, margin: float = 2.0):
    """Tagging loss on both documents, plus the two paired terms.

    * ``ce``: cross-entropy over the mixed sentences and the twin's sentences
      (all human), each twin down-weighted by how many records share its
      window. With no twins this is exactly the baseline objective.
    * ``margin``: at every replaced position the AI sentence must out-score
      the human sentence it replaced by ``margin`` in logit space. Position and
      topic are identical across the pair, so neither can satisfy this term;
      only authorship can.
    * ``consistency``: symmetric KL between the predictions for each untouched
      human sentence in the mixed document and in the twin. Its authorship does
      not change, so its prediction should not depend on whether machine text
      appears elsewhere. This is what targets the "every document has a
      boundary" prior.
    """
    flat = logits.reshape(-1, 2).float()
    lm, lt = flat[gm], flat[gt]
    tgt = torch.cat([y, torch.zeros_like(y)])
    w = torch.cat([torch.ones_like(tw), tw])
    if class_weight is not None:
        w = w * class_weight[tgt]
    ce_each = nn.functional.cross_entropy(torch.cat([lm, lt]), tgt,
                                          reduction="none")
    ce = (ce_each * w).sum() / w.sum()

    ai = y == 1
    zero = flat.new_zeros(())
    diff = (lm[:, 1] - lm[:, 0]) - (lt[:, 1] - lt[:, 0])
    mrg = torch.relu(margin - diff[ai]).mean() if ai.any() else zero
    if (~ai).any():
        pm = torch.log_softmax(lm[~ai], -1)
        pt = torch.log_softmax(lt[~ai], -1)
        cons = 0.5 * ((pm.exp() - pt.exp()) * (pm - pt)).sum(-1).mean()
    else:
        cons = zero
    return ce, mrg, cons


class SentenceTagger(nn.Module):
    """Sentence labels, plus an optional head that scores adjacent pairs.

    The pair head exists because the task is scored on *boundaries*, not on
    sentences, and a per-sentence head only reaches boundaries indirectly: it
    has to get two neighbours right, independently, for one boundary to appear
    in the right place. The pair head is given the two neighbouring
    representations together, with their difference and product, and asked the
    question the metric actually asks - did the author change here.
    """

    def __init__(self, model_name: str, dropout: float = 0.1,
                 use_pair_head: bool = True, extra_dim: int = 0):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        h = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.extra_dim = extra_dim
        # Likelihood features are concatenated to the contextual sentence
        # representation rather than used alone: they say how machine-like a
        # sentence looks in isolation, which is complementary to what the
        # encoder sees from its neighbours.
        self.head = nn.Linear(h + extra_dim, 2)
        self.use_pair_head = use_pair_head
        if use_pair_head:
            # [left ; right ; |left-right| ; left*right] - the difference and
            # product terms are what make discontinuity directly expressible.
            self.pair_head = nn.Sequential(
                nn.Linear(4 * h, h), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(h, 1))

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask,
                extra=None):
        out = self.encoder(input_ids=input_ids,
                           attention_mask=attention_mask).last_hidden_state
        # Gather the marker position for every sentence slot.
        idx = marker_pos.unsqueeze(-1).expand(-1, -1, out.size(-1))
        sent_repr = self.dropout(out.gather(1, idx))
        head_in = sent_repr
        if self.extra_dim and extra is not None:
            head_in = torch.cat([sent_repr, extra.to(sent_repr.dtype)], dim=-1)
        logits = self.head(head_in)
        if not self.use_pair_head:
            return logits, None
        left, right = sent_repr[:, :-1, :], sent_repr[:, 1:, :]
        pair_in = torch.cat([left, right, (left - right).abs(), left * right],
                            dim=-1)
        return logits, self.pair_head(pair_in).squeeze(-1)


def viterbi_decode(sent_logp, pair_logit, boundary_bias: float = 0.0):
    """Best label sequence under per-sentence scores and pair-wise transitions.

    This is a linear-chain CRF decode with input-dependent transitions: the
    cost of changing author between i-1 and i comes from the pair head rather
    than from a single global constant. Unlike the run-length smoother tried
    earlier, it never forbids a one-sentence span - it only has to pay for two
    changes instead of one, which is exactly the right prior for data whose
    multi-span construction uses spans of one to two sentences.

    `boundary_bias` shifts the change/stay trade-off and is tuned on dev.
    """
    n = sent_logp.shape[0]
    if n == 0:
        return []
    if n == 1:
        return [int(sent_logp[0, 1] > sent_logp[0, 0])]

    # log-sigmoid of the pair logit = log P(change); its complement = log P(stay)
    z = pair_logit + boundary_bias
    log_change = -np.logaddexp(0.0, -z)      # log sigmoid(z)
    log_stay = -np.logaddexp(0.0, z)         # log (1 - sigmoid(z))

    score = sent_logp[0].copy()              # shape (2,)
    back = np.zeros((n, 2), dtype=np.int64)
    for i in range(1, n):
        trans = np.array([[log_stay[i - 1], log_change[i - 1]],
                          [log_change[i - 1], log_stay[i - 1]]])
        total = score[:, None] + trans       # [prev, cur]
        back[i] = total.argmax(axis=0)
        score = total.max(axis=0) + sent_logp[i]

    path = [int(score.argmax())]
    for i in range(n - 1, 0, -1):
        path.append(int(back[i][path[-1]]))
    return path[::-1]


def _device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TransformerDetector:
    """Fine-tunes XLM-R for per-sentence human/AI tagging."""

    def __init__(self, model_name="xlm-roberta-base", max_length=512,
                 batch_size=4, grad_accum=4, lr=2e-5, epochs=4,
                 warmup_ratio=0.1, seed=42, fp16=True, class_weight=None,
                 max_grad_norm=1.0, verbose=True, use_pair_head=True,
                 pair_loss_weight=1.0, pair_pos_weight=2.0, feat_map=None,
                 margin_weight=1.0, consistency_weight=1.0, twin_margin=2.0,
                 twin_warmup_epochs=0.0):
        self.model_name = model_name
        # Counterfactual twin training (see twin_loss). Only active when fit()
        # is given twins; the weights at 0 give the "twins as plain extra
        # negatives" ablation.
        self.margin_weight = margin_weight
        self.consistency_weight = consistency_weight
        self.twin_margin = twin_margin
        # The paired terms are off for this many epochs, then ramp in linearly
        # over one more. Switched on from step 0 they trapped the model in a
        # constant output: an encoder that cannot yet tell the pair apart gets
        # no usable margin gradient, while the consistency term rewards
        # ignoring the input, which is exactly the collapsed solution.
        self.twin_warmup_epochs = twin_warmup_epochs
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
        self.use_pair_head = use_pair_head
        self.pair_loss_weight = pair_loss_weight
        # Boundaries are rare relative to non-boundaries; without an upweight
        # the pair head learns to say "no change" everywhere.
        self.pair_pos_weight = pair_pos_weight
        self.boundary_bias = 0.0
        # Per-sentence numeric features (LM likelihood stats), standardised
        # with training-set statistics only.
        self.feat_map = feat_map or {}
        self.feat_dim = 0
        if self.feat_map:
            self.feat_dim = len(next(iter(self.feat_map.values()))[0])
        self._mu = None
        self._sd = None
        # The runner's probe reports exact-boundary F1 regardless of whether
        # the pair head is on, so the label says so. A log that claims one
        # metric while printing another is how a regression goes unnoticed.
        self.dev_metric_name = "boundF1"
        self.tokenizer = None
        self.model = None
        self.device = _device()
        self.best_state = None
        self.best_score = -1.0
        self.best_epoch = -1

    @property
    def name(self) -> str:
        return f"{self.model_name} (ep={self.epochs}, lr={self.lr})"

    def _fit_scaler(self, docs) -> None:
        """Standardise features using TRAIN statistics only."""
        rows = [f for d in docs for f in self.feat_map.get(d.record_id, [])]
        if not rows:
            self._mu = np.zeros(self.feat_dim)
            self._sd = np.ones(self.feat_dim)
            return
        arr = np.asarray(rows, dtype=float)
        self._mu = arr.mean(0)
        sd = arr.std(0)
        sd[sd < 1e-6] = 1.0
        self._sd = sd

    def _scaled_map(self, docs):
        if not self.feat_dim:
            return None
        out = {}
        for d in docs:
            f = self.feat_map.get(d.record_id)
            if f:
                out[d.record_id] = ((np.asarray(f, dtype=float) - self._mu)
                                    / self._sd).tolist()
        return out

    def _loader(self, docs, shuffle: bool):
        ds = SentenceTaggingDataset(
            docs, self.tokenizer, self.max_length,
            feat_map=self._scaled_map(docs), feat_dim=self.feat_dim)
        pad = self.tokenizer.pad_token_id
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                          collate_fn=lambda b: collate(b, pad))

    def _twin_loader(self, docs, twins, weights):
        # Half as many pairs per batch as documents in the baseline, so the
        # encoder sees the same number of documents per step and the same
        # memory budget holds.
        ds = TwinPairDataset(docs, twins, weights, self.tokenizer,
                             self.max_length)
        pad = self.tokenizer.pad_token_id
        return DataLoader(ds, batch_size=max(1, self.batch_size // 2),
                          shuffle=True,
                          collate_fn=lambda b: collate_pairs(b, pad))

    def load(self, path):
        """Restore a saved tagger for evaluation without retraining."""
        path = Path(path)
        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        self.model = SentenceTagger(self.model_name,
                                    use_pair_head=self.use_pair_head,
                                    extra_dim=self.feat_dim)
        state = torch.load(path / "model.pt", map_location="cpu")
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        return self

    def fit(self, docs, dev_docs=None, eval_fn=None, twins=None,
            twin_weights=None, init_from=None):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.feat_dim:
            self._fit_scaler(docs)
        paired = twins is not None
        if paired and (self.feat_dim or self.use_pair_head):
            raise ValueError("twin training supports the sentence head only, "
                             "without likelihood features")
        self.model = SentenceTagger(
            self.model_name, use_pair_head=self.use_pair_head,
            extra_dim=self.feat_dim).to(self.device)
        if init_from is not None:
            # Continue from a trained tagger rather than the pretrained encoder,
            # e.g. to add twin training without re-learning the task.
            state = torch.load(Path(init_from) / "model.pt", map_location="cpu")
            self.model.load_state_dict(state)

        if paired:
            loader = self._twin_loader(docs, twins,
                                       twin_weights or [1.0] * len(docs))
        else:
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
        pair_bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(self.pair_pos_weight, device=self.device))

        if self.verbose:
            unit = "pairs" if paired else "chunks"
            print(f"  device={self.device} {unit}={len(loader.dataset)} "
                  f"steps={steps}")
        self.model.train()
        for ep in range(self.epochs):
            tot, nb = 0.0, 0
            parts = np.zeros(3)
            opt.zero_grad(set_to_none=True)
            for i, batch in enumerate(loader):
                if paired:
                    ids, attn, mk, mkm, gm, gt, y, tw = (
                        t.to(self.device) for t in batch)
                    with torch.amp.autocast("cuda", enabled=self.fp16 and
                                            self.device.type == "cuda"):
                        logits, _ = self.model(ids, attn, mk, mkm)
                    ce, mrg, cons = twin_loss(logits, gm, gt, y, tw, w,
                                              self.twin_margin)
                    ramp = ((ep + i / len(loader)) - self.twin_warmup_epochs
                            if self.twin_warmup_epochs else 1.0)
                    ramp = min(1.0, max(0.0, ramp))
                    loss = ce + ramp * (self.margin_weight * mrg
                                        + self.consistency_weight * cons)
                    parts += [ce.item(), mrg.item(), cons.item()]
                else:
                    ids, attn, mk, mkm, lab, _, ex = batch
                    ids, attn = ids.to(self.device), attn.to(self.device)
                    mk, lab = mk.to(self.device), lab.to(self.device)
                    ex = ex.to(self.device) if ex is not None else None
                    with torch.amp.autocast("cuda", enabled=self.fp16 and
                                            self.device.type == "cuda"):
                        logits, pair_logit = self.model(ids, attn, mk, mkm, ex)
                        loss = lossf(logits.reshape(-1, 2), lab.reshape(-1))
                        if pair_logit is not None and self.pair_loss_weight > 0:
                            # Boundary target for pair (i-1, i): did the label
                            # change. Masked to pairs where both are real.
                            left, right = lab[:, :-1], lab[:, 1:]
                            valid = (left >= 0) & (right >= 0)
                            if valid.any():
                                tgt = (left != right).float()
                                bl = pair_bce(pair_logit[valid], tgt[valid])
                                loss = loss + self.pair_loss_weight * bl
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
            if paired:
                ce_, mg_, cs_ = parts / max(1, nb)
                msg += (f" (ce={ce_:.4f} margin={mg_:.4f} cons={cs_:.4f} "
                        f"ramp={ramp:.2f})")
            if dev_docs is not None and eval_fn is not None:
                self.model.eval()
                score = eval_fn(self)
                msg += f"  dev {self.dev_metric_name}={score:.4f}"
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
    def _raw(self, docs):
        """Per-document sentence log-probs and pair logits, in doc order."""
        self.model.eval()
        loader = self._loader(docs, shuffle=False)
        # Chunking can split a document, so results are reassembled by
        # (doc index, sentence index) rather than by arrival order.
        logp: dict[tuple[int, int], np.ndarray] = {}
        pairs: dict[tuple[int, int], float] = {}
        for ids, attn, mk, mkm, lab, meta, ex in loader:
            ids, attn, mk = ids.to(self.device), attn.to(self.device), mk.to(self.device)
            ex = ex.to(self.device) if ex is not None else None
            with torch.amp.autocast("cuda", enabled=self.fp16 and
                                    self.device.type == "cuda"):
                logits, pair_logit = self.model(ids, attn, mk, mkm, ex)
            lp = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()
            pl = (pair_logit.float().cpu().numpy()
                  if pair_logit is not None else None)
            for b, (di, sidx) in enumerate(meta):
                for j, si in enumerate(sidx):
                    logp[(di, si)] = lp[b, j]
                if pl is not None:
                    # Pair j joins sentence sidx[j] and sidx[j+1]; both are in
                    # this chunk, so chunk adjacency is document adjacency.
                    for j in range(len(sidx) - 1):
                        pairs[(di, sidx[j + 1])] = float(pl[b, j])

        out = []
        for di, d in enumerate(docs):
            L = np.stack([logp.get((di, si), np.array([0.0, -np.inf]))
                          for si in range(d.n)])
            # A pair spanning a chunk split has no score; 0.0 logit = P(change)
            # 0.5, i.e. defer to the sentence head there rather than guess.
            P = np.array([pairs.get((di, si), 0.0) for si in range(1, d.n)])
            out.append((L, P))
        return out

    @torch.no_grad()
    def _infer(self, docs):
        raw = self._raw(docs)
        flat_p, flat_y = [], []
        for (L, _P) in raw:
            pr = np.exp(L[:, 1])
            flat_p.extend(pr.tolist())
            flat_y.extend((pr >= 0.5).astype(int).tolist())
        return flat_y, np.array(flat_p)

    @torch.no_grad()
    def predict_viterbi(self, docs, boundary_bias: float | None = None):
        """Structured decode: sentence scores + learned pair transitions."""
        bias = self.boundary_bias if boundary_bias is None else boundary_bias
        flat: list[int] = []
        for (L, P) in self._raw(docs):
            flat.extend(viterbi_decode(L, P, bias))
        return flat

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "model.pt")
        self.tokenizer.save_pretrained(path)
