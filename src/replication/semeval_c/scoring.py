"""Decoding word probabilities into a boundary, and the scores reported.

Two decoders:

* **first-word** (the task's native rule): words at or above a probability
  threshold are machine; the boundary is the first machine word, or ``n`` if
  there is none. One flagged word anywhere is a false alarm on a human text.
* **change point**: the single boundary ``b`` (``n`` = none) that maximises
  ``sum_{i<b} log P(human_i) + sum_{i>=b} log P(machine_i)``, the exact
  maximum a posteriori boundary under the task's human-then-machine structure.
  It has no threshold, and it must weigh a whole run of evidence before it
  places a boundary.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from corpus import boundary_from_labels, fill_labels, words_of

RULE = "prev"             # tokenless-word rule, fixed on train by corpus.py


def first_word(prob, thr: float, rule: str = RULE) -> int:
    has = ~np.isnan(prob)
    lab = np.where(has, np.nan_to_num(prob) >= thr, False).astype(int)
    return boundary_from_labels(fill_labels(lab.tolist(), has.tolist(), rule))


def change_point(prob) -> int:
    idx = np.flatnonzero(~np.isnan(prob))
    n = len(prob)
    if len(idx) == 0:
        return n
    p = np.clip(prob[idx].astype(np.float64), 1e-6, 1 - 1e-6)
    s0 = np.concatenate([[0.0], np.cumsum(np.log1p(-p))])
    s1 = np.concatenate([[0.0], np.cumsum(np.log(p))])
    score = s0 + (s1[-1] - s1)          # k human words, then machine
    k = int(np.argmax(score))
    return int(idx[k]) if k < len(idx) else n


def predict(probs, decoder: str, thr: float = 0.5) -> list[int]:
    if decoder == "first_word":
        return [first_word(p, thr) for p in probs]
    return [change_point(p) for p in probs]


def _mae(docs, pred):
    g = np.array([min(d["boundary"], len(words_of(d["text"]))) for d in docs])
    e = np.abs(np.asarray(pred) - g)
    return {"n": len(docs), "mae": float(e.mean()),
            "exact": float((e == 0).mean())}


def mixed_metrics(docs, pred) -> dict:
    out = {"overall": _mae(docs, pred), "by_domain": {}, "by_generator": {}}
    groups = defaultdict(list)
    for i, d in enumerate(docs):
        groups[("by_domain", d["domain"])].append(i)
        groups[("by_generator", f"{d['domain']}/{d['generator']}")].append(i)
    for (kind, key), ix in sorted(groups.items()):
        out[kind][key] = _mae([docs[i] for i in ix], [pred[i] for i in ix])
    return out


def human_metrics(docs, probs, pred, thr: float | None = None) -> dict:
    """False alarms on documents with no machine text.

    ``frac_flagged`` is the share of words labelled machine: from the
    threshold when one is given, else every word from the predicted
    boundary on.
    """
    n = [len(words_of(d["text"])) for d in docs]
    alarm = [b < k for b, k in zip(pred, n)]
    if thr is not None:
        frac = []
        for p in probs:
            has = ~np.isnan(p)
            lab = np.where(has, np.nan_to_num(p) >= thr, False).astype(int)
            frac.append(np.mean(fill_labels(lab.tolist(), has.tolist(), RULE)))
    else:
        frac = [(k - b) / k for b, k in zip(pred, n)]
    return {"n_docs": len(docs), "false_alarm": float(np.mean(alarm)),
            "frac_flagged": float(np.mean(frac))}


def deployment(mixed, mixed_pred, human, human_pred) -> dict:
    """Mixed and human test documents together: is there machine text?"""
    gm = [d["boundary"] < len(words_of(d["text"])) for d in mixed]
    pm = [b < len(words_of(d["text"])) for d, b in zip(mixed, mixed_pred)]
    ph = [b < len(words_of(d["text"])) for d, b in zip(human, human_pred)]
    hit = [a == b for a, b in zip(gm, pm)] + [not x for x in ph]
    pos = [a == b for a, b, g in zip(gm, pm, gm) if g]
    return {"n_mixed": len(mixed), "n_human": len(human),
            "presence_accuracy": float(np.mean(hit)),
            "mixed_detected": float(np.mean(pos)),
            "human_quiet": float(1 - np.mean(ph)),
            "balanced_accuracy": float((np.mean(pos) + 1 - np.mean(ph)) / 2),
            "mae_mixed": _mae(mixed, mixed_pred)["mae"]}
