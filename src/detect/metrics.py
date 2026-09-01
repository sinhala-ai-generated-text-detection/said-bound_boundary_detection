"""Evaluation for boundary detection.

Two levels, because they answer different questions:

*Sentence level* - did we label each sentence's author correctly. This is the
learning signal, but it flatters a detector on this dataset: the AI class is
~36% of sentences, so a degenerate "all human" predictor already scores 64%
accuracy. Always read F1 on the AI class, not accuracy.

*Boundary level* - did we find the actual points where authorship changes.
This is the task the dataset is named for and is much harder: a document with
one boundary gives one chance to be right. Reported with a tolerance, since
predicting a change one sentence early is materially better than missing it.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

from data import boundaries_from_labels


@dataclass
class SentenceMetrics:
    n: int
    accuracy: float
    precision_ai: float
    recall_ai: float
    f1_ai: float
    macro_f1: float
    support_ai: int

    def as_dict(self) -> dict:
        return asdict(self)


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def sentence_metrics(y_true, y_pred) -> SentenceMetrics:
    assert len(y_true) == len(y_pred), "length mismatch"
    n = len(y_true)
    if n == 0:
        return SentenceMetrics(0, 0, 0, 0, 0, 0, 0)
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    p1, r1, f1 = _prf(tp, fp, fn)
    p0, r0, f0 = _prf(tn, fn, fp)          # human class
    return SentenceMetrics(
        n=n, accuracy=(tp + tn) / n,
        precision_ai=p1, recall_ai=r1, f1_ai=f1,
        macro_f1=(f1 + f0) / 2, support_ai=tp + fn)


@dataclass
class BoundaryMetrics:
    n_docs: int
    gold: int
    pred: int
    precision: float
    recall: float
    f1: float
    exact_doc_match: float          # fraction of docs with the exact set
    tolerance: int

    def as_dict(self) -> dict:
        return asdict(self)


def _match_boundaries(gold: list[int], pred: list[int], tol: int) -> int:
    """Greedy one-to-one matching within +/- tol. Each gold matches once."""
    used: set[int] = set()
    hits = 0
    for g in gold:
        best, best_d = None, None
        for j, p in enumerate(pred):
            if j in used:
                continue
            d = abs(p - g)
            if d <= tol and (best_d is None or d < best_d):
                best, best_d = j, d
        if best is not None:
            used.add(best)
            hits += 1
    return hits


def boundary_metrics(gold_labels: list[list[int]], pred_labels: list[list[int]],
                     tolerance: int = 1) -> BoundaryMetrics:
    tp = n_gold = n_pred = 0
    exact = 0
    for g, p in zip(gold_labels, pred_labels):
        gb = boundaries_from_labels(g)
        pb = boundaries_from_labels(p)
        n_gold += len(gb)
        n_pred += len(pb)
        tp += _match_boundaries(gb, pb, tolerance)
        if gb == pb:
            exact += 1
    fp = n_pred - tp
    fn = n_gold - tp
    p, r, f = _prf(tp, max(0, fp), max(0, fn))
    return BoundaryMetrics(
        n_docs=len(gold_labels), gold=n_gold, pred=n_pred,
        precision=p, recall=r, f1=f,
        exact_doc_match=exact / len(gold_labels) if gold_labels else 0.0,
        tolerance=tolerance)


def regroup(docs, flat_pred, doc_index) -> list[list[int]]:
    """Turn flat per-sentence predictions back into per-document sequences."""
    out: list[list[int]] = [[] for _ in docs]
    for pred, di in zip(flat_pred, doc_index):
        out[di].append(int(pred))
    return out


def evaluate(docs, flat_true, flat_pred, doc_index,
             tolerance: int = 1) -> dict:
    """Sentence- and boundary-level metrics for one prediction set."""
    per_doc_pred = regroup(docs, flat_pred, doc_index)
    per_doc_gold = [list(d.labels) for d in docs]
    return {
        "sentence": sentence_metrics(flat_true, flat_pred).as_dict(),
        "boundary_exact": boundary_metrics(per_doc_gold, per_doc_pred, 0).as_dict(),
        "boundary_tol": boundary_metrics(per_doc_gold, per_doc_pred,
                                         tolerance).as_dict(),
    }


def fmt_row(name: str, m: dict) -> str:
    s, b0, bt = m["sentence"], m["boundary_exact"], m["boundary_tol"]
    return (f"| {name} | {s['accuracy']:.3f} | {s['f1_ai']:.3f} | "
            f"{s['precision_ai']:.3f} | {s['recall_ai']:.3f} | "
            f"{b0['f1']:.3f} | {bt['f1']:.3f} | {b0['exact_doc_match']:.3f} |")


HEADER = ("| model | sent acc | sent F1(AI) | P(AI) | R(AI) | "
          "bound F1 exact | bound F1 ±1 | doc exact |")
DIVIDER = "|---|---|---|---|---|---|---|---|"
