"""Loading the generated dataset into a form detectors can train on.

The task is per-sentence binary labelling: for each sentence in a document,
was it written by a human (0) or a model (1). Boundaries are derived from the
label sequence rather than read from a stored field, so every consumer agrees
on the convention: a boundary at index i means the label changed between
sentence i-1 and sentence i.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import load_config, read_jsonl  # noqa: E402


@dataclass
class Doc:
    """One generated document with its per-sentence gold labels."""
    record_id: str
    source_id: str
    split: str
    generator: str
    construction_type: str
    sentences: list[str]
    labels: list[int]

    @property
    def n(self) -> int:
        return len(self.sentences)

    @property
    def boundaries(self) -> list[int]:
        return boundaries_from_labels(self.labels)


@dataclass
class Dataset:
    docs: list[Doc] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.docs)

    def filter(self, *, split: str | None = None,
               generators: list[str] | None = None,
               roles: list[str] | None = None,
               role_map: dict[str, str] | None = None,
               construction_type: str | None = None) -> "Dataset":
        out = []
        for d in self.docs:
            if split is not None and d.split != split:
                continue
            if generators is not None and d.generator not in generators:
                continue
            if roles is not None:
                if role_map is None:
                    raise ValueError("roles filter needs role_map")
                if role_map.get(d.generator) not in roles:
                    continue
            if construction_type is not None and \
                    d.construction_type != construction_type:
                continue
            out.append(d)
        return Dataset(out)

    @property
    def n_sentences(self) -> int:
        return sum(d.n for d in self.docs)

    def sentence_view(self):
        """Flatten to (sentences, labels, doc_index, sentence_index)."""
        S, Y, D, I = [], [], [], []
        for di, d in enumerate(self.docs):
            for si, (s, y) in enumerate(zip(d.sentences, d.labels)):
                S.append(s)
                Y.append(int(y))
                D.append(di)
                I.append(si)
        return S, Y, D, I


def boundaries_from_labels(labels) -> list[int]:
    """Indices where the author changes. Index i = change between i-1 and i."""
    return [i for i in range(1, len(labels)) if labels[i] != labels[i - 1]]


def load_dataset(cfg: dict | None = None, path=None) -> Dataset:
    cfg = cfg or load_config()
    path = path or Path(cfg["paths"]["generated_dir"]) / "combined.jsonl"
    docs: list[Doc] = []
    for r in read_jsonl(path):
        sents = r["sentences"]
        labels = [int(x) for x in r["labels"]]
        if len(sents) != len(labels):
            print(f"  WARN skipping {r['record_id']}: "
                  f"{len(sents)} sentences vs {len(labels)} labels")
            continue
        docs.append(Doc(
            record_id=r["record_id"], source_id=r["source_id"],
            split=r["split"], generator=r["generator"],
            construction_type=r["construction_type"],
            sentences=sents, labels=labels))
    return Dataset(docs)


def role_map(cfg: dict) -> dict[str, str]:
    return {g: c["role"] for g, c in cfg["generators"].items()}


def describe(ds: Dataset, cfg: dict, label: str = "dataset") -> None:
    from collections import Counter
    rm = role_map(cfg)
    print(f"\n{label}: {len(ds)} docs, {ds.n_sentences} sentences")
    ai = sum(sum(d.labels) for d in ds.docs)
    print(f"  AI sentences: {ai}/{ds.n_sentences} "
          f"({ai / max(1, ds.n_sentences):.1%})")
    for name, ctr in (("split", Counter(d.split for d in ds.docs)),
                      ("generator", Counter(d.generator for d in ds.docs)),
                      ("type", Counter(d.construction_type for d in ds.docs))):
        if name == "generator":
            body = "  ".join(f"{k}[{rm.get(k, '?')}]={v}"
                             for k, v in sorted(ctr.items()))
        elif name == "type":
            body = "  ".join(f"{k.split('_')[0]}={v}"
                             for k, v in sorted(ctr.items()))
        else:
            body = "  ".join(f"{k}={v}" for k, v in sorted(ctr.items()))
        print(f"  {name:9s} " + body)
