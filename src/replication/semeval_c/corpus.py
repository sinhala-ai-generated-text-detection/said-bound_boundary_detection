"""SemEval-2024 Task 8 Subtask C, and the human-only documents added to it.

Subtask C documents are a human-written prefix followed by a machine
continuation. The official label is the index of the first machine *word*,
where words are ``text.split(' ')``: single spaces only, so a run of spaces
yields empty words and a newline stays inside its word. Everything here uses
that convention, because it is the one the gold labels were computed with
(``human_end_boundary == len(prefix.split(' '))`` holds for every record).

A document with no machine text has boundary ``n`` (one past the last word),
so "no boundary" is an ordinary index and MAE stays defined.

    python src/replication/semeval_c/corpus.py    # -> data/english/processed/
"""
from __future__ import annotations

import glob
import json
import pickle
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

RAW = Path("data/english")
OUT = RAW / "processed"
SEED = 42
NGRAM = 8
EVAL_SIZE = 1000          # documents per length-matched human eval set


# ------------------------------------------------------------- words ---
def words_of(text: str) -> list[str]:
    return text.split(" ")


def tokenless(word: str) -> bool:
    """Words no subword tokenizer gives a token: empty or whitespace only.

    They carry a gold label but cannot be predicted, so their label is
    inherited from a neighbour (see fill_labels).
    """
    return not word.strip()


def gold_labels(n: int, boundary: int) -> list[int]:
    b = min(max(boundary, 0), n)
    return [0] * b + [1] * (n - b)


def fill_labels(labels, has_token, rule: str) -> list[int]:
    """Give each tokenless word the label of its neighbour.

    ``prev`` copies the nearest preceding word with a token, ``next`` the
    nearest following one; words with no such neighbour take the other side.
    A document with no token at all is all human.
    """
    n = len(labels)
    idx = [i for i in range(n) if has_token[i]]
    if not idx:
        return [0] * n
    out = list(labels)
    if rule == "prev":
        cur = labels[idx[0]]
        for i in range(n):
            if has_token[i]:
                cur = labels[i]
            out[i] = cur
    elif rule == "next":
        cur = labels[idx[-1]]
        for i in range(n - 1, -1, -1):
            if has_token[i]:
                cur = labels[i]
            out[i] = cur
    else:
        raise ValueError(rule)
    return [int(x) for x in out]


def boundary_from_labels(labels) -> int:
    """Index of the first word labelled machine; len(labels) if none."""
    for i, y in enumerate(labels):
        if y:
            return i
    return len(labels)


def forced_error(docs, rule: str) -> dict:
    """MAE left by the tokenless-word rule with every other word labelled right."""
    errs = []
    for d in docs:
        w = words_of(d["text"])
        has = [not tokenless(x) for x in w]
        pred = boundary_from_labels(
            fill_labels(gold_labels(len(w), d["boundary"]), has, rule))
        errs.append(abs(pred - min(d["boundary"], len(w))))
    errs = np.asarray(errs)
    return {"mae": float(errs.mean()), "docs_with_error": int((errs > 0).sum()),
            "n": len(errs)}


# ------------------------------------------------------------ leakage ---
def ngrams(text: str, n: int = NGRAM) -> set:
    w = text.lower().split()
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


# ------------------------------------------------------ length matching ---
_SENT = re.compile(r"(?<=[.!?])(\s+)")


def truncate_to(text: str, target: int) -> str:
    """Cut at the sentence boundary whose word count is closest to target."""
    parts = _SENT.split(text)
    best, best_gap, acc = parts[0], None, ""
    for i in range(0, len(parts), 2):
        acc = "".join(parts[:i + 1])
        gap = abs(len(words_of(acc)) - target)
        if best_gap is None or gap < best_gap:
            best, best_gap = acc, gap
    return best.rstrip()


def length_match(targets, pool, rng: random.Random, tol: float = 0.10):
    """One pool document per target word count, without replacement.

    A document within ``tol`` of the target is taken whole; otherwise the
    shortest longer one is cut at a sentence boundary. Returns the chosen
    (item, text) pairs and how many targets needed a cut or fell short.
    """
    free = sorted(pool, key=lambda p: len(words_of(p[1])))
    lens = [len(words_of(t)) for _, t in free]
    used = [False] * len(free)
    order = list(range(len(targets)))
    rng.shuffle(order)
    out = [None] * len(targets)
    stats = Counter()
    for k in order:
        L = targets[k]
        near = [i for i in range(len(free)) if not used[i]
                and abs(lens[i] - L) <= tol * L]
        if near:
            i = rng.choice(near)
            text = free[i][1]
            stats["whole"] += 1
        else:
            longer = [i for i in range(len(free)) if not used[i]
                      and lens[i] > L]
            if longer:
                i = longer[0]
                text = truncate_to(free[i][1], L)
                stats["truncated"] += 1
            else:
                i = max((j for j in range(len(free)) if not used[j]),
                        key=lambda j: lens[j])
                text = free[i][1]
                stats["short"] += 1
        used[i] = True
        out[k] = (free[i][0], text)
    return out, dict(stats)


def pct(xs) -> dict:
    xs = np.asarray(xs)
    return {"n": int(len(xs)), "mean": float(xs.mean()),
            **{f"p{q}": float(np.percentile(xs, q)) for q in (5, 25, 50, 75, 95)},
            "total_words": int(xs.sum())}


# ------------------------------------------------------------- loading ---
def load_subtask_c():
    """Official Subtask C: PeerRead/ChatGPT train and dev, the full test set."""
    tr_dev, test = [], []
    for line in open(RAW / "semeval_c" / "subtaskC_train_dev.jsonl"):
        r = json.loads(line)
        if r["domain_model"] != "peerread_chatgpt":
            continue            # M4GT-Bench extra generators, not official
        tr_dev.append({"id": r["uuid"], "split": r["split"],
                       "text": r["mixed_review"],
                       "boundary": int(r["human_end_boundary"]),
                       "domain": "peerread", "generator": "chatgpt",
                       "human_original": r["full_human_review"]})
    for i, line in enumerate(open(RAW / "semeval_c" / "subtaskC_test.jsonl")):
        r = json.loads(line)
        dom, gen = r["domain_model"].split("_", 1)
        test.append({"id": f"test_{i}", "split": "test",
                     "text": r["mixed_text"],
                     "boundary": int(r["human_end_boundary"]),
                     "domain": dom, "generator": gen,
                     "human_original": r["human_text"]})
    return ([d for d in tr_dev if d["split"] == "train"],
            [d for d in tr_dev if d["split"] == "dev"], test)


def all_subtask_c_texts():
    """Every text in both files, every generator: the leakage reference."""
    for f, keys in (("subtaskC_train_dev.jsonl",
                     ("mixed_review", "full_human_review",
                      "truncated_human_review", "machine_review")),
                    ("subtaskC_test.jsonl",
                     ("mixed_text", "human_text", "truncated_human_text",
                      "machine_text"))):
        for line in open(RAW / "semeval_c" / f):
            r = json.loads(line)
            for k in keys:
                if r.get(k):
                    yield r[k]


def load_peerread():
    out = []
    for f in sorted(glob.glob(str(RAW / "PeerRead/data/*/*/reviews/*.json"))):
        d = json.load(open(f))
        for r in d.get("reviews", []):
            t = (r.get("comments") or "").strip()
            if t:
                out.append((f.split("/")[3], t))
    return out


def load_asap():
    out = []
    for f in sorted(glob.glob(str(RAW / "asap/dataset/*/*_review/*.json"))):
        d = json.load(open(f))
        venue = d["id"].rsplit("_", 1)[0]
        for j, r in enumerate(d["reviews"]):
            t = (r.get("review") or "").strip()
            if t:
                out.append({"venue": venue, "paper": d["id"],
                            "id": f"{d['id']}_r{j}", "text": t})
    return out


def load_outfox():
    out = []
    for sp in ("train", "valid", "test"):
        with open(RAW / f"OUTFOX/data/common/{sp}/{sp}_humans.pkl", "rb") as fh:
            for j, t in enumerate(pickle.load(fh)):
                out.append({"split": sp, "id": f"outfox_{sp}_{j}",
                            "text": t.strip()})
    return out


def human_doc(did, text, source, **extra):
    return {"id": did, "text": text, "boundary": len(words_of(text)),
            "domain": source, "generator": "human", **extra}


def write(name, docs):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")


def read(name):
    return [json.loads(l) for l in open(OUT / f"{name}.jsonl", encoding="utf-8")]


# ---------------------------------------------------------------- build ---
def main() -> None:
    rng = random.Random(SEED)
    train, dev, test = load_subtask_c()
    rep = {"subtask_c": {
        "train": len(train), "dev": len(dev), "test": len(test),
        "test_by_domain_generator": Counter(
            f"{d['domain']}/{d['generator']}" for d in test),
        "boundary_at_end": {k: sum(d["boundary"] >= len(words_of(d["text"]))
                                   for d in v)
                            for k, v in (("train", train), ("dev", dev),
                                         ("test", test))},
        "fully_machine": {k: sum(d["boundary"] == 0 for d in v)
                          for k, v in (("train", train), ("dev", dev),
                                       ("test", test))},
    }}

    # Tokenless-word rule, chosen on train only.
    rules = {r: forced_error(train, r) for r in ("prev", "next")}
    rule = min(rules, key=lambda r: rules[r]["mae"])
    rep["tokenless_rule"] = {
        "chosen": rule, "train": rules,
        "forced_error": {k: forced_error(v, rule)
                         for k, v in (("train", train), ("dev", dev),
                                      ("test", test))},
        "docs_with_tokenless_words": {
            k: sum(any(tokenless(w) for w in words_of(d["text"])) for d in v)
            for k, v in (("train", train), ("dev", dev), ("test", test))}}
    print("tokenless rule", rule, rules)

    print("indexing Subtask C 8-grams ...", flush=True)
    ref = set()
    for t in all_subtask_c_texts():
        ref |= ngrams(t)
    clean = lambda t: not (ngrams(t) & ref)     # noqa: E731

    # PeerRead: recorded to show why it cannot be used.
    pr = load_peerread()
    pr_kept = [t for _, t in pr if clean(t)]
    rep["peerread"] = {"reviews": len(pr),
                       "by_venue": Counter(v for v, _ in pr),
                       "kept": len(pr_kept), "kept_distinct": len(set(pr_kept)),
                       "kept_median_words": float(np.median(
                           [len(words_of(t)) for t in pr_kept]))
                       if pr_kept else 0}

    # ASAP-Review, excluding ICLR 2017 (PeerRead's ICLR year).
    asap = load_asap()
    by_venue = Counter(a["venue"] for a in asap)
    iclr17_clean = sum(clean(a["text"]) for a in asap
                       if a["venue"] == "ICLR_2017")
    asap = [a for a in asap if a["venue"] != "ICLR_2017"]
    seen_txt, asap_ok = set(), []
    for a in asap:
        if a["text"] in seen_txt or not clean(a["text"]):
            continue
        seen_txt.add(a["text"])
        asap_ok.append(a)
    rep["asap"] = {"before": by_venue,
                   "after": Counter(a["venue"] for a in asap_ok),
                   "iclr_2017_passing_filter": iclr17_clean,
                   "kept": len(asap_ok)}
    papers = sorted({a["paper"] for a in asap_ok})
    rng.shuffle(papers)
    eval_papers = set(papers[:len(papers) // 2])
    asap_train = [a for a in asap_ok if a["paper"] not in eval_papers]
    asap_eval = [a for a in asap_ok if a["paper"] in eval_papers]

    # OUTFOX: human essays that survive the filter; evaluation only.
    ofx = load_outfox()
    ofx_ok = [o for o in ofx if clean(o["text"])]
    rep["outfox"] = {"essays": len(ofx), "by_split": Counter(o["split"] for o in ofx),
                     "kept": len(ofx_ok),
                     "kept_by_split": Counter(o["split"] for o in ofx_ok)}

    # Condition C: each distinct human original behind the training
    # documents, once, minus any that also appears in test.
    test_orig = {d["human_original"] for d in test}
    k = Counter(d["human_original"] for d in train)
    twins = sorted(t for t in k if t not in test_orig)
    twin_docs = [human_doc(f"twin_{i}", t, "peerread") for i, t in
                 enumerate(twins)]
    rep["twins_train"] = {"distinct": len(k), "also_in_test": len(k) - len(twins),
                          "used": len(twins),
                          "training_docs_per_twin": Counter(k.values())}

    # Condition B: the same number of ASAP reviews, matched one to one to
    # the twins' word counts.
    tlen = [len(words_of(t)) for t in twins]
    picked, st = length_match(tlen, [(a, a["text"]) for a in asap_train], rng)
    asap_docs = [human_doc(a["id"], t, "asap", venue=a["venue"],
                           paper=a["paper"]) for a, t in picked]
    rep["human_train"] = {
        "twins": pct(tlen),
        "asap": pct([len(words_of(d["text"])) for d in asap_docs]),
        "asap_matching": st,
        "mixed_train": pct([len(words_of(d["text"])) for d in train]),
        "ratio_docs": len(twins) / len(train)}

    # Evaluation sets.
    def matched_eval(target_docs, pool, name):
        tl = [len(words_of(d["text"])) for d in target_docs]
        tgt = rng.sample(tl, min(EVAL_SIZE, len(tl)))
        got, st = length_match(tgt, pool, rng)
        rep.setdefault("eval", {})[name] = {
            "target": pct(tl), "matched": pct([len(words_of(t)) for _, t in got]),
            "matching": st}
        return got

    ev_asap = matched_eval([d for d in test if d["domain"] == "peerread"],
                           [(a, a["text"]) for a in asap_eval], "asap")
    ev_ofx = matched_eval([d for d in test if d["domain"] == "outfox"],
                          [(o, o["text"]) for o in ofx_ok], "outfox")
    assert not {a["paper"] for a, _ in ev_asap} & \
        {d["paper"] for d in asap_docs}, "an ASAP paper is in train and eval"

    train_orig = {d["human_original"] for d in train + dev}
    tw_test = {}
    for dom in ("peerread", "outfox"):
        s = sorted({d["human_original"] for d in test if d["domain"] == dom}
                   - train_orig)
        tw_test[dom] = [human_doc(f"testtwin_{dom}_{i}", t, dom)
                        for i, t in enumerate(s)]
    rep["eval"]["twins"] = {dom: pct([len(words_of(d["text"])) for d in v])
                            for dom, v in tw_test.items()}

    write("train", train)
    write("dev", dev)
    write("test", test)
    write("human_train_twins", twin_docs)
    write("human_train_asap", asap_docs)
    write("eval_asap", [human_doc(a["id"], t, "asap", venue=a["venue"],
                                  paper=a["paper"]) for a, t in ev_asap])
    write("eval_outfox", [human_doc(o["id"], t, "outfox") for o, t in ev_ofx])
    write("eval_twins_peerread", tw_test["peerread"])
    write("eval_twins_outfox", tw_test["outfox"])
    (OUT / "build_report.json").write_text(json.dumps(rep, indent=2, default=dict))
    print(json.dumps(rep, indent=2, default=dict))


if __name__ == "__main__":
    sys.exit(main())
