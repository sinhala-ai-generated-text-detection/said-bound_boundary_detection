"""Train and evaluate one Subtask C tagger under one training condition.

    control   the official Subtask C training set only
    human     plus unrelated human-only peer reviews (ASAP-Review), all human
    twins     plus the full human originals behind the training documents

The human and twin additions have the same number of documents, the same word
counts and the same weight (each distinct document once), so they differ only
in whether the human text is content-matched. The epoch is chosen on dev MAE
(first-word rule, threshold 0.5), which cannot see false alarms.

    python src/replication/semeval_c/run.py --condition control --seed 42 \\
        --tag deberta_control_s42
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpus import gold_labels, read, tokenless, words_of  # noqa: E402
from scoring import (deployment, human_metrics, mixed_metrics,  # noqa: E402
                     predict)

OUT = Path("results/replication/semeval_c")
PREDS = Path("data/english/preds")
EVAL_SETS = ("eval_asap", "eval_outfox", "eval_twins_peerread",
             "eval_twins_outfox")
ADD = {"control": None, "human": "human_train_asap",
       "twins": "human_train_twins"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=list(ADD), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--model", default="microsoft/deberta-v3-base")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke test: N documents per set")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--preds-dir", default=str(PREDS))
    a = ap.parse_args()

    import torch
    from tagger import WordTagger
    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  "
          f"{a.tag}", flush=True)

    lim = (lambda xs: xs[:a.limit]) if a.limit else (lambda xs: xs)  # noqa
    train, dev, test = (lim(read(s)) for s in ("train", "dev", "test"))
    if a.limit:     # keep both test domains in a smoke test
        full = read("test")
        test = full[:a.limit // 2] + full[-(a.limit // 2):]
    human = lim(read(ADD[a.condition])) if ADD[a.condition] else []
    evals = {s: lim(read(s)) for s in EVAL_SETS}

    # Class weights from the official training words only, so every
    # condition uses the same numbers.
    n_h = n_m = 0
    for d in train:
        w = words_of(d["text"])
        for x, y in zip(w, gold_labels(len(w), d["boundary"])):
            if not tokenless(x):
                n_h += y == 0
                n_m += y == 1
    cw = [(n_h + n_m) / (2 * n_h), (n_h + n_m) / (2 * n_m)]
    hw = sum(len(words_of(d["text"])) for d in human)
    mw = sum(len(words_of(d["text"])) for d in train)
    print(f"train: {len(train)} mixed ({mw} words) + {len(human)} human "
          f"({hw} words); class weights human={cw[0]:.3f} "
          f"machine={cw[1]:.3f}", flush=True)

    det = WordTagger(model_name=a.model, max_length=a.max_length,
                     batch_size=a.batch_size, lr=a.lr, epochs=a.epochs,
                     seed=a.seed, class_weight=cw)

    def dev_mae(m):
        pred = predict(m.predict_proba(dev), "first_word", 0.5)
        return mixed_metrics(dev, pred)["overall"]["mae"]

    t0 = time.time()
    det.fit(train + human, dev, dev_mae,
            log=lambda s: print(s, flush=True))
    train_min = (time.time() - t0) / 60

    t1 = time.time()
    P = {"dev": det.predict_proba(dev), "test": det.predict_proba(test)}
    for s, docs in evals.items():
        P[s] = det.predict_proba(docs)
    res = {}
    for dec in ("first_word", "change_point"):
        tp = predict(P["test"], dec)
        r = {"dev": mixed_metrics(dev, predict(P["dev"], dec))["overall"],
             "test": mixed_metrics(test, tp), "human": {}}
        hp = {}
        for s, docs in evals.items():
            hp[s] = predict(P[s], dec)
            r["human"][s] = human_metrics(
                docs, P[s], hp[s], 0.5 if dec == "first_word" else None)
        r["deployment"] = deployment(
            test, tp, evals["eval_asap"] + evals["eval_outfox"],
            hp["eval_asap"] + hp["eval_outfox"])
        res[dec] = r
    eval_min = (time.time() - t1) / 60

    payload = {
        "tag": a.tag, "condition": a.condition, "seed": a.seed,
        "model": a.model, "epochs": a.epochs, "lr": a.lr,
        "batch_size": a.batch_size, "max_length": a.max_length,
        "limit": a.limit, "class_weight": cw,
        "train": {"mixed_docs": len(train), "mixed_words": mw,
                  "human_docs": len(human), "human_words": hw},
        "best_epoch": det.best_epoch, "best_dev_mae": det.best_score,
        "history": det.history, "train_minutes": train_min,
        "eval_minutes": eval_min, **res,
    }
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.tag}.json").write_text(json.dumps(payload, indent=2))
    write_report(out / f"{a.tag}.md", payload)
    pdir = Path(a.preds_dir)
    pdir.mkdir(parents=True, exist_ok=True)
    with open(pdir / f"{a.tag}.pkl", "wb") as fh:
        pickle.dump({k: [x.astype(np.float16) for x in v]
                     for k, v in P.items()}, fh)
    fw = res["first_word"]
    print(f"\nTEST MAE={fw['test']['overall']['mae']:.2f} "
          f"exact={fw['test']['overall']['exact']:.3f}  "
          + "  ".join(f"{s}: FA={m['false_alarm']:.3f}"
                      for s, m in fw["human"].items()), flush=True)
    print(f"wrote -> {out / a.tag}.{{md,json}}  ({train_min:.1f} min train, "
          f"{eval_min:.1f} min eval)")


NAMES = {"eval_asap": "ASAP reviews", "eval_outfox": "OUTFOX essays",
         "eval_twins_peerread": "test twins, PeerRead",
         "eval_twins_outfox": "test twins, OUTFOX"}


def write_report(path: Path, p: dict) -> None:
    t = p["train"]
    L = [f"# Subtask C tagger: {p['tag']}", "",
         f"`{p['model']}`, condition **{p['condition']}**, seed {p['seed']}: "
         f"{t['mixed_docs']} mixed training documents ({t['mixed_words']:,} "
         f"words) plus {t['human_docs']} human-only documents "
         f"({t['human_words']:,} words). {p['epochs']} epochs, lr {p['lr']}, "
         f"batch {p['batch_size']}; best epoch {p['best_epoch']} on dev MAE "
         f"({p['best_dev_mae']:.2f}); {p['train_minutes']:.1f} min training.",
         "", "## Subtask C test", "",
         "| decoder | slice | docs | MAE | exact |", "|---|---|---|---|---|"]
    for dec in ("first_word", "change_point"):
        r = p[dec]["test"]
        L.append(f"| {dec} | overall | {r['overall']['n']} | "
                 f"{r['overall']['mae']:.2f} | {r['overall']['exact']:.3f} |")
        for kind in ("by_domain", "by_generator"):
            for k, m in r[kind].items():
                L.append(f"| {dec} | {k} | {m['n']} | {m['mae']:.2f} | "
                         f"{m['exact']:.3f} |")
    L += ["", "## Human-only documents", "",
          "*False alarm*: share of documents with a predicted boundary "
          "(any machine word, for the first-word rule). *Flagged*: mean share "
          "of words labelled machine.", "",
          "| decoder | set | docs | false alarm | flagged |",
          "|---|---|---|---|---|"]
    for dec in ("first_word", "change_point"):
        for s, m in p[dec]["human"].items():
            L.append(f"| {dec} | {NAMES[s]} | {m['n_docs']} | "
                     f"{m['false_alarm']:.3f} | {m['frac_flagged']:.3f} |")
    L += ["", "## Deployment view", "",
          "Mixed test documents plus the ASAP and OUTFOX human documents: "
          "is there machine text at all?", "",
          "| decoder | accuracy | balanced | mixed detected | human quiet | "
          "MAE, mixed |", "|---|---|---|---|---|---|"]
    for dec in ("first_word", "change_point"):
        m = p[dec]["deployment"]
        L.append(f"| {dec} | {m['presence_accuracy']:.3f} | "
                 f"{m['balanced_accuracy']:.3f} | {m['mixed_detected']:.3f} | "
                 f"{m['human_quiet']:.3f} | {m['mae_mixed']:.2f} |")
    L += ["", "## Training", "", "| epoch | loss | dev MAE | min |",
          "|---|---|---|---|"]
    for h in p["history"]:
        L.append(f"| {h['epoch']} | {h['loss']:.4f} | {h['dev_mae']:.2f} | "
                 f"{h['minutes']:.1f} |")
    L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
