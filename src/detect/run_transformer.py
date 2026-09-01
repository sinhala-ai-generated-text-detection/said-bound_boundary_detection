"""Fine-tune and evaluate the XLM-R sentence tagger.

Follows exactly the protocol used for the linear models so the numbers are
comparable: train on `train`, select on `dev` restricted to seen generators,
report on `test` split into seen vs held-out.

The decision threshold is tuned on dev-seen as well. Leaving it at 0.5 is an
arbitrary choice on an imbalanced task, and tuning it on the held-out portion
of dev would leak the generalisation signal being measured.

    python src/detect/run_transformer.py --epochs 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import load_dataset, describe, role_map  # noqa: E402
from metrics import evaluate, sentence_metrics  # noqa: E402
from utils import force_utf8_stdout, load_config  # noqa: E402


def predict_at(model, ds, threshold: float):
    proba = model.predict_proba(ds.docs)
    return (proba >= threshold).astype(int).tolist(), proba


def eval_ds(model, ds, threshold: float) -> dict:
    if len(ds) == 0:
        return {}
    S, Y, D, I = ds.sentence_view()
    pred, _ = predict_at(model, ds, threshold)
    return evaluate(ds.docs, Y, pred, D)


def main() -> None:
    force_utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", default="xlm-roberta-base")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--save-to", default="models/xlmr_tagger")
    ap.add_argument("--json-out", default="reports/transformer_results.json")
    ap.add_argument("--out", default="reports/transformer_report.md")
    a = ap.parse_args()

    import torch
    from transformer import TransformerDetector

    print("=" * 70)
    print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"gpu: {p.name}  {p.total_memory / 1e9:.1f} GB  sm_{p.major}{p.minor}")
    print("=" * 70)

    cfg = load_config(a.config)
    rm = role_map(cfg)
    ds = load_dataset(cfg)
    train = ds.filter(split="train")
    dev = ds.filter(split="dev")
    test = ds.filter(split="test")
    dev_seen = dev.filter(roles=["seen"], role_map=rm)
    describe(train, cfg, "train")
    describe(dev_seen, cfg, "dev (seen only)")
    describe(test, cfg, "test")

    # Weight the AI class by inverse frequency so the minority class is not
    # simply ignored by the loss.
    tot = train.n_sentences
    ai = sum(sum(d.labels) for d in train.docs)
    w_h = tot / (2 * (tot - ai))
    w_ai = tot / (2 * ai)
    print(f"\nclass weights: human={w_h:.3f} ai={w_ai:.3f}")

    det = TransformerDetector(
        model_name=a.model, epochs=a.epochs, lr=a.lr,
        batch_size=a.batch_size, grad_accum=a.grad_accum,
        max_length=a.max_length, fp16=not a.no_fp16,
        class_weight=[w_h, w_ai], seed=cfg["seed"])
    if a.cpu:
        import torch as _t
        det.device = _t.device("cpu")
        det.fp16 = False

    def dev_probe(m):
        S, Y, D, I = dev_seen.sentence_view()
        pred, _ = predict_at(m, dev_seen, 0.5)
        return sentence_metrics(Y, pred).f1_ai

    print(f"\nfine-tuning {a.model} ...")
    t0 = time.time()
    det.fit(train.docs, dev_docs=dev_seen.docs, eval_fn=dev_probe)
    train_secs = time.time() - t0
    print(f"training took {train_secs / 60:.1f} min")

    # ---- threshold selection on dev-seen only ---------------------------
    Sd, Yd, Dd, Id = dev_seen.sentence_view()
    _, dev_proba = predict_at(det, dev_seen, 0.5)
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.20, 0.81, 0.025):
        f1 = sentence_metrics(Yd, (dev_proba >= t).astype(int).tolist()).f1_ai
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    print(f"\nthreshold tuned on dev-seen: {best_t:.3f} (dev F1(AI)={best_f1:.4f})")

    # ---- test evaluation ------------------------------------------------
    seen = test.filter(roles=["seen"], role_map=rm)
    held = test.filter(roles=["held_out"], role_map=rm)
    res = {
        "overall": eval_ds(det, test, best_t),
        "seen": eval_ds(det, seen, best_t),
        "held_out": eval_ds(det, held, best_t),
    }
    print("\n=== TEST ===")
    for k, v in res.items():
        print(f"  {k:9s} F1(AI)={v['sentence']['f1_ai']:.3f}  "
              f"acc={v['sentence']['accuracy']:.3f}  "
              f"bound_exact={v['boundary_exact']['f1']:.3f}  "
              f"bound±1={v['boundary_tol']['f1']:.3f}")

    per_gen = {}
    for g in sorted({d.generator for d in test.docs}):
        sub = test.filter(generators=[g])
        per_gen[g] = {"role": rm.get(g), "n_docs": len(sub),
                      **eval_ds(det, sub, best_t)}
        print(f"  {g:18s}[{rm.get(g):8s}] F1(AI)="
              f"{per_gen[g]['sentence']['f1_ai']:.3f}  bound_exact="
              f"{per_gen[g]['boundary_exact']['f1']:.3f}")
    per_type = {}
    for t in sorted({d.construction_type for d in test.docs}):
        sub = test.filter(construction_type=t)
        per_type[t] = {"n_docs": len(sub), **eval_ds(det, sub, best_t)}
        print(f"  {t:36s} F1(AI)={per_type[t]['sentence']['f1_ai']:.3f}  "
              f"bound_exact={per_type[t]['boundary_exact']['f1']:.3f}")

    payload = {
        "model": a.model,
        "epochs": a.epochs, "lr": a.lr, "batch_size": a.batch_size,
        "grad_accum": a.grad_accum, "threshold": best_t,
        "train_minutes": train_secs / 60,
        "dataset": {"train": len(train), "dev": len(dev), "test": len(test)},
        "test": res, "per_generator": per_gen, "per_construction_type": per_type,
    }
    Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json_out).write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                                encoding="utf-8")
    write_report(Path(a.out), payload)
    if a.save_to:
        det.save(a.save_to)
        print(f"saved model -> {a.save_to}")
    print(f"wrote -> {a.out}\nwrote -> {a.json_out}")


def write_report(path: Path, p: dict) -> None:
    lin_path = Path("reports/detector_results.json")
    lin = json.loads(lin_path.read_text(encoding="utf-8")) if lin_path.exists() else None

    L = ["# Transformer detector — XLM-RoBERTa sentence tagger", ""]
    L.append(f"`{p['model']}` fine-tuned for per-sentence human/AI tagging, "
             f"{p['epochs']} epochs, lr {p['lr']}, effective batch "
             f"{p['batch_size'] * p['grad_accum']}, "
             f"{p['train_minutes']:.1f} min on GPU.")
    L.append("")
    L.append("The whole document is encoded in one pass with a marker token "
             "before each sentence; the marker's hidden state is classified. "
             "This is the point of the model: unlike the linear baseline, each "
             "sentence representation is built with its neighbours in "
             "attention range, so the model can represent discontinuity rather "
             "than judging sentences in isolation.")
    L.append("")
    L.append(f"Decision threshold {p['threshold']:.3f}, tuned on dev "
             "restricted to seen generators — the same protocol as the linear "
             "models, so the held-out number stays an honest generalisation "
             "estimate.")
    L.append("")
    L.append("## Test results")
    L.append("")
    L.append("| slice | sent acc | sent F1(AI) | P(AI) | R(AI) | "
             "bound F1 exact | bound F1 ±1 | doc exact |")
    L.append("|---|---|---|---|---|---|---|---|")
    for k, lbl in (("overall", "overall"), ("seen", "seen generators"),
                   ("held_out", "held-out (Gemini)")):
        m = p["test"][k]
        s, b0, bt = m["sentence"], m["boundary_exact"], m["boundary_tol"]
        L.append(f"| {lbl} | {s['accuracy']:.3f} | {s['f1_ai']:.3f} | "
                 f"{s['precision_ai']:.3f} | {s['recall_ai']:.3f} | "
                 f"{b0['f1']:.3f} | {bt['f1']:.3f} | "
                 f"{b0['exact_doc_match']:.3f} |")
    L.append("")

    if lin:
        L.append("## Versus the linear baselines")
        L.append("")
        L.append("| model | sent F1(AI) | bound F1 exact | held-out F1(AI) |")
        L.append("|---|---|---|---|")
        for r in lin["results"]:
            if not r.get("overall"):
                continue
            ho = r.get("held_out", {}).get("sentence", {}).get("f1_ai")
            L.append(f"| {r['name']} | {r['overall']['sentence']['f1_ai']:.3f} "
                     f"| {r['overall']['boundary_exact']['f1']:.3f} | "
                     + (f"{ho:.3f} |" if ho is not None else "— |"))
        t = p["test"]
        L.append(f"| **{p['model']}** | "
                 f"**{t['overall']['sentence']['f1_ai']:.3f}** | "
                 f"**{t['overall']['boundary_exact']['f1']:.3f}** | "
                 f"**{t['held_out']['sentence']['f1_ai']:.3f}** |")
        L.append("")
        L.append("The row that matters is **bound F1 exact**: it is the only "
                 "column the trivial baselines cannot game. `all-AI` scores "
                 "0.000 there by predicting no boundaries at all, and the ±1 "
                 "column rewards scattering boundaries liberally.")
        L.append("")

    L.append("## By generator")
    L.append("")
    L.append("| generator | role | docs | sent F1(AI) | bound F1 exact |")
    L.append("|---|---|---|---|---|")
    for g, m in p["per_generator"].items():
        L.append(f"| {g} | {m['role']} | {m['n_docs']} | "
                 f"{m['sentence']['f1_ai']:.3f} | "
                 f"{m['boundary_exact']['f1']:.3f} |")
    L.append("")
    L.append("## By construction type")
    L.append("")
    L.append("| construction | docs | sent F1(AI) | bound F1 exact |")
    L.append("|---|---|---|---|")
    for t, m in p["per_construction_type"].items():
        L.append(f"| {t} | {m['n_docs']} | {m['sentence']['f1_ai']:.3f} | "
                 f"{m['boundary_exact']['f1']:.3f} |")
    L.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
