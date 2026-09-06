"""Local web app for trying the boundary detector on your own Sinhala text.

Loads the fine-tuned XLM-R tagger from models/xlmr_tagger, segments whatever you
paste with the same segmenter the dataset was built with, and labels each
sentence human or machine.

Runs locally because the model is a 1.1 GB PyTorch checkpoint: it cannot run in
a browser, and the text you paste never leaves your machine.

    python src/serve.py                 # then open http://127.0.0.1:5000
    python src/serve.py --port 8000 --cpu
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "detect"))

from flask import Flask, jsonify, request  # noqa: E402

from segment import segment_sentences, word_count  # noqa: E402
from utils import force_utf8_stdout, load_config  # noqa: E402

app = Flask(__name__)
STATE: dict = {}

MODEL_DIR = Path("models/xlmr_tagger")
RESULTS = Path("reports/transformer_results.json")


# ------------------------------------------------------------------ model ---
def load_model(model_dir: Path, cpu: bool = False):
    """Rebuild the tagger and restore the fine-tuned weights."""
    import torch
    from transformers import AutoTokenizer

    from transformer import SentenceTagger

    cfg = load_config()
    thr, bias, base = 0.5, 0.0, "xlm-roberta-base"
    if RESULTS.exists():
        r = json.loads(RESULTS.read_text(encoding="utf-8"))
        thr = float(r.get("threshold", 0.5))
        bias = float(r.get("boundary_bias", 0.0))
        base = r.get("model", base)

    device = "cpu" if cpu or not torch.cuda.is_available() else "cuda"
    print(f"loading {base} weights from {model_dir} on {device} ...")
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = SentenceTagger(base, use_pair_head=False, extra_dim=0)
    state = torch.load(model_dir / "model.pt", map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  note: {len(missing)} params not in checkpoint (e.g. pair head)")
    model.eval().to(device)

    STATE.update(torch=torch, tok=tok, model=model, device=device,
                 threshold=thr, bias=bias, base=base, cfg=cfg)
    print(f"ready. decision threshold {thr:.3f} (tuned on dev, seen generators)")


def analyse(text: str, threshold: float | None = None) -> dict:
    """Segment, score, and label each sentence."""
    torch = STATE["torch"]
    tok, model, device = STATE["tok"], STATE["model"], STATE["device"]

    sentences = segment_sentences(text)
    if not sentences:
        return {"sentences": [], "error": "No sentences found."}

    # Same encoding as training: a marker token before each sentence, whole
    # document in one pass, so each sentence is judged in context.
    marker_id = tok.convert_tokens_to_ids(tok.cls_token)
    budget = 512 - 2
    ids, marker_pos, covered = [], [], []
    for i, s in enumerate(sentences):
        t = tok.encode(s, add_special_tokens=False) or [tok.unk_token_id]
        if len(ids) + len(t) + 1 > budget:
            break                      # sentences past the window are unscored
        marker_pos.append(len(ids))
        ids.extend([marker_id] + t)
        covered.append(i)
    full = [tok.bos_token_id] + ids + [tok.eos_token_id]
    marker_pos = [m + 1 for m in marker_pos]

    with torch.no_grad():
        out = model(
            torch.tensor([full], device=device),
            torch.ones(1, len(full), dtype=torch.long, device=device),
            torch.tensor([marker_pos], device=device),
            torch.ones(1, len(marker_pos), dtype=torch.bool, device=device),
        )
        logits = out[0] if isinstance(out, tuple) else out
        probs = torch.softmax(logits.float(), dim=-1)[0, :, 1].cpu().tolist()

    thr = STATE["threshold"] if threshold is None else float(threshold)
    rows, labels = [], []
    for k, si in enumerate(covered):
        p = float(probs[k])
        lab = int(p >= thr)
        labels.append(lab)
        rows.append({
            "index": si, "text": sentences[si], "prob_ai": round(p, 4),
            "label": lab, "words": word_count(sentences[si]),
        })
    for si in range(len(covered), len(sentences)):
        rows.append({"index": si, "text": sentences[si], "prob_ai": None,
                     "label": None, "words": word_count(sentences[si]),
                     "truncated": True})

    boundaries = [i for i in range(1, len(labels)) if labels[i] != labels[i - 1]]
    n_ai = sum(labels)
    return {
        "sentences": rows,
        "boundaries": boundaries,
        "n_sentences": len(sentences),
        "n_scored": len(labels),
        "n_ai": n_ai,
        "ai_ratio": round(n_ai / len(labels), 4) if labels else 0.0,
        "threshold": thr,
        "default_threshold": STATE["threshold"],
        "truncated": len(covered) < len(sentences),
    }


# -------------------------------------------------------------------- api ---
@app.post("/api/analyse")
def api_analyse():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Empty input."}), 400
    if len(text) > 20000:
        return jsonify({"error": "Text too long (limit 20,000 characters)."}), 400
    thr = data.get("threshold")
    try:
        thr = None if thr is None else min(0.99, max(0.01, float(thr)))
    except (TypeError, ValueError):
        thr = None
    return jsonify(analyse(text, thr))


@app.get("/api/example")
def api_example():
    """A real labelled document from the test split, for a quick sanity check."""
    from utils import read_jsonl
    p = Path(load_config()["paths"]["generated_dir"]) / "combined.jsonl"
    best = None
    for r in read_jsonl(p):
        if r["split"] == "test" and 6 <= len(r["sentences"]) <= 9:
            best = r
            break
    if not best:
        return jsonify({"error": "No example available."}), 404
    return jsonify({
        "text": " ".join(best["sentences"]),
        "gold_labels": best["labels"],
        "title": best["title"],
        "generator": best["generator"],
        "construction_type": best["construction_type"],
    })


@app.get("/")
def index():
    return PAGE


# ------------------------------------------------------------------- page ---
PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sinhala Boundary Detector</title>
<style>
:root{
  --bg:#fbfbfa; --surface:#fff; --ink:#111; --ink-2:#5a5a55; --line:#e3e2dd;
  --human:#1baf7a; --human-bg:#eaf7f1; --ai:#eb6834; --ai-bg:#fdefe8;
  --accent:#2a78d6;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#17171a; --surface:#1f1f23; --ink:#f2f2f0; --ink-2:#a9a9a4; --line:#33333a;
  --human:#2fc98f; --human-bg:#123028; --ai:#ff8055;
  --ai-bg:#3a2018; --accent:#4d94e8;
}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.6 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.sin{font-family:"Noto Sans Sinhala","Iskoola Pota","Nirmala UI",
  system-ui,sans-serif;line-height:2}
header{padding:22px 20px 8px;max-width:960px;margin:0 auto}
h1{margin:0 0 4px;font-size:20px;letter-spacing:-.01em}
.sub{color:var(--ink-2);font-size:13px}
main{max-width:960px;margin:0 auto;padding:12px 20px 60px}
.card{background:var(--surface);border:1px solid var(--line);
  border-radius:10px;padding:16px;margin-bottom:16px}
textarea{width:100%;min-height:170px;padding:12px;border-radius:8px;
  border:1px solid var(--line);background:var(--bg);color:var(--ink);
  font-family:"Noto Sans Sinhala","Iskoola Pota","Nirmala UI",system-ui,sans-serif;
  font-size:15px;line-height:1.9;resize:vertical}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:10px}
button{font:inherit;font-size:14px;padding:9px 16px;border-radius:7px;
  border:1px solid var(--line);background:var(--surface);color:var(--ink);cursor:pointer}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff;font-weight:600}
button:disabled{opacity:.55;cursor:default}
.hint{color:var(--ink-2);font-size:12.5px}
.stats{display:flex;gap:18px;flex-wrap:wrap;margin-bottom:14px}
.stat{min-width:96px}
.stat .n{font-size:22px;font-weight:650;letter-spacing:-.02em}
.stat .l{font-size:11.5px;color:var(--ink-2);text-transform:uppercase;letter-spacing:.05em}
.sent{display:flex;gap:12px;padding:11px 12px;border-radius:8px;margin-bottom:7px;
  border:1px solid transparent;align-items:flex-start}
.sent.h{background:var(--human-bg);border-color:color-mix(in srgb,var(--human) 24%,transparent)}
.sent.a{background:var(--ai-bg);border-color:color-mix(in srgb,var(--ai) 30%,transparent)}
.sent.u{background:transparent;border-color:var(--line);opacity:.65}
.idx{font-size:11.5px;color:var(--ink-2);min-width:20px;padding-top:5px;
  font-variant-numeric:tabular-nums}
.body{flex:1;min-width:0}
.tag{display:inline-block;font-size:10.5px;font-weight:700;letter-spacing:.06em;
  padding:2px 7px;border-radius:4px;margin-bottom:5px;text-transform:uppercase}
.tag.h{background:var(--human);color:#fff}
.tag.a{background:var(--ai);color:#fff}
.tag.u{background:var(--ink-2);color:#fff}
.meter{height:5px;background:var(--line);border-radius:3px;margin-top:7px;overflow:hidden}
.meter i{display:block;height:100%;background:var(--ai);border-radius:3px}
.pct{font-size:11.5px;color:var(--ink-2);font-variant-numeric:tabular-nums;
  margin-left:8px}
.bd{display:flex;align-items:center;gap:9px;margin:9px 0;color:var(--accent);
  font-size:11.5px;font-weight:650;letter-spacing:.04em;text-transform:uppercase}
.bd:before,.bd:after{content:"";flex:1;height:1px;background:var(--accent);opacity:.4}
.err{color:#c0392b;font-size:13.5px}
.legend{display:flex;gap:16px;font-size:12.5px;color:var(--ink-2);margin-top:10px;flex-wrap:wrap}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}
.note{font-size:12.5px;color:var(--ink-2);border-left:2px solid var(--line);
  padding-left:11px;margin-top:14px}
.warn{margin-top:12px;padding:11px 13px;border-radius:8px;font-size:12.5px;
  line-height:1.65;background:var(--ai-bg);
  border:1px solid color-mix(in srgb,var(--ai) 32%,transparent)}
.thr{display:flex;align-items:center;gap:7px;font-size:12.5px;color:var(--ink-2)}
.thr input{width:120px}
.thr b{font-variant-numeric:tabular-nums;color:var(--ink);min-width:30px}
</style></head><body>
<header>
  <h1>Sinhala human–AI boundary detector</h1>
  <div class="sub">Paste Sinhala text. Each sentence is labelled human- or
    machine-written, and the points where authorship changes are marked.</div>
</header>
<main>
  <div class="card">
    <textarea id="t" class="sin" placeholder="ඔබේ සිංහල පෙළ මෙහි අලවන්න..."></textarea>
    <div class="row">
      <button class="primary" id="go">Analyse</button>
      <button id="ex">Load example</button>
      <button id="clr">Clear</button>
      <label class="thr">threshold
        <input type="range" id="thr" min="0.05" max="0.99" step="0.01" value="0.80">
        <b id="thrv">0.80</b></label>
      <span class="hint" id="hint"></span>
    </div>
    <div class="warn"><b>Read this before trusting a result.</b> The model was
      trained only on <i>mixed</i> documents, where every passage contains both
      human and machine sentences. It has never seen an all-human document, so
      on fully human text it still flags about <b>24% of sentences as machine</b>
      (measured on 59 held-out human passages; only 14% came back completely
      clean). Use it to locate <i>where</i> authorship changes in text you
      already believe is mixed &mdash; not to decide whether a document contains
      any machine text at all. Raising the threshold trades this false-positive
      rate against missing real machine sentences.</div>
  </div>
  <div id="out"></div>
</main>
<script>
const $=s=>document.querySelector(s), out=$("#out");
const esc=s=>s.replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));

function render(d){
  if(d.error){out.innerHTML=`<div class="card err">${esc(d.error)}</div>`;return;}
  const pct=(d.ai_ratio*100).toFixed(0);
  let h=`<div class="card"><div class="stats">
    <div class="stat"><div class="n">${d.n_sentences}</div><div class="l">sentences</div></div>
    <div class="stat"><div class="n">${d.n_ai}</div><div class="l">machine</div></div>
    <div class="stat"><div class="n">${pct}%</div><div class="l">machine share</div></div>
    <div class="stat"><div class="n">${d.boundaries.length}</div><div class="l">boundaries</div></div>
  </div>`;
  if(d.truncated) h+=`<div class="err">Only the first ${d.n_scored} sentences
    fit the model's 512-token window; the rest are shown unscored.</div>`;
  h+=`<div class="legend">
    <span><i class="dot" style="background:var(--human)"></i>human</span>
    <span><i class="dot" style="background:var(--ai)"></i>machine</span>
    <span>bar = P(machine), threshold ${d.threshold.toFixed(2)}${
      Math.abs(d.threshold-d.default_threshold)>1e-6
        ? ` (tuned value ${d.default_threshold.toFixed(2)})` : ""}</span></div></div>`;

  h+=`<div class="card">`;
  d.sentences.forEach((s,i)=>{
    if(d.boundaries.includes(i)) h+=`<div class="bd">authorship changes</div>`;
    const cls=s.label===null?"u":(s.label?"a":"h");
    const tag=s.label===null?"not scored":(s.label?"machine":"human");
    h+=`<div class="sent ${cls}"><div class="idx">${s.index}</div><div class="body">
      <span class="tag ${cls}">${tag}</span>`;
    if(s.prob_ai!==null) h+=`<span class="pct">P(machine) ${(s.prob_ai*100).toFixed(1)}%</span>`;
    h+=`<div class="sin">${esc(s.text)}</div>`;
    if(s.prob_ai!==null)
      h+=`<div class="meter"><i style="width:${(s.prob_ai*100).toFixed(1)}%"></i></div>`;
    h+=`</div></div>`;
  });
  h+=`<div class="note">Trained on Wikipedia-derived passages of 6&ndash;12
    sentences from DeepSeek V3 and GPT-4o. On a generator it had never seen
    during training it scores 0.74 sentence F1 and 0.56 exact-boundary F1, so
    treat any single-sentence call as indicative rather than proof. Other
    domains, very short inputs, and human-edited machine text are all outside
    what it was trained on.</div>`;
  out.innerHTML=h+`</div>`;
}

async function go(){
  const text=$("#t").value.trim();
  if(!text){$("#hint").textContent="Enter some text first.";return;}
  $("#go").disabled=true;$("#hint").textContent="analysing...";
  try{
    const r=await fetch("/api/analyse",{method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({text,threshold:parseFloat($("#thr").value)})});
    render(await r.json());$("#hint").textContent="";
  }catch(e){$("#hint").textContent="request failed: "+e;}
  $("#go").disabled=false;
}
$("#thr").oninput=()=>{$("#thrv").textContent=(+$("#thr").value).toFixed(2);};
$("#go").onclick=go;
$("#clr").onclick=()=>{$("#t").value="";out.innerHTML="";$("#hint").textContent="";};
$("#ex").onclick=async()=>{
  $("#hint").textContent="loading example...";
  const r=await fetch("/api/example"),d=await r.json();
  if(d.error){$("#hint").textContent=d.error;return;}
  $("#t").value=d.text;
  const gold=d.gold_labels.map((l,i)=>l?i:null).filter(v=>v!==null);
  $("#hint").textContent=`test-split doc (${d.generator}) — true machine sentences: ${gold.join(", ")}`;
  go();
};
$("#t").addEventListener("keydown",e=>{
  if((e.metaKey||e.ctrlKey)&&e.key==="Enter")go();});
</script></body></html>
"""


def main():
    force_utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(MODEL_DIR))
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()

    md = Path(a.model_dir)
    if not (md / "model.pt").exists():
        raise SystemExit(
            f"No model at {md}. Train one first:\n"
            "  python src/detect/run_transformer.py --epochs 8 --grad-accum 1")
    load_model(md, cpu=a.cpu)
    print(f"\n  open  http://{a.host}:{a.port}\n")
    app.run(host=a.host, port=a.port, debug=False)


if __name__ == "__main__":
    main()
