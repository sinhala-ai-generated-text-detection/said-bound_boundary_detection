# Detectors

[README](../README.md) · [Dataset](dataset.md) · [Detectors](detectors.md) · [Likelihood](likelihood.md) · [Counterfactual twins](counterfactual-twins.md) · [English replication](english-replication.md) · [Limitations](limitations.md) · [Reproducing](reproducing.md)

## The task, stated precisely

**Input:** a document split into sentences `s₁ … sₙ`.
**Output:** a label for each sentence — 0 = human, 1 = machine.
**Derived:** the boundaries, i.e. the positions where the label changes.

```python
boundaries(labels) = [i for i in 1..n-1 if labels[i] != labels[i-1]]
```

A boundary at index *i* means authorship changed **between sentence i−1 and
sentence i**.

Boundaries are always *computed from* the labels, never stored separately. That
guarantees generation, assembly and evaluation cannot disagree about what a
boundary is.

## Every metric, and what it actually means

This is worth reading carefully, because several of these numbers are
misleading if taken at face value.

### Sentence accuracy
Fraction of sentences labelled correctly.

**Do not use this as your headline.** 63.6% of sentences are human, so a
detector that outputs "human" for everything scores **0.636 accuracy while
detecting nothing at all.**

### Sentence F1 (machine class)
The harmonic mean of precision and recall *on the machine class only*.

- **Precision** = of the sentences we called machine, what fraction really were
- **Recall** = of the sentences that really were machine, what fraction we found

F1 balances the two. Better than accuracy, but still gameable: predicting
"machine" for *everything* gives recall 1.0 and precision 0.364, for
**F1 = 0.542** — a meaningless detector scoring above half.

### Exact-boundary F1 — **the primary metric**
Treats boundaries as the objects being retrieved. We compare the predicted
boundary set to the true one, and compute precision/recall/F1 over boundaries.

This is the metric that matters, for one specific reason: **the trivial
baselines cannot game it.** "All machine" and "all human" both predict a single
label everywhere, which produces *no boundaries at all*, so both score exactly
**0.000**. Any non-zero score here reflects genuine localisation.

### Boundary F1 ±1 (tolerant)
Same, but a predicted boundary within one sentence of a true one counts as a
hit.

**We report this only to argue against using it.** Scattering boundaries
liberally puts one near almost every true boundary, so the *random* baseline
scores **0.542** under it. It rewards over-prediction. Use exact.

### Document-exact accuracy
Fraction of documents where the *entire* boundary set is predicted perfectly.
The strictest metric — one mistake anywhere fails the document. Best model:
**0.358**, i.e. about a third of documents are completely right.

### Seen vs held-out
Every metric is reported three ways: overall, on the two **seen** generators,
and on the **held-out** generator (Gemini). **The gap between the last two is
the most important number in the project** — it separates "learned machine text"
from "learned DeepSeek and GPT-4o".

## The evaluation protocol

- **Train** on `train` only. The held-out generator is absent by construction.
- **Tune** on `dev` **restricted to seen generators**.
- **Report** on `test`, split into overall / seen / held-out.

**Why restrict tuning to seen generators?** Gemini appears in dev *and* test. If
we tuned hyperparameters on all of dev, we would be choosing settings with
information from the very generator whose novelty we then claim to measure. The
held-out number would be quietly contaminated. Restricting model selection to
seen generators keeps it honest.

We also select the **epoch** and **decision threshold** on exact-boundary F1,
not sentence F1. Selecting on one metric while reporting another optimises the
wrong objective — a subtle error that was present in an earlier version and is
now fixed.

## The baselines, and why each one exists

These are not filler. Each exists to close off a specific way of scoring well
without detecting anything.

| baseline | What it does | Score (F1 / bound exact) |
|---|---|---|
| **all-human** | labels everything human | 0.000 / 0.000 |
| **all-AI** | labels everything machine | 0.542 / 0.000 |
| **random** | coin flip weighted by the training machine rate | 0.367 / 0.357 |
| **position only** | **reads no text at all** | 0.567 / 0.284 |

**The position-only baseline is the important one.** It bins each sentence by
its normalised position in the document (sentence 3 of 9 → position 0.33) and
predicts the majority label seen in that bin during training. It never looks at
a single character.

It exists because our construction *deliberately* constrains where boundaries
fall: 20–80% for Type 1, never at the first or last sentence for Types 2/3. That
creates a positional prior, and we need to know how much of any score comes from
exploiting it rather than from reading Sinhala.

Answer: **0.284 exact-boundary F1 from position alone.** Every text model must
beat that before its score means anything.

> **A finding that changed with corpus size.** On an early 20% slice (854
> documents) the position-only baseline scored **0.421** and *beat* the linear
> text model. On the full 4,244-document corpus it drops to **0.284** and loses
> clearly. The early result was an artifact: with few documents, a 10-bin
> position table can fit the boundary distribution closely. **This is a caution
> against drawing baseline conclusions from partial data.**

## The linear detector — and what "text + context" means

This is where the naming in the results tables comes from, so let me be explicit.

### The base model: "text only"

For each sentence independently:

1. Convert the sentence into **character n-gram TF-IDF features**.
2. Feed those into **logistic regression**, which outputs P(machine).

**What are character n-grams?** Instead of splitting into words, we slide a
window over the raw characters. For `ගණිතය`, the 3-grams are `ගණි`, `ණිත`,
`ිතය` … The model learns which character sequences are associated with machine
text.

**Why characters rather than words?** Sinhala is highly inflected and
agglutinative — one root produces many surface forms through suffixes, and there
is no reliable Sinhala word tokenizer. A word-level model would see
`විද්‍යාලය` and `විද්‍යාලයේ` as unrelated. Character n-grams capture the shared
stem and the differing suffix, which is exactly where authorial habit lives.

**TF-IDF** weights each n-gram by how often it appears in this sentence,
discounted by how common it is across all sentences — so distinctive patterns
count more than ubiquitous ones.

Tuned settings: 2–4 character n-grams, `C=4.0`, balanced class weights.

### "text + context" — what the context is, and how it is given

The base model looks at one sentence in isolation. But authorship change is
about how a sentence relates to its **neighbours**.

So "+ context" adds a **second, separate TF-IDF feature block** built from the
neighbouring sentences:

```
sentence i's features = [ TF-IDF of sentence i ] ++ [ TF-IDF of (sentence i-1 + sentence i+1) ]
```

The two blocks are concatenated into one long feature vector. The model can
therefore learn weights that respond to the *combination* — "this sentence looks
like X while its neighbours look like Y".

**Important limitation, and the reason this approach ultimately loses to the
transformer:** concatenating features is *not* the same as comparing them. A
linear model computes a weighted sum of features; it has no way to express
"sentence i differs from its neighbours" as a single quantity, because
difference is not a linear function of concatenated inputs. It can notice that
machine-ish features are present; it cannot represent **discontinuity**.

### The other variants in the table

- **`+ smoothing`** — a post-process that merges predicted authorship runs
  shorter than 2 sentences, on the theory that isolated single-sentence flips
  are noise. **It made things worse** (exact-boundary F1 0.438 → 0.314), because
  Type 3 spans are 1–2 sentences *by design*, so the smoother deletes real spans
  along with the noise. A useful negative result: any structural prior here must
  *permit* single-sentence spans.
- **`+ position` (diagnostic)** — adds the normalised sentence position as a
  feature. Included to measure how much position adds on top of text, not as a
  model we would deploy.

### Linear results

| model | F1 | bound exact | seen | held-out | **gap** |
|---|---|---|---|---|---|
| text only | 0.675 | 0.482 | 0.719 | 0.575 | +0.144 |
| text + context | 0.678 | 0.438 | 0.724 | 0.571 | +0.153 |
| text + context + smoothing | 0.651 | 0.314 | 0.698 | 0.538 | +0.160 |
| text + position | 0.696 | 0.474 | 0.736 | 0.606 | +0.129 |

## The transformer detector — how it actually works

The main model. `xlm-roberta-base`, fine-tuned.

**Why XLM-R?** It is a multilingual encoder pretrained on 100 languages
**including Sinhala**. Monolingual English models have no Sinhala coverage at
all. Its SentencePiece tokenizer handles Sinhala orthography without needing a
language-specific tokenizer.

### The core mechanism: whole-document encoding with sentence markers

This is the key architectural idea, so here it is step by step.

**Step 1 — build one long input from the whole document.** Insert a special
marker token (`<s>`) immediately before each sentence:

```
<s> [sentence 1 tokens] <s> [sentence 2 tokens] <s> [sentence 3 tokens] ...
 ↑                       ↑                       ↑
 marker 1                marker 2                marker 3
```

**Step 2 — run it through the encoder once.** Self-attention means every token
attends to every other token *in the whole document*. So the representation
built at marker 2 has already "seen" sentences 1 and 3.

**Step 3 — read the markers.** Take the final hidden state at each marker
position. That vector is the **sentence representation**: a 768-dimensional
summary of that sentence *in the context of its neighbours*.

**Step 4 — classify.** A linear head maps each marker vector to two logits
(human / machine).

**Why this beats the linear model:** the sentence representation is *built* from
its surroundings by attention, rather than being a bag of features sitting next
to another bag of features. The model can genuinely represent "this sentence
does not fit here", which is what boundary detection needs. That shows up
directly in the numbers: exact-boundary F1 goes from 0.438 to 0.603, a 38%
relative improvement.

### Handling long documents

XLM-R has a 512-token limit. Measured: median document is 335 tokens, and only
**3.0% exceed 512**.

Documents that do exceed it are **chunked on sentence boundaries**, never
truncated mid-document. Truncation would drop sentences while their labels
remained, corrupting the alignment. Predictions are reassembled by
`(document, sentence index)` rather than arrival order, so chunking is
invisible downstream.

### Class weighting

Machine sentences are 36.4% of the data. Without correction the loss is
dominated by the human class. We weight each class by inverse frequency
(human 0.78, machine 1.39) so the minority class is not simply ignored.

### Training configuration

8 epochs, learning rate 3e-5, batch size 4, no gradient accumulation
(5,264 optimizer steps), mixed precision, best-epoch restore.
**109 minutes on an RTX 4050 Laptop (6 GB).** No cluster needed.

> ### The training failure, and the diagnostic that found it
>
> Worth knowing because it nearly produced a false conclusion.
>
> An early run **collapsed to the majority class**: loss frozen at ~0.69
> (= ln 2, i.e. random), dev score identical from epoch 1, zero boundaries
> predicted, test numbers exactly equal to the all-machine baseline. This looks
> *exactly like* a genuine negative result — "the task is too hard".
>
> Rather than guess, I ran the standard diagnostic: **can the model overfit 16
> documents?** If it can memorise a tiny set, the wiring (marker gathering,
> label alignment, chunking) is correct and the fault is elsewhere. It reached
> **F1 = 1.000**, proving the plumbing and localising the fault to the
> optimisation schedule.
>
> The cause: gradient accumulation of 4 left only **165 optimizer steps**.
> XLM-R sits in majority-class collapse for several epochs before it starts
> separating classes, and 165 steps never escaped it. Removing accumulation
> (1,584 steps) fixed it immediately.
>
> **Takeaway: always run the overfit test before believing a negative result.**
> It takes minutes and distinguishes "hard task" from "under-trained model".

### Structured decoding (Viterbi / CRF)

Instead of thresholding each sentence independently, we can decode the whole
label sequence jointly, adding a cost for *changing* author between adjacent
sentences. This is a linear-chain CRF decode; the change-cost is tuned on
dev-seen for exact-boundary F1.

Unlike the run-length smoother, it **penalises** rather than **forbids**
one-sentence spans, so Type 3 stays reachable.

**Honest result:** it helped at mid corpus size (0.583 vs 0.554 on a
2,312-document build) but at full scale it is within noise — 0.607 vs 0.603,
a 0.0004 gap — while losing five points on document-exact accuracy. So
thresholding is reported. The decoder is chosen from the numbers with a
tolerance, not fixed in advance.

### A learned pair head (attempted, abandoned)

I also tried adding a second head that scores each *adjacent pair* of sentences
for "did the author change here", from
`[left; right; |left−right|; left·right]` — targeting the boundary metric
directly. **It stalled training** on two attempts (loss flat near 1.66, dev
score frozen across epochs). It is disabled by default behind `--pair-head`.
Reported as a negative result rather than quietly dropped.

### Final transformer results

| slice | acc | F1 | P | R | **bound exact** | ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| overall | 0.846 | 0.802 | 0.690 | 0.748 | **0.603** | 0.702 | 0.358 |
| seen | 0.865 | 0.830 | — | — | 0.628 | 0.793 | 0.418 |
| held-out | 0.809 | 0.742 | — | — | 0.556 | 0.733 | 0.239 |

**By generator:** DeepSeek 0.816 F1 · GPT-4o 0.843 · **Gemini (held-out) 0.742**

**By construction type:**

| construction | linear F1 | XLM-R F1 | linear bE | XLM-R bE |
|---|---|---|---|---|
| Type 1 — continuation | 0.821 | **0.925** | 0.391 | **0.635** |
| Type 2 — single span | 0.474 | **0.698** | 0.350 | **0.546** |
| Type 3 — multi span | 0.432 | **0.653** | 0.256 | **0.628** |

Two things to read here. **Continuation is far easier than embedded rewriting**
(0.925 vs ~0.67) — a trailing machine block is detectable, short rewritten spans
inside human text are much harder. And **the ordering inverts on boundary F1**,
because a Type 1 document offers exactly one boundary with no partial credit,
while a Type 3 document offers up to six.

## Headline comparison

| model | F1 | **bound exact** | held-out F1 |
|---|---|---|---|
| position only (no text) | 0.567 | 0.284 | 0.550 |
| best linear | 0.678 | 0.438 | 0.571 |
| **XLM-RoBERTa** | **0.802** | **0.603** | **0.742** |
