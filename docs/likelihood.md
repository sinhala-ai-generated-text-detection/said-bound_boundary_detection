# Likelihood features

[README](../README.md) · [Dataset](dataset.md) · [Detectors](detectors.md) · [Likelihood](likelihood.md) · [Counterfactual twins](counterfactual-twins.md) · [Limitations](limitations.md) · [Reproducing](reproducing.md)

## The idea

A language model assigns a probability to every token. Text a model would
itself have produced is, by definition, **high-probability text** under that
model. Human text is more surprising.

So: **use a language model's own probability estimates as a detection feature.**
This is the family that DetectGPT, Fast-DetectGPT and especially **SeqXGPT**
belong to — SeqXGPT uses per-token log-probability lists for exactly this
sentence-level task. What is new here is applying it to a **low-resource
language where it has not been tested**.

## How the scoring works mechanically

We need, for each sentence, "how predictable was this text?"

**Which model does the scoring?** XLM-R, used as a masked language model. Two
reasons: it has guaranteed Sinhala coverage, and causal multilingual models of
comparable size do not reliably list Sinhala. It is also already local, so no
extra download.

**The masking problem.** A masked LM predicts a token given its surroundings —
but only if that token is *hidden*. If the token can see itself, the prediction
is trivially correct and carries no information. Exact scoring therefore needs
**one forward pass per token**: mask token 1, predict it, unmask; mask token 2,
predict it… For our corpus that is ~1.2 million passes. Infeasible.

**The solution: strided masking.** Mask *every k-th* token in the same pass.
With k = 8, eight passes score the entire document, and no token ever sees
itself:

```
pass 1:  [MASK] t2 t3 t4 t5 t6 t7 t8 [MASK] t10 ...
pass 2:  t1 [MASK] t3 t4 t5 t6 t7 t8 t9 [MASK] ...
...
```

Likelihoods are computed **with the whole document in context**, which is what
makes cross-sentence comparisons meaningful.

## The 14 features per sentence

Eight direct measurements:

| feature | Meaning |
|---|---|
| `mean_logp` | how expected the sentence is overall — the core signal |
| `std_logp` | how uneven that expectation is |
| `min_logp` | the single most surprising token |
| `p10_logp` | robust version of the same |
| `mean_rank` | log rank of the true token among the model's predictions |
| `mean_entropy` | how uncertain the model was at these positions |
| `frac_top1` | fraction of tokens the model would itself have chosen |
| `n_tokens` | sentence length in tokens |

Plus six **delta** features comparing a sentence to its neighbours
(`d_prev_mean_logp`, `d_next_mean_logp`, `abs_d_prev_mean_logp`, etc.).

## The probe — testing the premise before building on it

Before building anything, I tested two hypotheses on 120 dev documents:

| hypothesis | Cohen's *d* | verdict |
|---|---|---|
| machine sentences are more predictable than human ones | **+0.602** | **strong** |
| likelihood *jumps* at an author change | +0.029 | **refuted** |

(Cohen's *d* is a standardised effect size: 0.2 is small, 0.5 medium, 0.8 large.)

- Machine sentences: mean log-prob **−1.06**
- Human sentences: mean log-prob **−1.43**

So machine Sinhala is markedly more predictable — a large, usable effect.

**But my discontinuity hypothesis was wrong.** The size of the likelihood jump
across an author change (0.633) is essentially identical to the jump across a
same-author pair (0.618). Sinhala Wikipedia prose is simply noisy enough
sentence-to-sentence that an authorship change does not stand out as a
likelihood discontinuity.

The feature is therefore an **absolute machine-ness score**, not a change-point
detector. I built the version the evidence supported rather than the one I had
guessed.

## The main result — likelihood closes the generalisation gap

Added to the linear detector:

| model | F1 | bound exact | seen | held-out | **gap** |
|---|---|---|---|---|---|
| text + context | 0.678 | 0.438 | 0.724 | 0.571 | **+0.153** |
| **likelihood only (no text)** | 0.606 | 0.323 | 0.615 | 0.588 | **+0.027** |
| **text + context + likelihood** | **0.704** | 0.450 | **0.743** | **0.615** | +0.128 |

Three things here, in order of importance.

**1. A detector that reads no characters at all reaches 0.606 F1.** The
"likelihood only" row uses *nothing* but the 14 numbers describing how
predictable each sentence is. It beats the position-only baseline (0.567) and
comes close to the full character n-gram model (0.678). That is a striking
result on its own.

**2. It barely degrades on the unseen generator — a 0.027 gap versus 0.153.**
This is the most valuable finding in the project. Character n-grams learn
*generator-specific habits*: the particular character sequences DeepSeek and
GPT-4o favour. Those habits do not transfer to Gemini, hence the large gap.
Likelihood measures something **generator-agnostic** — "is this text
predictable?" — which is a property of machine-generated text in general, not
of any one model.

**3. Combining them beats either alone**, and lifts held-out F1 from 0.571 to
**0.615**. That directly attacks the weakness the study exists to measure.

## Where it did not work, and why

Fusing the same features into the XLM-R tagger (concatenating the 14
standardised values onto each sentence representation before the head):

| | F1 | bound exact | doc exact | held F1 | held bE |
|---|---|---|---|---|---|
| XLM-R baseline | **0.802** | 0.603 | **0.358** | **0.742** | 0.556 |
| XLM-R + likelihood | 0.785 | **0.607** | 0.302 | 0.737 | **0.564** |

**A wash at best.** Exact-boundary F1 moves +0.004, sentence F1 drops 0.017, and
document-exact drops five points. Best epoch fell from 8 to 4 — the extra inputs
mostly brought earlier overfitting.

**The reason is clean, and it is not a failure of the idea.** I scored the
features with `xlm-roberta-base`, and the tagger **is** `xlm-roberta-base`
fine-tuned. A model already has access to its own likelihood estimates — the
features are largely a re-derivation of what the encoder computes internally.
They helped the linear model precisely because character n-grams have **no
access to that information at all**.

**This predicts the fix:** score the likelihood features with a **different
model family** — a causal multilingual LM with different pretraining — so the
signal is genuinely complementary rather than redundant. This has not been run
yet. Note that a very similar English-only idea has since been published as a
zero-shot method (change-point detection over Fast-DetectGPT scores,
arXiv 2605.03723), so on its own it would be an application to a new language
rather than a new method. That is why the project's main technical
contribution moved to [Counterfactual twins](counterfactual-twins.md).

## An engineering detail worth knowing

The first implementation hit CUDA out-of-memory. The cause is instructive:
XLM-R has a **250,002-token vocabulary**, so HuggingFace's masked-LM head
projects every position to 250k logits. For a batch of 8 × 512 tokens that is a
**~4 GB tensor** — on a 6 GB card, before any softmax.

The fix: split the model into its encoder (`roberta`) and its output head
(`lm_head`), run the encoder once for all stride offsets, and apply the
vocabulary projection **only at the masked positions** (~64 per pass instead of
512). This is both the memory fix and an ~8× compute saving, since seven of
every eight positions were being projected and discarded.

**7 hours → 45 minutes** for the full corpus, with output verified identical to
four decimal places.
