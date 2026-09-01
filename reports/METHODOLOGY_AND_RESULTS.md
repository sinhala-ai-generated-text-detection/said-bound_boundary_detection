# Sinhala Human–AI Boundary Detection: Methodology and Results

A complete technical account of the dataset construction pipeline and the
detection experiments run on it. Written as a reference for a research paper:
every number below is reproducible from the committed code and logs, and the
source of each is named.

**Status.** All figures come from a **20% slice** of the planned corpus
(854 documents of a planned ~4,600). The pipeline is complete and the remaining
80% is a resumable extension of the same plan, not a re-run. Where dataset size
plausibly limits a result, this is stated explicitly rather than left implicit.

---

## Table of contents

1. [Task definition](#1-task-definition)
2. [Source corpus and cleaning](#2-source-corpus-and-cleaning)
3. [Segmentation and windowing](#3-segmentation-and-windowing)
4. [Split protocol](#4-split-protocol)
5. [Generator models](#5-generator-models)
6. [Construction types](#6-construction-types)
7. [Worked examples](#7-worked-examples)
8. [Prompting](#8-prompting)
9. [Validation](#9-validation)
10. [Dataset composition](#10-dataset-composition)
11. [Cross-model comparison](#11-cross-model-comparison)
12. [Detection experiments](#12-detection-experiments)
13. [Findings](#13-findings)
14. [Threats to validity](#14-threats-to-validity)
15. [Future work](#15-future-work)
16. [Reproduction](#16-reproduction)

---

## 1. Task definition

Given a document of *n* sentences, assign each sentence a label
`0 = human-written` or `1 = machine-written`, and thereby locate the
**boundaries** at which authorship changes.

A boundary at index *i* means the label changed between sentence *i−1* and
sentence *i*. Boundaries are always *derived* from the label sequence rather
than stored independently, so every stage of the pipeline — generation,
assembly, evaluation — agrees on the convention by construction:

```python
boundaries(labels) = [i for i in 1..n-1 if labels[i] != labels[i-1]]
```

This is harder than document-level AI-text detection. The detector cannot rely
on a global judgement about the document; it must localise a *stylistic
discontinuity* inside text that is largely human-written, in a low-resource
language.

---

## 2. Source corpus and cleaning

**Source.** `wikipedia.parquet` — a HuggingFace export of Sinhala Wikipedia.
26,470 articles, 173 MB, columns `id, url, title, raw_mediawiki, text`.

**Cleaning source: `raw_mediawiki`, not `text`.** The shipped `text` column is
a pre-stripped rendering, but it is unreliable: it still contains literal
`Category:` lines, unexpanded `{{#ifexpr:...}}` template calls, and
`__NOTOC____NOEDITSECTION__` magic words, and it flattens list bullets to
leading spaces — which corrupts sentence segmentation. Cleaning from raw
wikitext gives exact control over what is removed.

### Filter cascade

Applied in order; each row shows how many pages that filter removed.

| # | Filter | Removed | % of raw | Remaining |
|---|---|---|---|---|
| 0 | raw articles | — | — | 26,470 |
| 1 | main page (`මුල් පිටුව`) | 1 | 0.00% | 26,469 |
| 2 | namespace prefix (`Talk:`, `සාමාජික:`, `Draft:`, …) | 7 | 0.03% | 26,462 |
| 3 | redirects | 6 | 0.02% | 26,456 |
| 4 | disambiguation (`{{බහුරුත්හරණය}}`) | 166 | 0.63% | 26,290 |
| 5 | stubs (`{{දියුණු කරන්න}}`, `{{stub}}`) | 1,672 | 6.32% | 24,618 |
| 6 | list pages by title | 383 | 1.45% | 24,235 |
| 7 | list pages by content | 2,375 | 8.97% | 21,860 |
| 8 | prose gate: too few characters | 8,168 | 30.86% | 13,692 |
| 9 | prose gate: < 120 words | 3,100 | 11.71% | 10,592 |
| 10 | not predominantly Sinhala script | 428 | 1.62% | 10,164 |
| 11 | prose gate: < 8 sentences | 774 | 2.92% | **9,390** |

**9,390 articles survive (35.5% of the dump).** The dominant loss is the prose
gate (filters 8–11 remove 12,470 pages, 47%), which reflects the size profile of
Sinhala Wikipedia: the median raw article is only 2,048 characters.

Namespace filtering is nearly a no-op — only 43 of 26,470 titles carry any
`X:` prefix, and most are article titles containing a colon. The dump is
already essentially main-namespace. This was verified before relying on it.

### Markup removal

Applied to `raw_mediawiki`, in order: HTML comments `<!-- -->`; `<ref>` blocks;
tables `{| … |}`; templates `{{…}}` (nested-aware); file and image links
`[[File:|ගොනු:|රූපය:…]]`; category links `[[Category:|ප්‍රවර්ගය:…]]`;
interwiki links; then `[[link|text]] → text` and `[[link]] → link`; then
`'''bold'''` / `''italic''` markers; then whitespace normalisation.

**Paragraph reflow.** A subtle and consequential detail: in MediaWiki a
*single* newline inside a paragraph is a soft wrap, not a break — only a blank
line starts a new paragraph. Treating single newlines as breaks split sentences
mid-clause and corrupted every downstream sentence count. `_reflow()` implements
the correct semantics.

---

## 3. Segmentation and windowing

### Segmenter

One deterministic segmenter (`src/segment.py`) is used everywhere — source
preparation, generation length targets, labelling, and evaluation. Using
different logic at any stage would silently misalign sentences and labels.

Splits on `.`, `?`, `!`, `…`, and `෴` (kunddaliya), with guards for:

- **Decimals** — `3.14` must not split.
- **Abbreviations** — `ආචාර්ය.`, `මහාචාර්ය.`, `පෙ.ව.`, `ප.ව.`, and Latin
  `Dr.`, `Mr.`, `etc.`
- **ASCII initials** — `A. B. Smith`.

> **Bug found and fixed.** The initial-detection guard originally treated *any*
> single letter before a period as an initial. In Sinhala, `ය.` is an extremely
> common sentence-final particle, so the guard was silently merging sentences
> and under-counting throughout the pipeline. The guard is now ASCII-only.
> Covered by regression tests.

### Windowing

Long articles are reduced to a contiguous window of **6–12 sentences**
(configurable) rather than used whole, so that generation targets stay bounded
and documents are of comparable size. Every window records its `source_id`, so
all derivatives trace back to one article.

- **9,390 sources → 7,944 windows**
- Window length: min 6, median 9, max 12 sentences
- Source articles: p25 = 13, p50 = 21, p75 = 41 sentences

---

## 4. Split protocol

Splits are assigned **over source article IDs, before any generation**, seeded
with 42:

| split | source IDs | windows |
|---|---|---|
| train | 6,573 (70%) | 5,544 |
| dev | 1,408 (15%) | 1,204 |
| test | 1,409 (15%) | 1,196 |

Every derivative of a source inherits its source's partition. This is what
prevents the most damaging leak available to this task: the same human passage
appearing, with different AI spans, on both sides of the train/test boundary. A
detector could otherwise memorise the human sentences rather than learn the
distinction.

**Verified invariants** (checked on the built dataset, all pass):

- No `source_id` appears in more than one split — 0 violations.
- Every record's split matches `splits/*.txt` — 0 mismatches.
- The held-out generator has 0 records in train.

---

## 5. Generator models

Three models, in two roles. **Seen** generators write all three splits.
The **held-out** generator writes only dev and test, so that generalisation to
an unseen generator can be measured.

| role | model | slug (pinned) | in $/M | out $/M |
|---|---|---|---|---|
| seen | DeepSeek V3 | `deepseek/deepseek-chat` | 0.25 | 0.85 |
| seen | GPT-4o | `openai/gpt-4o-2024-11-20` | 2.50 | 10.00 |
| held-out | Gemini 2.5 Pro | `google/gemini-2.5-pro` | 1.25 | 10.00 |

All generation used `temperature = 0.8`, `top_p = 0.95`, `max_tokens = 1200`,
via OpenRouter. Prices were verified against the live OpenRouter model list at
runtime rather than assumed.

**Reasoning suppression.** Gemini 2.5 Pro is a reasoning model. It was
configured with `reasoning: {effort: minimal, exclude: true}`, and the client
parses only message content, never a reasoning field. **Zero of the 854
accepted records contain reasoning-trace markers**, verified by pattern scan.

### Mistral Nemo was tested and rejected

The original plan specified `mistralai/mistral-nemo` as the second seen
generator. It was piloted and **dropped on quality grounds**.

Its call-level validation pass rate was **9.2%**, against 55–67% for the other
three. But the pass rate understates the problem. Observed failure modes:

1. **Degenerate word-salad** — malformed non-words (`මාත්‍රිකාවකිනීම්`,
   `කැමින්නටුවක්`) in syntactically shaped but meaningless sentences.
2. **Prompt-instruction leakage into output** — generations containing the
   instructions themselves as content, e.g. `වාක්‍ය 18 සිට 30 අතර විය යුතුය`
   ("sentences must be between 18 and 30"), `පෙළ පමණක් ලබා දෙන්න` ("provide
   only the text"). One output was simply `මම පිළිපීඩා කරමි` ("I comply").
3. **Repetition degeneration** — `පත්‍රිකාවක් පත්‍රිකාවක් ගියේය`.
4. **Meta-commentary as content** — `ජැමෙයිකාවේ රජය සහ දේශපාලනයක් විකිපීඩියා
   ලිපියකි` ("Jamaica's government and politics is a Wikipedia article").

**The decisive point: the 7 documents it *did* pass were also unusable.**
Failure modes 2–4 appear in accepted output. They passed only by landing on the
right sentence and word counts.

This exposes a real limitation of the validator, which matters for the paper:
**no automatic check tests meaning.** The script check measures the *fraction of
characters in the Sinhala Unicode block*, so text made of Sinhala characters
that is not Sinhala *language* scores `sinhala_ratio = 1.0` and passes. Any
future generator must be read by a human before it is trusted; the validator is
a format gate, not a fluency gate.

Those 7 records were removed from the dataset and archived as evidence in
`reports/evidence/mistral_nemo_pilot_records.jsonl` (with 128 rejections
alongside). GPT-4o replaced it.

---

## 6. Construction types

Two primitives only: **continuation** and **context-aware span replacement**.
Span replacement — never insertion of invented sentences — so that the human
text around a boundary remains genuine and the AI text is constrained to say
what the human text said.

### Type 1 — single boundary (continuation)

A human prefix followed by an AI continuation. The split point is sampled at
**20–80%** of the passage, sentence-aligned, with ≥2 human and ≥2 AI sentences
enforced. The true human remainder is **discarded** and the model writes a
replacement, conditioned on the prefix and given the removed portion's sentence
and word counts as targets.

- Labels: `[0…0, 1…1]` — exactly one boundary.
- Eligibility: ≥ 4 sentences.
- Realised: mean 1.00 boundaries, mean AI fraction 0.491.

The 20–80% sampling is deliberate: a fixed midpoint would let a detector score
well by always guessing the middle.

### Type 2 — one internal AI segment (span replacement)

An internal span of 1–3 sentences is replaced. The span cannot touch sentence 0
or the final sentence, and ≥2 human sentences are preferred on each side. The
model receives `LEFT + TARGET + RIGHT` and is instructed to rewrite **only**
TARGET, preserving its facts, using LEFT/RIGHT for flow but not copying them.

- Labels: two boundaries (span open, span close).
- Eligibility: ≥ 6 sentences.
- Realised: mean 2.00 boundaries, mean AI fraction 0.246.

### Type 3 — multiple internal AI segments

2–3 non-adjacent spans of 1–2 sentences each, no overlaps, ≥1 human sentence
between spans, none at start or end.

**Each span is generated independently from the ORIGINAL human document.** Span
2 is never conditioned on span 1's output. This matters: conditioning would let
the model build a self-consistent AI register across spans, making the
document more uniform than a real multi-edit document would be. The same
generator is used for all spans within one document.

- Labels: up to six boundaries.
- Eligibility: ≥ 8 sentences.
- Realised: mean 5.01 boundaries, mean AI fraction 0.332.

**Documents are never padded into a higher type.** A short article is simply
ineligible.

---

## 7. Worked examples

Real records from the dataset, with gold labels.

### Type 1 — `මැඩගස්කරයේ භූගෝලය` (Geography of Madagascar), DeepSeek V3

`labels = [0,0,0,0,1,1]`, `boundaries = [4]`, boundary index 4.

| # | label | sentence |
|---|---|---|
| 0 | HUMAN | මිනිසුන් මැඩගස්කරයට පැමිණීමෙන් පසු අවම වශයෙන් ලීමර් විශේෂ 17 ක් වඳ වී ගොස් ඇති අතර, ඒවා සියල්ලම ඉතිරිව ඇති ලීමර් විශේෂයට වඩා විශාල විය. |
| 1 | HUMAN | ෆොසා (බළලුන් වැනි) ඇතුළු තවත් ක්ෂීරපායින් ගණනාවක් මැඩගස්කරයට ආවේණික වේ. |
| 2 | HUMAN | දිවයිනේ කුරුලු විශේෂ 300කට අධික ප්‍රමාණයක් වාර්තා වී ඇති අතර ඉන් සියයට 60කට වැඩි ප්‍රමාණයක් ආවේණික වේ. |
| 3 | HUMAN | (එක් ආවේණික පවුලක් ඇතුළුව). |
| 4 | **AI** | මැඩගස්කරයේ සීනීන් ඇතුළුව උභයජීවීන්ගේ විශේෂ 370කට අධික ප්‍රමාණයක් වාර්තා වී ඇති අතර ඉන් සියයට 99කට වැඩි ප්‍රමාණයක් ආවේණික වේ. |
| 5 | **AI** | තවද මෙම දිවයිනේ කටුස්සා වැනි සත්ව විශේෂ ගණනාවක් පමණක් නොව සත්ව ගණයේ අනන්‍යතාවයන් දක්නට ලැබේ. |

Note the difficulty: sentence 4 continues the endemism statistics pattern of
sentence 2 fluently. The boundary is not lexically obvious.

### Type 2 — `රෙකෝන්ඩය` (Recorder), GPT-4o

`labels = [0,0,0,1,0,0]`, `boundaries = [3,4]`, replaced span `[3,4)`.

| # | label | sentence |
|---|---|---|
| 2 | HUMAN | මෙය දැවය භාණ්යක් වන අතර ඇගිලි ගණනක් සදහා සිදුරු සහිත මෙය මුඛ්‍ය හා සම්බන්ධ මත ස්ථානයේ වඩා පලල්වද ඉතා දුරට යන්නට සිහින්මය යන ආකාරයද නිපදවා ඇත. |
| 3 | **AI** | මධ්‍ය යුගයේ විශේෂයෙන්ම ජනප්‍රිය වූ රෙකෝඩරය, දහ අටවන ශතවර්ෂයෙන් පසු සංගීත උපකරණයකි, නමුත් ඊට පසුව එහි භාවිතය සන්සුන් වශයෙන් අඩු විය. |
| 4 | HUMAN | මධ්‍යතන යුගයේ මෙම සංගීත භාණ්ය කොතරම් ජනප්‍රියව කිබණේද යත් සාහිත්‍යමය කලා නිර්මාණ ආදියේ නිර්තන්තර එම සංගීත භාණ්ඩය පිළිබද සටහන් දැකිය හැකි විය.|

This example illustrates an important property of the data: the **AI sentence
is more fluent and better-formed than the human sentences around it**. Sinhala
Wikipedia contains substantial typographical noise (`භාණ්යක්`, `කිබණේද`,
`මෙ මංගීත`). A detector may therefore learn *"cleaner text = AI"* — a shortcut
discussed in §14.

### Type 3 — `උගන්ඩාවේ භූගෝලය` (Geography of Uganda), DeepSeek V3

`labels = [0,0,1,0,1,0,1,0]`, `boundaries = [2,3,4,5,6,7]`,
spans `[[2,3],[4,5],[6,7]]` — three single-sentence AI spans, each isolated
between human sentences.

| # | label | sentence |
|---|---|---|
| 1 | HUMAN | රට සාමාන්‍යයෙන් මුහුදු මට්ටමේ සිට මීටර් 900 ක උසකින් පිහිටා ඇත. |
| 2 | **AI** | උගන්ඩාවේ නැගෙනහිර හා බටහිර දෙපසම කඳුකරයන් පිහිටා ඇත. |
| 3 | HUMAN | රුවෙන්සෝරි කඳුවැටිය උගන්ඩාවේ උසම කඳු මුදුන අඩංගු වන අතර … මීටර් 5,094 කි. |
| 4 | **AI** | රටේ දකුණු කොටසෙහි විශාල ප්‍රදේශයක් … වික්ටෝරියා විලේ බලපෑමට යටත්ව ඇති අතර එහි බොහෝ දූපත් දක්නට ලැබේ. |
| 5 | HUMAN | කම්පාලා අගනුවර සහ එන්ටෙබේ අගනුවර ඇතුළුව මෙම වැව ආසන්නයේ වඩාත් වැදගත් නගර දකුණේ පිහිටා ඇත. |
| 6 | **AI** | රටේ මධ්‍යයේ පිහිටි කියෝගා විල පුළුල් වගුරු බිම් වලින් යුක්ත වේ. |

Single-sentence alternating spans are the hardest configuration in the dataset,
and they are the reason run-smoothing fails (§12.4).

---

## 8. Prompting

Prompt templates live in `prompts/task2_prompts.yaml`, one family per
`domain × primitive`: `single_boundary_prefix`, `span_replacement`, and
`single_boundary_reverse` (defined, unused so far). All prompts are written in
Sinhala and instruct the model to:

- write **only** in Sinhala;
- produce no markdown, headings, or lists;
- preserve facts — names, dates, numbers;
- not fabricate unverifiable specifics;
- return **only** the generated segment, with no preamble.

Length control is explicit: each prompt carries `target_sentence_count` and a
`target_word_count` **range** derived from the replaced span.

---

## 9. Validation

Pipeline per attempt: **generate → clean → segment → validate →
PASS: keep / FAIL: retry (≤3) / 3× fail: log and skip.** Model output is never
hand-repaired.

Seven automatic checks:

| # | check | rule |
|---|---|---|
| 1 | sentence count | matches the requested count exactly |
| 2 | length | within ±25% of the target word count |
| 3 | no preamble | rejects `Sure…`, `Certainly…`, `මෙන්න…` etc. |
| 4 | no markdown | rejects `#`, `*`, `-` bullets, code fences |
| 5 | script | predominantly Sinhala, Latin proper nouns allowed |
| 6 | not a copy | normalised comparison against the original span |
| 7 | number consistency | numeric tokens must not mutate (e.g. 2025 → 2027) |

### Rejection statistics (all runs, 2,393 API calls)

| reason | count | share |
|---|---|---|
| length | 664 | 54.7% |
| sentence_count | 282 | 23.2% |
| numbers | 169 | 13.9% |
| script | 46 | 3.8% |
| copy | 45 | 3.7% |
| markdown | 8 | 0.7% |

**Length dominates.** Models write Sinhala spans shorter than the human text
they replace: mean signed deviation is −3.1 words (DeepSeek), −7.9 (GPT-4o),
−4.0 (Gemini). Deviations are bimodal, spanning roughly −60% to +64%.

The ±25% tolerance was **deliberately not loosened**. Loosening to ±35% would
rescue only ~2 of 12 sampled failures while admitting exactly the length
artifact the design guards against — if AI spans were systematically shorter,
a detector could exploit sentence length instead of style. The cost is roughly
1.6 API calls per accepted document.

Rejections by generator: DeepSeek 439, GPT-4o 342, Gemini 100, Mistral 128.

---

## 10. Dataset composition

**854 documents · 7,935 labelled sentences · 798 unique source articles.**
Integrity failures: 0.

### By generator and split

| generator | role | train | dev | test | total |
|---|---|---|---|---|---|
| deepseek_v3 | seen | 249 | 51 | 52 | 352 |
| gpt_4o | seen | 253 | 59 | 53 | 365 |
| gemini_2_5_pro | held-out | **0** | 62 | 75 | 137 |
| **total** | | **502** | **172** | **180** | **854** |

### By construction type

| type | docs | mean boundaries | mean AI fraction |
|---|---|---|---|
| Type 1 — single boundary | 313 | 1.00 | 0.491 |
| Type 2 — one internal span | 285 | 2.00 | 0.246 |
| Type 3 — multiple spans | 256 | 5.01 | 0.332 |

**Class balance:** 2,844 AI sentences of 7,935 (**35.8%**). Document length:
p25 = 8, p50 = 9, p75 = 11 sentences.

**Retries:** 679 documents accepted first try, 116 after one retry, 55 after
two, 4 after three or more.

### Boundary position distribution

Normalised first-boundary position, by decile:

| decile | 0–.1 | .1–.2 | .2–.3 | .3–.4 | .4–.5 | .5–.6 | .6–.7 | .7–.8 | .8–.9 | .9–1 |
|---|---|---|---|---|---|---|---|---|---|---|
| count | 18 | 87 | 198 | 163 | 115 | 140 | 61 | 57 | 15 | 0 |

Spread rather than spiked, with only **2.1%** in the first decile. The
distribution is not uniform — it peaks at 0.2–0.3 — which is a direct
consequence of the 20–80% sampling rule and the constraint that spans cannot
touch the first or last sentence. **This is measurable and a detector can
exploit it**, which is why a position-only baseline is reported in §12.

---

## 11. Cross-model comparison

To compare generators fairly, 20 matched examples were produced in which
**every model rewrote the same span of the same window under the same plan** —
the span-selection RNG is normally seeded per generator, which would otherwise
give each model a different span. Drawn from dev/test so the held-out generator
is permitted. Full text in `reports/model_comparison.md`.

| model | produced | mean Δ words | mean abs Δ words | mean Δ sentences | mean retries |
|---|---|---|---|---|---|
| deepseek_v3 | 17/20 | −3.1 | 5.5 | +0.00 | 0.47 |
| gpt_4o | 16/20 | −7.9 | 8.1 | +0.00 | 0.25 |
| gemini_2_5_pro | 18/20 | −4.0 | 6.2 | +0.00 | 0.11 |

All three match the requested **sentence** count exactly (Δ = 0.00) — that check
is binding — while under-shooting **word** count. GPT-4o under-writes most;
Gemini needs fewest retries.

One matched example failed for **all three models**: a 7-sentence, 196-word
target. Long targets are where every model under-generates, and the failure is
systematic rather than model-specific.

### Qualitative fluency assessment

- **DeepSeek V3** — fluent, natural encyclopedic Sinhala; correct honorifics
  and inflection. Sentences run longer than the human originals, which is why
  `length` is its most common rejection.
- **GPT-4o** — fluent with the strongest instruction-following. Handles
  technical vocabulary well: in a quantum-computing article it produced
  `කුබිට්ස්` (qubits) and `සුපර්පොසිෂන්` (superposition) rather than falling
  back to English.
- **Gemini 2.5 Pro** — the most natural Sinhala of the three. Correct
  honorifics (`පියතුමා`, `මහතා`, `මහත්මිය`), accurate Sri Lankan place names
  and historical detail.

---

## 12. Detection experiments

### 12.1 Protocol

- **Train** on `train` — seen generators only, automatically, since the
  held-out generator is barred from train by construction.
- **Tune** on `dev` **restricted to seen generators**. This is a deliberate
  and important choice: Gemini appears in dev *and* test, so tuning on all of
  dev would select hyperparameters (and the decision threshold) using the very
  generator whose novelty is then being reported. Restricting model selection
  to seen generators keeps the held-out number an honest estimate.
- **Report** on `test`, split into overall / seen / held-out.

**Read F1 on the AI class, not accuracy.** The AI class is 35.8% of sentences,
so a degenerate all-human predictor scores 0.644 accuracy while detecting
nothing.

**Read exact-boundary F1, not ±1.** With ±1 tolerance even a random baseline
scores 0.542, because scattering boundaries liberally puts one near almost
every true change. The tolerant metric rewards over-prediction. Exact-boundary
F1 is also the only metric the trivial baselines cannot game: `all-AI` scores
0.000 because predicting a single class produces no boundaries at all.

### 12.2 Baselines and linear models

| model | sent acc | F1(AI) | **bound F1 exact** | bound F1 ±1 | seen F1 | held-out F1 |
|---|---|---|---|---|---|---|
| all-human | 0.644 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| all-AI | 0.356 | 0.526 | 0.000 | 0.000 | 0.526 | 0.525 |
| random (train prior) | 0.552 | 0.371 | 0.360 | 0.542 | 0.377 | 0.349 |
| **position only (no text)** | 0.686 | 0.518 | **0.421** | 0.612 | 0.540 | 0.487 |
| linear, text only | 0.707 | 0.574 | 0.377 | 0.572 | 0.629 | 0.490 |
| linear, text + context | 0.720 | **0.586** | 0.295 | 0.508 | 0.638 | 0.508 |
| linear, + smoothing | 0.716 | 0.561 | 0.201 | 0.375 | 0.625 | 0.464 |
| linear, + position | 0.674 | 0.580 | 0.366 | 0.581 | 0.639 | 0.495 |

Linear model: character n-gram TF-IDF (`char_wb`, 3–5-grams, `min_df=2`,
sublinear TF, `C=0.5`, balanced class weights), logistic regression per
sentence. Character n-grams rather than words because Sinhala is highly
inflected and agglutinative.

The **position-only baseline** predicts from normalised sentence position
alone, using no text whatsoever. It is the bar every text model must clear.

### 12.3 Transformer

`xlm-roberta-base` fine-tuned as a document-level sentence tagger. The whole
document is encoded in one pass with a marker token (`<s>`) inserted before each
sentence; that marker's final hidden state is classified by a linear head. Each
sentence representation is therefore built with its neighbours in attention
range, which is what lets the model represent *discontinuity* rather than
judging sentences in isolation.

Documents exceeding the 512-token window are chunked **on sentence boundaries**,
never truncated, so sentence/label alignment always holds. Measured token
lengths: p50 = 335, p90 = 467, p99 = 604; only **3.0%** of documents exceed 512.

Hyperparameters: 12 epochs, lr 3e-5, batch 4, `grad_accum=1` (1,584 optimizer
steps), fp16, inverse-frequency class weights (human 0.780 / AI 1.393),
best-epoch restore. Decision threshold 0.800, tuned on dev-seen.
**24.6 minutes on an RTX 4050 Laptop (6 GB), 5.9 GB VRAM.**

| slice | sent acc | F1(AI) | P(AI) | R(AI) | **bound F1 exact** | bound F1 ±1 | doc exact |
|---|---|---|---|---|---|---|---|
| overall | 0.790 | 0.718 | 0.690 | 0.748 | **0.513** | 0.702 | 0.178 |
| seen | 0.805 | 0.749 | 0.692 | 0.816 | 0.554 | 0.760 | 0.238 |
| held-out | 0.771 | 0.671 | 0.687 | 0.655 | 0.460 | 0.629 | 0.093 |

#### A training failure worth recording

The first run **collapsed to the majority class**: loss frozen at ~0.69 (= ln 2),
dev F1 identical from epoch 1, zero boundaries predicted, test numbers exactly
equal to the all-AI baseline.

Rather than guess, an **overfit diagnostic** was run: train on 16 documents and
check whether the model can memorise them. It reached F1 = 1.000, which proved
the wiring — marker gathering, label alignment, chunking — was correct and
localised the fault to the optimisation schedule.

The cause was `grad_accum=4` at lr 2e-5, giving only **165 optimizer steps**.
XLM-R sits in a majority-class collapse for several epochs before it begins
separating classes; the overfit test showed escape took ~10 epochs. Fixing this
(`grad_accum=1`, lr 3e-5, 12 epochs = 1,584 steps) produced the results above.
Best-epoch restore mattered: epoch 10 (dev 0.742) beat the final epoch 12
(0.737).

**This is a methodological point worth reporting in a paper.** A transformer
that collapses to the majority class produces numbers that look like a genuine
negative result about the task. The overfit test distinguishes "the task is
hard" from "the optimiser did not run long enough" in a few minutes.

### 12.4 Headline comparison

| model | sent F1(AI) | **bound F1 exact** | held-out F1(AI) |
|---|---|---|---|
| position only (no text) | 0.518 | 0.421 | 0.487 |
| best linear | 0.586 | 0.295 | 0.508 |
| **XLM-RoBERTa** | **0.718** | **0.513** | **0.671** |

### 12.5 By generator and construction type (XLM-R)

| generator | role | sent F1(AI) | bound F1 exact |
|---|---|---|---|
| deepseek_v3 | seen | 0.761 | 0.562 |
| gpt_4o | seen | 0.736 | 0.547 |
| gemini_2_5_pro | **held-out** | **0.671** | **0.460** |

| construction | linear F1 | **XLM-R F1** | linear bE | **XLM-R bE** |
|---|---|---|---|---|
| Type 1 — single boundary | 0.787 | **0.877** | 0.262 | **0.410** |
| Type 2 — one internal span | 0.357 | **0.588** | 0.226 | **0.493** |
| Type 3 — multiple spans | 0.346 | **0.592** | 0.152 | **0.570** |

---

## 13. Findings

**1. Boundary detection in Sinhala is feasible but far from solved.** The best
detector reaches 0.513 exact-boundary F1 and 0.718 sentence F1(AI). Only 17.8%
of documents have their boundary set predicted exactly right.

**2. Document context is necessary, not merely helpful.** The linear model
scores *below* the position-only baseline on exact boundaries (0.295 vs 0.421)
despite higher sentence accuracy. A per-sentence classifier can recognise
"machine-sounding sentences" but cannot localise change points, because it
structurally cannot represent discontinuity between neighbours. Encoding the
document jointly raises exact-boundary F1 from 0.295 to 0.513.

**3. Generalisation to an unseen generator is materially harder.** XLM-R drops
from 0.749 (seen) to 0.671 (held-out) sentence F1, and 0.554 to 0.460 on exact
boundaries. Document-exact accuracy more than halves, 0.238 → 0.093. The
detector has partly learned *these two models' habits* rather than machine text
in general. **This gap is the single most important number in the study**, and
it is only visible because the held-out generator was excluded from training
before generation began.

**4. Continuation is much easier than span replacement.** Type 1 reaches 0.877
sentence F1; Types 2 and 3 reach 0.588 and 0.592. A trailing AI block is
detectable; short rewritten spans embedded in human text are far harder. Note
the inversion on boundary F1: Type 1 scores *lowest* (0.410) despite the
highest sentence F1, because it offers exactly one boundary to get right, while
Type 3 offers up to six and partial credit accumulates.

**5. Naive post-processing hurts.** Merging predicted author runs shorter than
two sentences dropped exact-boundary F1 from 0.295 to 0.201. Type 3 spans are
1–2 sentences *by design*, so the smoother deletes genuine spans. Any structural
prior over run length must permit single-sentence spans.

**6. Automatic validation cannot gate fluency.** Mistral Nemo passed the script
check with `sinhala_ratio = 1.0` while producing meaningless text, and 7 of its
accepted documents were unusable. Format checks and quality checks are different
things.

---

## 14. Threats to validity

**Dataset size.** All results are from 502 training documents (20% slice). Dev
F1 was still climbing at epoch 10–12 while training loss reached 0.084 — the
signature of a model limited by data volume, not capacity. The transformer
numbers should be treated as a lower bound.

**Source-text noise as a confound.** Sinhala Wikipedia contains substantial
typographical and grammatical noise, while all three generators produce clean,
well-formed prose. A detector may partly be learning *"cleaner text = AI"*
rather than authorship style. The Type 2 example in §7 shows this directly. This
is arguably a genuine signal, but it is a *corpus* property, not a *model*
property, and would not transfer to well-edited human text. **Not yet measured.**

**Positional prior.** By construction, boundaries fall at 20–80% and spans
cannot touch the first or last sentence. Position alone yields 0.421
exact-boundary F1. Reported baselines quantify this, but do not remove it.

**Length prior.** The ±25% validation gate constrains AI spans to a length band
around the human text, but generators still under-write systematically
(−3 to −8 words). A residual length cue may persist.

**Single domain.** Wikipedia only. Prompt families are parameterised by domain
and News/QA are planned, but nothing outside encyclopedic prose is tested.

**Single held-out generator.** Generalisation is measured against exactly one
unseen model. A 0.078 F1 gap on one generator is suggestive, not conclusive.

**No human ceiling.** No annotation study establishes how well Sinhala readers
perform on this task, so it is not known whether 0.513 exact-boundary F1 is
close to or far from the achievable maximum.

---

## 15. Future work

### 15.1 Generation

**Complete the corpus.** The immediate step: run the remaining 80% (~3,680
documents, ~$22, ~2.5–3 h). The run is nested — the current 854 records are a
verified strict prefix — so it resumes and extends rather than regenerating.
Verified: 20% ⊂ 50% ⊂ 100%, deterministic, proportional across generator, type
and split.

**Add a second held-out generator.** One unseen model cannot separate
"generalises to unseen generators" from "generalises to Gemini specifically". A
second held-out model of a *different family* (e.g. Claude or Qwen) would make
the generalisation claim much stronger, at low marginal cost since only dev/test
need coverage.

**Add domains.** Prompt families are already keyed by `domain × primitive`.
News and QA text would test whether findings survive outside encyclopedic
register — and, critically, whether the "cleaner text = AI" confound weakens on
better-edited human sources.

**Vary decoding.** Everything was generated at `temperature = 0.8, top_p = 0.95`.
Sweeping temperature would show whether detectability varies with sampling
entropy — a cheap and directly publishable ablation.

**Add adversarial conditions.** Prompting generators to *imitate* the source
article's style, or paraphrasing AI output through a second model, would test
robustness against the realistic case where an author actively hides the seam.

**Human-edited AI spans.** Real mixed documents are usually AI text lightly
edited by a human. A condition where AI spans receive small human-like
perturbations would be closer to the deployment distribution.

**Balance the type mix by difficulty.** Type 1 is close to saturated at 0.877.
Shifting the mix toward Types 2 and 3 would concentrate annotation budget where
headroom remains.

**Fix the length shortfall at the prompt level.** Length is 55% of all
rejections and costs ~1.6× API calls per accepted document. Since all models
under-write, calibrating the requested word count upward per model (e.g. +8
words for GPT-4o) should raise the acceptance rate without loosening the gate.

### 15.2 Detection

**Retrain on the full corpus.** Highest expected value per unit effort, given
that dev F1 was still improving when training ended.

**Structured decoding over sentence labels.** A CRF or semi-Markov layer on the
sentence outputs would model run-length structure directly, instead of the
post-hoc smoothing that failed here — while permitting single-sentence spans,
which the naive smoother did not.

**Explicit boundary-pair classification.** Reframe the task from "label each
sentence" to "classify each adjacent pair as same-author or different-author".
This targets discontinuity directly and matches the evaluation metric.

**Larger and Sinhala-specialised encoders.** `xlm-roberta-large` is feasible on
6 GB with gradient checkpointing. Sinhala-adapted or continued-pretraining
variants would test how much of the gap is representational.

**Ablate the confounds.** Three specific experiments:
1. *Position-blind* — shuffle sentence order at inference and re-measure, to
   isolate how much of the score is positional.
2. *Noise-controlled* — normalise typographic noise in human sentences and
   re-measure, to quantify the "cleaner text = AI" shortcut.
3. *Length-controlled* — resample to equalise sentence-length distributions
   across classes.

**Cross-generator transfer matrix.** Train on each generator alone, test on
every other, producing a full transfer matrix. This would show whether
generators share a detectable "machine style" or each has an idiosyncratic
signature — directly relevant to the held-out gap in §13.3.

**Calibration and abstention.** Report expected calibration error and allow the
detector to abstain. For a realistic use case (flagging documents for review),
a well-calibrated confidence is worth more than raw F1.

**Human baseline.** An annotation study with fluent Sinhala readers on a test
subset would establish the ceiling and make the automatic numbers interpretable.

**Zero-shot LLM detectors.** Prompt a strong model to locate the boundary
directly, as a comparison point against the fine-tuned encoder — and as a check
on whether the task is solvable without task-specific training data.

---
