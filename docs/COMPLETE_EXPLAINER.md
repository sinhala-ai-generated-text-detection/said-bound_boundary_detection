# Everything in This Project, Explained

A ground-up walkthrough of what was built, how each piece works mechanically,
and *why* each design choice was made rather than an alternative. Written to be
readable without looking at the code, and detailed enough that you could
reimplement it or defend it to a reviewer.

Five parts:

1. **[Dataset generation](#part-1--dataset-generation)** — turning Sinhala
   Wikipedia into labelled mixed-authorship documents
2. **[The detectors](#part-2--the-detectors)** — what each model does, what
   every score means, what "context" actually is
3. **[The generative-AI work](#part-3--the-generative-ai-component)** — using a
   language model's own probability estimates as a detection signal
4. **[Counterfactual twins](#part-4--counterfactual-twins)** — the finding that
   the detector invents boundaries in purely human text, and a training method
   that uses the dataset's own structure to address it
5. **[Threats to validity](#part-5--threats-to-validity)** — what the numbers
   do and do not show

Every number here is taken from a results file in `reports/`, laid out as:

| folder | contents |
|---|---|
| `reports/dataset/` | the dataset audit |
| `reports/generation/` | generator pilot, fluency assessment, cross-model comparison, screening evidence |
| `reports/detection/` | every detector result: linear, XLM-R, likelihood, twin runs, operating-point analysis |

---

# Part 1 — Dataset generation

## 1.1 What we are actually building, and why it is hard

We want documents that are **partly human-written and partly machine-written**,
where we know *exactly* which sentences are which.

You cannot collect this from the wild, because nobody labels which half of their
essay ChatGPT wrote. So it has to be constructed: take genuine human text, and
replace part of it with machine text.

Everything downstream depends on one property: **the label must be true by
construction**. If sentence 4 is labelled AI, it must be because we personally
replaced sentence 4 with model output — not because a heuristic guessed.

The task is harder than ordinary AI-text detection. Normally a detector asks
"was this document machine-written?" and can use the whole document as evidence.
Here most of the document *is* human, and the detector must find the exact seam.

## 1.2 The source material

`wikipedia.parquet` — 26,470 Sinhala Wikipedia articles, 173 MB, with columns
`id`, `url`, `title`, `raw_mediawiki` (raw wikitext), and `text` (a pre-stripped
plaintext version).

The snapshot was taken in 2022, **before the public release of ChatGPT**, so
the human side of the dataset is human-written rather than possibly
LLM-assisted.

**Decision: clean from `raw_mediawiki`, not `text`.**

The provided `text` column looks convenient but is unreliable. Inspecting it
showed it still contains:

- literal `Category:සිංහල` lines left in the prose
- unexpanded template calls like `{{#ifexpr:>150|...}}`
- magic words like `__NOTOC____NOEDITSECTION__`
- list bullets flattened into leading spaces

That last one is the dangerous one: a flattened bullet list looks like prose to
a sentence splitter, so it would silently produce garbage "sentences" and every
label index after it would be wrong. Cleaning from raw wikitext is more work but
gives exact control over what is removed.

## 1.3 Cleaning: the filter cascade

Two different jobs, often confused:

- **Page-type filtering** — is this page even an article? (a disambiguation
  page is not prose)
- **Markup stripping** — removing wiki syntax from a page we are keeping

### Page-type filters, in order

| # | Filter | Removed | Why |
|---|---|---|---|
| 1 | main page (`මුල් පිටුව`) | 1 | a portal, not an article |
| 2 | namespace prefix (`Talk:`, `සාමාජික:`, `Draft:`) | 7 | not article namespace |
| 3 | redirects | 6 | no content of their own |
| 4 | disambiguation (`{{බහුරුත්හරණය}}`) | 166 | a list of links, not prose |
| 5 | stubs (`{{දියුණු කරන්න}}`) | 1,672 | too short to window |
| 6 | list pages by title | 383 | enumerations, not prose |
| 7 | list pages by content | 2,375 | same, detected structurally |
| 8 | too few characters | 8,168 | prose gate |
| 9 | fewer than 120 words | 3,100 | prose gate |
| 10 | not predominantly Sinhala | 428 | some pages are mostly English |
| 11 | fewer than 8 sentences | 774 | cannot support Type 3 |

**26,470 → 9,390 articles survive (35.5%).**

The prose gate (filters 8–11) does most of the work, removing 12,470 pages.
That is not a bug — it reflects Sinhala Wikipedia's size profile, where the
median raw article is only ~2,048 characters. We need articles long enough to
cut a 6–12 sentence window out of.

A useful sanity finding: namespace filtering is nearly a no-op. Only 43 of
26,470 titles carry any `X:` prefix, and most are false positives (article
titles that happen to contain a colon, like `බාහුබලි 2: ...`). The dump was
already essentially main-namespace. This was **verified before relying on it**
rather than assumed.

### Markup stripping, in order

Order matters, because later rules would otherwise operate on debris left by
earlier ones:

1. HTML comments `<!-- ... -->`
2. `<ref>...</ref>` citation blocks
3. tables `{| ... |}`
4. templates `{{...}}` — nesting-aware, since templates contain templates
5. file/image links `[[File:...]]`, `[[ගොනු:...]]`, `[[රූපය:...]]`
6. category links `[[Category:...]]`, `[[ප්‍රවර්ගය:...]]`
7. interwiki links
8. `[[link|text]] → text`, and `[[link]] → link`
9. `'''bold'''` and `''italic''` markers
10. whitespace normalisation

### The paragraph-reflow subtlety

This one caused a real bug and is worth understanding.

In MediaWiki, a **single newline inside a paragraph is a soft wrap**, not a
paragraph break. Only a *blank* line starts a new paragraph. An editor may write:

```
ගණිතය බොහෝ විට අර්ථ දැක්වෙනුයේ
ප්‍රමාණය, ව්‍යුහය සහ අවකාශය අධ්‍යයනය කිරීම ලෙසය.
```

That is **one sentence**, wrapped across two lines. Treating each newline as a
break splits it into two fragments mid-clause, and every sentence index after it
shifts. `_reflow()` implements the correct semantics: join single newlines,
break only on blank lines.

## 1.4 Sentence segmentation

Everything in this project is indexed by sentence, so segmentation is the
foundation. **One deterministic segmenter is used everywhere** — preparing
sources, setting generation length targets, assigning labels, and evaluating. If
any stage used different logic, sentences and labels would silently misalign.

Splits on `.`, `?`, `!`, `…` and `෴` (kunddaliya, the Sinhala full stop), with
guards for:

- **decimals** — `3.14` must not split
- **abbreviations** — `ආචාර්ය.` (Dr.), `මහාචාර්ය.` (Prof.), `පෙ.ව.` (a.m.),
  `ප.ව.` (p.m.), plus Latin `Dr.`, `Mr.`, `etc.`
- **initials** — `A. B. Smith`

> ### A bug worth knowing about
>
> The initials guard originally treated *any single letter before a period* as
> an initial and refused to split there. In Sinhala, **`ය.` is an extremely
> common sentence ending** (`ය` is a frequent sentence-final particle). So the
> guard was silently merging huge numbers of real sentences, under-counting
> every document and corrupting length targets throughout the pipeline.
>
> The fix: apply the initials guard to **ASCII letters only**. This is now
> covered by regression tests. It is a good example of a rule that is correct
> for English and actively harmful for Sinhala.

## 1.5 Windowing

Rather than use whole articles, each article is reduced to a **contiguous window
of 6–12 sentences**.

Why:

- **Bounded generation cost.** Asking a model to continue a 200-sentence article
  is expensive and the output drifts.
- **Comparable documents.** A detector evaluated on documents ranging from 6 to
  600 sentences produces metrics dominated by length, not authorship.
- **More documents from the same corpus.** 9,390 articles yield 7,944 windows.

The window is *contiguous* — a real passage, not sampled sentences — so the
human text reads naturally. Every window stores its `source_id` so all
derivatives trace back to one article. That matters for the split (§1.6).

## 1.6 The train/dev/test split — and the leak it prevents

Splits are assigned **over source article IDs, before any generation happens**,
seeded with 42.

| split | source IDs | windows |
|---|---|---|
| train | 6,573 (70%) | 5,544 |
| dev | 1,408 (15%) | 1,204 |
| test | 1,409 (15%) | 1,196 |

**Why split before generating, and by source rather than by document?**

One article can produce several documents — different construction types,
different generators. If those landed in different splits, the *same human
sentences* would appear in both training and test. A detector could then
memorise "this passage about Madagascar is human" instead of learning what human
Sinhala looks like, and the test score would be inflated by pure memorisation.

Assigning by `source_id` makes that impossible by construction. Verified on the
built dataset: **0 sources appear in more than one split.**

## 1.7 The generators, and the two roles

| model | slug (pinned) | role | writes |
|---|---|---|---|
| DeepSeek V3 | `deepseek/deepseek-chat` | **seen** | train, dev, test |
| GPT-4o | `openai/gpt-4o-2024-11-20` | **seen** | train, dev, test |
| Gemini 2.5 Pro | `google/gemini-2.5-pro` | **held-out** | dev, test only |

**Why a held-out generator?** This is the most important experimental design
decision in the project.

A detector trained on DeepSeek and GPT-4o might learn *"machine-written text"*,
or it might only learn *"DeepSeek's and GPT-4o's particular habits"*. Those look
identical on a normal test set. The only way to tell them apart is to test on a
generator the detector has never seen. Gemini is barred from `train` entirely,
so its test score is an honest measure of generalisation to an unseen model.

Model slugs are **pinned to dated versions where available** and prices were
verified against the live OpenRouter model list at runtime, because slugs and
prices drift.

### Mistral Nemo was tested and rejected

The original plan had Mistral Nemo as the second seen generator. It was piloted
and dropped. Its validation pass rate was 9.2% against 55–67% for the others,
but the pass rate understates the problem. Its output showed:

1. **Degenerate word-salad** — non-words like `මාත්‍රිකාවකිනීම්` in
   grammatically shaped but meaningless sentences
2. **Prompt instructions leaking into the output as content** — e.g.
   `වාක්‍ය 18 සිට 30 අතර විය යුතුය` ("sentences must be between 18 and 30").
   One output was just `මම පිළිපීඩා කරමි` ("I comply").
3. **Repetition loops** — `පත්‍රිකාවක් පත්‍රිකාවක් ගියේය`
4. **Meta-commentary as content** — "Jamaica's government and politics is a
   Wikipedia article"

The decisive point: **the 7 documents it *did* pass validation were also
unusable** — same defects. They passed only by landing on the right sentence and
word counts.

**This exposes a real limitation you should know about:** none of the automatic
checks test *meaning*. The script check measures what fraction of characters are
in the Sinhala Unicode block, so text made of Sinhala characters that is not
Sinhala *language* scores 1.0 and passes. Format checks and quality checks are
different things. Any new generator must be read by a human before it is
trusted.

GPT-4o replaced it. Evidence archived in `reports/generation/evidence/`.

### Comparing the three generators on identical inputs

To compare generators fairly, 20 matched examples were produced in which
**every model rewrote the same span of the same window under the same plan**.
(Normally the span-selection RNG is seeded per generator, which would give each
model a different span.) They were drawn from dev/test so the held-out
generator was permitted. Full text in `reports/generation/model_comparison.md`.

| model | produced | mean Δ words | mean abs Δ words | mean Δ sentences | mean retries |
|---|---|---|---|---|---|
| DeepSeek V3 | 17/20 | −3.1 | 5.5 | +0.00 | 0.47 |
| GPT-4o | 16/20 | −7.9 | 8.1 | +0.00 | 0.25 |
| Gemini 2.5 Pro | 18/20 | −4.0 | 6.2 | +0.00 | 0.11 |

All three hit the requested **sentence** count exactly, because that check is
binding, while under-shooting the **word** count. GPT-4o under-writes most, and
Gemini needs the fewest retries. One matched example failed for all three: a
7-sentence, 196-word target. Long targets are where every model under-generates,
so that failure is systematic rather than model-specific.

Qualitatively, all three write fluent Sinhala. DeepSeek V3 writes natural
encyclopedic prose whose sentences run longer than the originals. GPT-4o
follows instructions most closely and renders technical terms in Sinhala
(`කුබිට්ස්`, qubits) rather than falling back to English. Gemini 2.5 Pro
produces the most natural Sinhala, with correct honorifics (`පියතුමා`,
`මහත්මිය`) and accurate Sri Lankan place names.

## 1.8 The three construction types

Two primitives only: **continuation** and **span replacement**. Critically, we
*replace* existing human text rather than *inserting* invented sentences — so
the surrounding human text stays genuine and the machine text is constrained to
express roughly what the human text expressed.

### Type 1 — single boundary (continuation)

```
[human][human][human] | [AI][AI][AI]
labels: 0  0  0  1  1  1        boundary at index 3
```

A human prefix, then the model writes a continuation. The true human remainder
is **discarded** — the model is not paraphrasing it, it is writing a new tail
given the prefix, with the removed portion's sentence and word counts as
targets.

- Split point sampled uniformly in **20–80%** of the passage, sentence-aligned
- At least 2 human and 2 AI sentences enforced
- Eligibility: ≥ 4 sentences
- Realised: 1,675 documents, exactly 1.00 boundaries, 49.4% AI sentences

**Why 20–80% rather than the midpoint?** If the boundary were always in the
middle, a detector could score well by always guessing the middle, learning
nothing about language. Randomising the position removes that shortcut. It
cannot be fully uniform, though, because we require ≥2 sentences on each side —
which is exactly why we measure the residual positional prior with a baseline
(§2.4).

### Type 2 — one internal AI segment (span replacement)

```
[human][human][AI][human][human]
labels: 0  0  1  0  0        boundaries at 2 and 3
```

An internal span of 1–3 sentences is rewritten. The model receives
`LEFT + TARGET + RIGHT` and is told to **rewrite only TARGET**, preserving its
facts, using LEFT/RIGHT for flow but not copying them.

- Span must not touch sentence 0 or the final sentence
- AI proportion constrained to 15–35% of the document
- ≥2 human sentences preferred on each side
- Eligibility: ≥ 6 sentences
- Realised: 1,367 documents, exactly 2.00 boundaries, 24.5% AI

**Why can't the span touch the first or last sentence?** Because then it would
degenerate into a Type 1 (a prefix or suffix change), and the whole point of
Type 2 is that the machine text is *surrounded* by human text on both sides —
which is much harder to detect.

### Type 3 — multiple internal AI segments

```
[human][human][AI][human][AI][human][AI][human]
labels: 0  0  1  0  1  0  1  0        six boundaries
```

Two or three non-adjacent spans of 1–2 sentences each.

- No overlaps, ≥1 human sentence between spans, none at start or end
- Eligibility: ≥ 8 sentences
- Realised: 1,202 documents, 5.01 boundaries on average, 33.0% AI

**The most important detail here:** each span is generated **independently from
the ORIGINAL human document**. Span 2 is never conditioned on span 1's output.

Why this matters: if you generated the spans sequentially, feeding span 1's
output into the context for span 2, the model would build a self-consistent
machine register across the whole document. Real multi-edit documents don't work
that way — someone edits one paragraph today and another next week. Generating
independently keeps the spans stylistically uncorrelated, which is both more
realistic and harder.

All spans in a document use the **same generator**, so a document has one
machine author, not three.

**Documents are never padded into a higher type.** A 5-sentence window is simply
ineligible for Type 3 rather than being stretched.

**A property that turns out to matter later.** Because every construction
*replaces* sentences instead of inserting them, each mixed document has an
exactly aligned all-human counterpart: its source window. Part 4 builds on this.

### Worked examples

Real records from the dataset, with gold labels.

**Type 1 — `මැඩගස්කරයේ භූගෝලය` (Geography of Madagascar), DeepSeek V3.**
`labels = [0,0,0,0,1,1]`, boundary at index 4.

| # | label | sentence |
|---|---|---|
| 0 | HUMAN | මිනිසුන් මැඩගස්කරයට පැමිණීමෙන් පසු අවම වශයෙන් ලීමර් විශේෂ 17 ක් වඳ වී ගොස් ඇති අතර, ඒවා සියල්ලම ඉතිරිව ඇති ලීමර් විශේෂයට වඩා විශාල විය. |
| 1 | HUMAN | ෆොසා (බළලුන් වැනි) ඇතුළු තවත් ක්ෂීරපායින් ගණනාවක් මැඩගස්කරයට ආවේණික වේ. |
| 2 | HUMAN | දිවයිනේ කුරුලු විශේෂ 300කට අධික ප්‍රමාණයක් වාර්තා වී ඇති අතර ඉන් සියයට 60කට වැඩි ප්‍රමාණයක් ආවේණික වේ. |
| 3 | HUMAN | (එක් ආවේණික පවුලක් ඇතුළුව). |
| 4 | **AI** | මැඩගස්කරයේ සීනීන් ඇතුළුව උභයජීවීන්ගේ විශේෂ 370කට අධික ප්‍රමාණයක් වාර්තා වී ඇති අතර ඉන් සියයට 99කට වැඩි ප්‍රමාණයක් ආවේණික වේ. |
| 5 | **AI** | තවද මෙම දිවයිනේ කටුස්සා වැනි සත්ව විශේෂ ගණනාවක් පමණක් නොව සත්ව ගණයේ අනන්‍යතාවයන් දක්නට ලැබේ. |

Sentence 4 fluently continues the endemism statistics of sentence 2. The
boundary is not lexically obvious.

**Type 2 — `රෙකෝන්ඩය` (Recorder), GPT-4o.** `labels = [0,0,0,1,0,0]`,
boundaries at 3 and 4.

| # | label | sentence |
|---|---|---|
| 2 | HUMAN | මෙය දැවය භාණ්යක් වන අතර ඇගිලි ගණනක් සදහා සිදුරු සහිත මෙය මුඛ්‍ය හා සම්බන්ධ මත ස්ථානයේ වඩා පලල්වද ඉතා දුරට යන්නට සිහින්මය යන ආකාරයද නිපදවා ඇත. |
| 3 | **AI** | මධ්‍ය යුගයේ විශේෂයෙන්ම ජනප්‍රිය වූ රෙකෝඩරය, දහ අටවන ශතවර්ෂයෙන් පසු සංගීත උපකරණයකි, නමුත් ඊට පසුව එහි භාවිතය සන්සුන් වශයෙන් අඩු විය. |
| 4 | HUMAN | මධ්‍යතන යුගයේ මෙම සංගීත භාණ්ය කොතරම් ජනප්‍රියව කිබණේද යත් සාහිත්‍යමය කලා නිර්මාණ ආදියේ නිර්තන්තර එම සංගීත භාණ්ඩය පිළිබද සටහන් දැකිය හැකි විය. |

Here the **AI sentence is cleaner than the human sentences around it**:
Sinhala Wikipedia carries typographical noise (`භාණ්යක්`, `කිබණේද`). A
detector can therefore learn "cleaner text = AI". Part 5 measures this.

**Type 3 — `උගන්ඩාවේ භූගෝලය` (Geography of Uganda), DeepSeek V3.**
`labels = [0,0,1,0,1,0,1,0]`: three single-sentence AI spans, each isolated
between human sentences.

| # | label | sentence |
|---|---|---|
| 1 | HUMAN | රට සාමාන්‍යයෙන් මුහුදු මට්ටමේ සිට මීටර් 900 ක උසකින් පිහිටා ඇත. |
| 2 | **AI** | උගන්ඩාවේ නැගෙනහිර හා බටහිර දෙපසම කඳුකරයන් පිහිටා ඇත. |
| 3 | HUMAN | රුවෙන්සෝරි කඳුවැටිය උගන්ඩාවේ උසම කඳු මුදුන අඩංගු වන අතර … මීටර් 5,094 කි. |
| 4 | **AI** | රටේ දකුණු කොටසෙහි විශාල ප්‍රදේශයක් … වික්ටෝරියා විලේ බලපෑමට යටත්ව ඇති අතර එහි බොහෝ දූපත් දක්නට ලැබේ. |
| 5 | HUMAN | කම්පාලා අගනුවර සහ එන්ටෙබේ අගනුවර ඇතුළුව මෙම වැව ආසන්නයේ වඩාත් වැදගත් නගර දකුණේ පිහිටා ඇත. |
| 6 | **AI** | රටේ මධ්‍යයේ පිහිටි කියෝගා විල පුළුල් වගුරු බිම් වලින් යුක්ත වේ. |

Alternating single-sentence spans are the hardest configuration in the
dataset, and they are why run-length smoothing fails (§2.5).

## 1.9 Prompting

Prompts live in `prompts/task2_prompts.yaml`, organised by `domain × primitive`,
and are **written in Sinhala**, not English.

**Why Sinhala prompts?** English instructions that ask for Sinhala output tend
to induce *translationese* — Sinhala with English sentence structure. That would
be an artifact the detector could latch onto, and it would not resemble how
Sinhala AI text actually appears in the wild.

Every prompt instructs the model to: write only in Sinhala; produce no markdown,
headings or lists; preserve facts (names, dates, numbers); not fabricate
unverifiable specifics; and return **only** the generated segment with no
preamble.

Length is controlled explicitly: each prompt carries a target sentence count and
a target word range derived from the span being replaced.

Generation settings: `temperature = 0.8`, `top_p = 0.95`, `max_tokens = 1200`.

**Reasoning suppression.** Gemini 2.5 Pro is a reasoning model. It was
configured with `reasoning: {effort: minimal, exclude: true}` and the client
reads only the message content, never a reasoning field. Verified: **zero of the
4,244 records contain reasoning-trace markers.**

## 1.10 Validation — 7 automatic checks

Every generation goes through: **generate → clean → segment → validate →
pass: keep / fail: retry (up to 3) / 3 failures: log and skip.**

Model output is **never hand-repaired**. Editing outputs would mean the dataset
contains text no model actually produced.

| # | Check | Rule |
|---|---|---|
| 1 | sentence count | must match the request exactly |
| 2 | length | within ±25% of the target word count |
| 3 | no preamble | rejects "Sure…", "Certainly…", "මෙන්න…" |
| 4 | no markdown | rejects `#`, `**bold**`, bullets, code fences |
| 5 | script | ≥60% Sinhala characters, ≤25% Latin |
| 6 | not a copy | normalised similarity against the original span |
| 7 | number consistency | numeric tokens must not mutate (2025 → 2027) |

### What actually gets rejected (5,109 rejections across the project)

| reason | count | share |
|---|---|---|
| length | 3,194 | 53.7% |
| sentence_count | 1,423 | 23.9% |
| numbers | 972 | 16.3% |
| script | 208 | 3.5% |
| copy | 139 | 2.3% |
| markdown | 15 | 0.3% |

**Length dominates, and we deliberately did not loosen it.** All three models
systematically under-write Sinhala relative to the human text they replace
(mean −3 to −8 words). Loosening the tolerance to ±35% would have rescued only
about 2 of 12 sampled failures, while admitting exactly the artifact the check
exists to prevent: if AI spans were systematically shorter than human spans, a
detector could "solve" the task by counting words instead of reading them.

The cost of keeping the gate strict is roughly 1.6 API calls per accepted
document. That is a price worth paying for a clean dataset.

**Retry distribution:** 3,260 documents accepted first try, 646 after one retry,
322 after two, 16 after three or more.

## 1.11 Engineering properties

- **Resumable.** Every record has a deterministic `record_id`. On re-run,
  completed records are skipped. This survived multiple interruptions.
- **Nested partial runs.** `--fraction 0.2` runs a deterministic *prefix* of the
  full plan, verified so that 20% ⊂ 50% ⊂ 100%. Running 20% now and 100% later
  extends rather than collides. (The full task list is always built and *then*
  sliced — otherwise a smaller run would pick different windows for Types 2/3.)
- **Every call logged** to `logs/cost.jsonl` with tokens and cost.
- **Rate-limited with exponential backoff** on 429/5xx.

### Two bugs found by running at scale

**HTTP 402 was swallowed.** When the OpenRouter account hit zero mid-run, the
orchestrator treated 402 like any other API error and kept going — burning
**1,964 guaranteed-failing calls** and hiding the real cause behind a generic
error count. Now 402 raises `InsufficientCredits` and aborts immediately.

**Concurrent log appends corrupted a line.** Worker threads appending to
`rejected.jsonl` interleaved and produced one torn line in ~2,800. Fixed with a
lock; the reader now skips an unparseable line with a warning instead of
crashing.

## 1.12 Final dataset

**4,244 documents · 39,358 labelled sentences · 2,630 source articles.**
0 integrity failures.

| generator | role | train | dev | test | total |
|---|---|---|---|---|---|
| DeepSeek V3 | seen | 1,186 | 273 | 302 | 1,761 |
| GPT-4o | seen | 1,304 | 283 | 311 | 1,898 |
| Gemini 2.5 Pro | held-out | **0** | 280 | 305 | 585 |
| **total** | | **2,490** | **836** | **918** | **4,244** |

**Class balance:** 14,318 of 39,358 sentences are machine-written (**36.4%**).

**Cost:** 11,587 API calls, 15.2M input tokens, 2.8M output tokens, **$26.73**.

Each record stores: `record_id`, `source_id`, `window_id`, `title`, `url`,
`domain`, `generator`, `generator_role`, `construction_type`, `split`, `text`,
`sentences[]`, `labels[]`, `boundaries[]`, `boundary_position_normalized`,
`spans`, `prompt_id`, `raw_output`, `cleaned_output`, `temperature`, `top_p`,
`retry_count`, `validation_passed`, `validation_errors[]`, `generated_at`.

Both `raw_output` and `cleaned_output` are kept so cleaning can be audited or
redone without regenerating.

---

# Part 2 — The detectors

## 2.1 The task, stated precisely

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

## 2.2 Every metric, and what it actually means

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

## 2.3 The evaluation protocol

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

## 2.4 The baselines, and why each one exists

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

## 2.5 The linear detector — and what "text + context" means

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

## 2.6 The transformer detector — how it actually works

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

## 2.7 Headline comparison

| model | F1 | **bound exact** | held-out F1 |
|---|---|---|---|
| position only (no text) | 0.567 | 0.284 | 0.550 |
| best linear | 0.678 | 0.438 | 0.571 |
| **XLM-RoBERTa** | **0.802** | **0.603** | **0.742** |

---

# Part 3 — The generative-AI component

## 3.1 The idea

A language model assigns a probability to every token. Text a model would
itself have produced is, by definition, **high-probability text** under that
model. Human text is more surprising.

So: **use a language model's own probability estimates as a detection feature.**
This is the family that DetectGPT, Fast-DetectGPT and especially **SeqXGPT**
belong to — SeqXGPT uses per-token log-probability lists for exactly this
sentence-level task. What is new here is applying it to a **low-resource
language where it has not been tested**.

## 3.2 How the scoring works mechanically

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

## 3.3 The 14 features per sentence

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

## 3.4 The probe — testing the premise before building on it

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

## 3.5 The main result — likelihood closes the generalisation gap

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

## 3.6 Where it did not work, and why

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
contribution moved to Part 4.

## 3.7 An engineering detail worth knowing

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

---

# Part 4 — Counterfactual twins

## 4.1 The problem nobody measures

Every document the detector is trained on contains at least one boundary, by
construction. So does every test document. A detector can therefore learn
**"there is always machine text somewhere, find the most machine-like
sentence"** and still score well, because the benchmark never shows it a
document with no machine text.

In real use, most documents a detector sees are entirely human. So the question
the mixed-document metrics cannot answer is: **what does the detector do on
purely human text?**

## 4.2 The twin: a free, exactly matched control

The construction in §1.8 *replaces* human sentences rather than inserting new
ones. So for every mixed document, its source window is the **same document
with every AI sentence swapped back for the human sentence it displaced**:

```
mixed:  [H0] [H1] [AI2] [H3] [H4]      labels 0 0 1 0 0
twin:   [H0] [H1] [H2 ] [H3] [H4]      labels 0 0 0 0 0
```

Same topic, same length, same positions. Only authorship differs, and only at
the replaced positions. This was verified, not assumed: all 4,244 documents
align sentence-for-sentence with their window in `sources/windows.jsonl`, and
the loader (`data.load_twins`) refuses to build a twin if any human sentence
does not match.

Many boundary-detection datasets cannot do this. Datasets built by insertion,
or by continuing a prompt with no human remainder kept, have no aligned human
counterpart.

## 4.3 What the baseline does on twins

The saved XLM-R tagger, re-scored with no retraining
(`reports/detection/xlmr_rescored.md`), threshold decoding:

| on the 578 unique test twins | overall | seen | held-out |
|---|---|---|---|
| twins with ≥1 sentence flagged AI | **95.3%** | 94.4% | 95.8% |
| boundaries predicted per twin (correct: 0) | **3.07** | 3.01 | 3.15 |
| sentence false-positive rate | **26.2%** | 26.0% | 26.5% |

Three things make this worse than it looks:

- **The false-positive rate is higher with no machine text present.** Human
  sentences inside mixed documents are flagged about 18% of the time. In pure
  human documents it is 26%. The detector searches for machine text and
  "finds" it.
- **31% of the human originals at replaced positions are flagged as AI**, even
  though they are the genuine human sentences. Part of what the model calls
  "machine" is position and topic, not authorship.
- **No threshold fixes it.** Sweeping the threshold from 0.05 to 0.95 (§4.7),
  the baseline never flags fewer than 93% of twins. Its scores are too extreme
  for any cut-off to separate them.

The Viterbi decoder is worse still: 99.5% of twins flagged, 4.55 boundaries per
twin.

A useful single number is **exact-boundary F1 over the mixed documents and the
twins together**, where any boundary predicted in a twin counts as a false
positive. That is closer to deployment than mixed-only scoring, and for the
baseline it drops from **0.603 to 0.434**.

## 4.4 The training method

With `--twins`, each mixed training document is batched **together with its
twin**, and the loss has three terms (`transformer.twin_loss`):

1. **Tagging loss on both documents.** Cross-entropy over the mixed document's
   labels and over the twin (all human). With no twins this is exactly the
   baseline objective.
2. **Margin term.** At every replaced position, the AI sentence's score must
   exceed the score of the human sentence it replaced by a margin (2.0 in logit
   space). Position and topic are identical across the pair, so neither can
   satisfy this term. Only authorship can.
3. **Consistency term.** For every sentence that is human in *both* documents,
   the two predictions should match (symmetric KL). Its author does not
   change, so its prediction should not depend on whether machine text
   appears elsewhere. This term targets the "there is always a boundary" prior
   directly.

`loss = CE + ramp × (λ_margin × margin + λ_consistency × consistency)`, with
both λ = 1.

### Details that matter

- **Shared windows are down-weighted.** One window can produce several records
  (different construction types or generators). In train, 856 windows back two
  records and 51 back three or four. Without correction their twins would count
  2–4 times, so each twin's loss is weighted by 1/k.
- **Pairs are matched per sentence, not per position.** A twin sentence can
  tokenize to a different length than the AI sentence it stands in for, so a
  long document may split into 512-token chunks at different places. The
  collate function (`collate_pairs`) records, for each sentence, where its
  marker landed in each document, and the loss compares those. A unit test
  forces the two documents to chunk differently and checks the mapping.
- **Memory is unchanged.** Batches hold 2 pairs (4 documents), the same as the
  baseline's 4 documents.

## 4.5 The first attempt collapsed, and why

With the paired terms on from the first step, the model **collapsed to a
constant output** (`reports/detection/twin_no_warmup.md`):

| epoch | CE | margin | consistency | dev boundary F1 |
|---|---|---|---|---|
| 1 | 0.670 | 1.991 | 0.020 | 0.453 |
| 2–7 | ~0.642 | ~2.000 | 0.002–0.005 | 0.444, frozen |

Reading the log:

- **Margin stuck at 2.00** means the score gap between an AI span and its human
  original was always zero. The margin loss equals the margin exactly when the
  two scores are identical.
- **Consistency near zero** was achieved the cheap way: a model that outputs
  the same score for every sentence satisfies it perfectly.
- On test, every sentence got nearly the same score (paired win rate 0.03),
  and the threshold decoder labelled everything AI.

This is the same trap described in §2.6: XLM-R spends its first epochs
predicting one class before it learns to separate them. Early on, the encoder
cannot tell a document from its twin, so the margin gradient cancels itself
out, while the consistency term actively *rewards* ignoring the input. The
paired terms held the model in the collapsed state it needed to escape.

**How that diagnosis was confirmed.** The `twin_plain` ablation uses the same
twins with both paired terms switched off. It trained normally (CE 0.556 →
0.039 over 8 epochs), so neither the extra human data nor the class balance
caused the collapse; the paired terms did.

**The fix: a warm-up.** `--twin-warmup 2` keeps the paired terms off for two
epochs, then ramps them in linearly over the third. With it, the model escapes
the single-class phase during epoch 2 (CE 0.645 → 0.443), and when the paired
terms arrive, training continues normally (margin 0.33 → 0.008 by epoch 8).

## 4.6 Results

Single seed, 8 epochs each, threshold decoding unless noted. Twin columns are
on the unique test twins.

| | baseline | twins only (`twin_plain`) | twins + paired terms (`twin_warm`) |
|---|---|---|---|
| twin false-alarm rate | 95.3% | 58.7% | **49.0%** |
| boundaries per twin | 3.07 | 1.72 | **1.19** |
| sentence FPR on twins | 26.2% | 15.7% | **10.0%** |
| context stability | 0.885 | 0.931 | **0.956** |
| paired win rate | 0.878 | 0.921 | 0.921 |
| exact-boundary F1, mixed + twins | 0.434 | 0.448 | **0.464** |
| exact-boundary F1, mixed only (Viterbi) | **0.603** | 0.595 | 0.580 |
| held-out exact-boundary F1 (Viterbi) | **0.548** | 0.527 | 0.524 |

*Context stability* is how often an untouched human sentence gets the same
label in the mixed document and in its twin. *Paired win rate* is how often an
AI sentence scores above the human sentence it replaced.

What this shows:

- **Adding twins at all fixes much of the problem.** Twins as plain extra human
  documents cut false alarms by 37 points.
- **The paired terms go further**: fewer false alarms (49% vs 59%), fewer
  hallucinated boundaries (1.19 vs 1.72), better context stability.
- **The consistency term does the work; the margin term adds nothing
  measurable.** The paired win rate is 0.921 with or without the paired terms.
  Plain cross-entropy on the twin already pushes the human original down.
- **There is a cost on the original benchmark.** Mixed-only exact-boundary F1
  falls, and the held-out generator falls more.

## 4.7 Operating points: is the cost real?

A single tuned threshold compares models at whatever point the selection rule
happened to pick. Here that rule saw only mixed dev documents, and the
threshold grid (0.20–0.80) cut off both the baseline (tuned to 0.80) and
`twin_plain` (tuned to 0.20). So `src/detect/twin_tradeoff.py` sweeps the
threshold from 0.05 to 0.95 for every saved model
(`reports/detection/twin_tradeoff.md`):

| model | twin false alarm, across all thresholds | best mixed-only exact-boundary F1 |
|---|---|---|
| baseline | 98% → 93% | **0.603** |
| twins only | 63% → 43% | 0.554 |
| twins + paired terms | **84% → 24%** | 0.554 |

- **The baseline cannot be made quiet on human text.** At every threshold it
  flags at least 93% of pure-human documents. This is the clearest finding of
  the whole study, and it is a property of the model, not of the tuning.
- **Twins alone lower the curve but flatten it.** False alarms bottom out at
  43%, and the threshold barely moves them.
- **Paired training is the only model with a usable operating range.** At
  threshold 0.925 it flags 28% of human documents while keeping 0.502 mixed-only
  exact-boundary F1. Neither other model can reach a 30% false-alarm rate at
  any threshold.
- **The mixed-only cost is real, not a tuning artifact.** Even at its best
  threshold, each twin model tops out at 0.554, against the baseline's 0.603.

Choosing each model's threshold on dev *mixed documents plus their twins*
(a deployment-aware rule) gives exact-boundary F1 over mixed + twins of 0.438
(baseline), 0.465 (twins only) and 0.464 (paired). On the held-out generator:
0.359, 0.364 and **0.381**.

## 4.8 Fine-tuning the trained baseline with twins

Training from scratch with twins costs about 5 points of mixed-only accuracy.
An alternative is to start from the already trained baseline and **fine-tune
it with twins** (`--init-from models/xlmr_tagger`, 3 epochs, learning rate
1e-5, same loss, no warm-up needed since the model is already past the
single-class phase). This is `twin_ft`.

That comparison needs a control. The baseline was still improving when its
training stopped, so any gain could simply come from training longer.
`xlmr_ft` gets the same 3 extra epochs at the same learning rate, on mixed
documents only.

| test set | baseline | control `xlmr_ft` | **`twin_ft`** |
|---|---|---|---|
| exact-boundary F1, mixed only (Viterbi) | 0.603 | 0.593 | **0.594** |
| held-out exact-boundary F1 (Viterbi) | 0.548 | 0.543 | **0.548** |
| sentence F1 (Viterbi) | 0.792 | 0.789 | **0.794** |
| twin false-alarm rate (threshold) | 95.3% | 96.7% | **58.8%** |
| boundaries per twin | 3.07 | 3.16 | **1.61** |
| sentence FPR on twins | 26.2% | 27.5% | **14.4%** |
| exact-boundary F1, mixed + twins | 0.434 | 0.421 | **0.472** |
| … on the held-out generator | 0.353 | 0.337 | **0.401** |

- **Against the fair control, twin fine-tuning costs nothing on mixed
  documents** with Viterbi decoding (0.594 vs 0.593), and matches the baseline
  on the held-out generator. With threshold decoding a small cost remains:
  its best mixed-only score across thresholds is 0.578, against 0.597 for the
  control.
- **Extra training alone does not help.** The control is slightly *worse* than
  the original baseline, and just as prone to false alarms (96.7%).
- **It gives the best deployment-view score of any model**: 0.481 exact-boundary
  F1 over mixed documents plus twins at its best dev-selected threshold, and
  0.408 on the held-out generator.
- **The trade-off: a narrower range.** Across thresholds, `twin_ft`'s false
  alarms move between 69% and 53%. It cannot reach the 24% of the
  from-scratch model.

So the two variants serve different needs. **Fine-tuning** keeps the
baseline's accuracy and roughly halves false alarms. **Training from
scratch** reaches much lower false alarms at a cost of about 5 points.

## 4.9 Where this stands

- **Established:** a detector trained only on mixed documents invents
  authorship changes in nearly every human document, and no threshold fixes
  it. More training on mixed documents does not fix it either. The twin-based
  evaluation that shows this costs nothing extra to build.
- **Strong result:** fine-tuning with twins cuts false alarms from 97% to 59%
  with no loss of mixed-document accuracy against a matched control.
- **Promising:** training from scratch with twins reaches a 24% false-alarm
  rate, at a cost of about 5 points. The consistency term is what matters.
- **Not yet established:** everything above is one seed. Small gaps
  (0.594 vs 0.593) cannot be claimed without at least three.

Next experiments: three seeds of `twin_ft` and its control; drop the margin
term and vary the consistency weight; fine-tune for longer or at a higher
consistency weight to push the false-alarm floor lower; select checkpoints on
dev mixed + twins rather than mixed only.

---

# Part 5 — Threats to validity

**Surface noise in the human text ("cleaner text = AI").** Sinhala Wikipedia
carries typographical noise that generator output lacks. Measured over all
39,358 sentences:

| feature | human sentences | machine sentences |
|---|---|---|
| space before punctuation | 3.55% | 0.01% |
| parentheses | 12.0% | 2.6% |
| no final punctuation | 3.33% | 0.17% |
| Latin characters | 12.1% | 5.6% |
| zero-width non-joiner | 0.52% | 0.01% |

Any sentence with one of the first three is almost certainly human, so a
detector can score partly on formatting. Unicode normalization is **not** the
issue: neither class contains non-NFC text in meaningful amounts. How much of
each detector's score comes from these cues has not yet been measured (for
example, by re-scoring after normalizing them on both sides).

**Positional prior.** Boundaries fall at 20–80% of a document, and spans never
touch the first or last sentence. A position-only baseline reaches 0.284
exact-boundary F1. That is measured and weak, but it is not zero, and the twin
results show the tagger does use position (§4.3).

**Length prior.** The ±25% length gate bounds AI spans, but all generators
still under-write by 3–8 words on average, so a residual length cue may remain.

**Selection protocol.** Epochs and thresholds are chosen on dev *mixed*
documents, which cannot see false alarms. The threshold grid in
`run_transformer.py` (0.20–0.80) was hit at an edge by two runs. Both issues
are quantified in §4.7, but the reported headline numbers still use the
original protocol.

**Single seed.** Every transformer result is one training run.

**Single domain, single held-out generator.** Wikipedia only. Generalisation is
measured against one unseen model, which cannot separate "generalises to
unseen models" from "generalises to Gemini".

**No human ceiling.** No annotation study establishes how well Sinhala readers
do on this task.

---

# Summary — what to take away

**Dataset.** 4,244 documents where the labels are true by construction, not
inferred. Three construction types of increasing difficulty, a held-out
generator that makes generalisation measurable, and splits assigned by source
article before generation so no human passage can appear on both sides.

**Detection.** Document-level encoding is the single biggest win — 0.438 → 0.603
exact-boundary F1 — because it can represent discontinuity, which a per-sentence
classifier structurally cannot. Continuation is much easier than embedded
rewriting. A gap to an unseen generator persists (0.628 → 0.556).

**Generative AI.** Language-model likelihood is a real, usable detection signal
in Sinhala (Cohen's *d* = 0.602), and it is **generator-agnostic** in a way
surface features are not — a 0.027 held-out gap against 0.153. Its one
limitation is that it must come from a *different* model than the detector, or
it is redundant.

**Counterfactual twins.** Because construction replaces rather than inserts,
every mixed document has an exactly aligned all-human twin. On those twins the
baseline flags machine text in 95% of documents, and no threshold brings that
below 93%. Fine-tuning the baseline on each document together with its twin
cuts false alarms from 97% to 59% with no loss of mixed-document accuracy
against a matched control. Training from scratch the same way reaches a 24%
false-alarm rate, at a cost of about 5 points. Single seed so far.

**Five things that were tested and reported as negatives**, because they are as
useful as the positives: run-length smoothing hurts (Type 3 spans are one
sentence by design), a learned pair head stalls training, the likelihood
discontinuity hypothesis is false in this data, paired twin losses switched on
from the first step collapse training (a warm-up fixes it), and the twin margin
term adds nothing over plain cross-entropy on the twin.
