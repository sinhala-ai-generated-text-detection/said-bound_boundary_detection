# Dataset construction

[README](../README.md) · [Dataset](dataset.md) · [Detectors](detectors.md) · [Likelihood](likelihood.md) · [Counterfactual twins](counterfactual-twins.md) · [Limitations](limitations.md) · [Reproducing](reproducing.md)

## What we are actually building, and why it is hard

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

## The source material

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

## Cleaning: the filter cascade

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

## Sentence segmentation

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

## Windowing

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
derivatives trace back to one article. That matters for the split ([The train/dev/test split — and the leak it prevents](#the-traindevtest-split--and-the-leak-it-prevents)).

## The train/dev/test split — and the leak it prevents

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

## The generators, and the two roles

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

GPT-4o replaced it. Evidence archived in `results/generation/evidence/`.

### Comparing the three generators on identical inputs

To compare generators fairly, 20 matched examples were produced in which
**every model rewrote the same span of the same window under the same plan**.
(Normally the span-selection RNG is seeded per generator, which would give each
model a different span.) They were drawn from dev/test so the held-out
generator was permitted. Full text in `results/generation/model_comparison.md`.

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

## The three construction types

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
([The baselines, and why each one exists](detectors.md#the-baselines-and-why-each-one-exists)).

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
- Realised: 1,202 documents, 4.97 boundaries on average, 33.0% AI

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
exactly aligned all-human counterpart: its source window. [Counterfactual twins](counterfactual-twins.md) builds on this.

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
detector can therefore learn "cleaner text = AI". [Limitations](limitations.md) measures this.

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
dataset, and they are why run-length smoothing fails ([The linear detector — and what "text + context" means](detectors.md#the-linear-detector--and-what-text--context-means)).

## Prompting

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

## Validation — 7 automatic checks

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

## Engineering properties

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

## Final dataset

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
