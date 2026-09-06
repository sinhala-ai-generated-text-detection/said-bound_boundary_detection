# Sinhala-ABD content for the SAID paper — what to paste where

Everything here is dataset-description material only. No detector results.

## Files

| file | what it is |
|---|---|
| `said_abd_sections.tex` | 8 numbered blocks to paste into `main.tex` |
| `said_abd_prompts.tex` | prompts appendix — `\input` it |
| `figures/abd_fig1_constructions.pdf` | the three construction types |
| `figures/abd_fig2_positions.pdf` | boundary positions + machine share |

## Integration steps

1. **Table 1** — replace `tab:overview` with block **[1]**. Removes "counts to
   be finalised".
2. **Construction §3** — replace the whole `\paragraph{Sinhala-ABD.}` including
   its `\ph{}` with block **[2]**.
3. Insert block **[3]** (`tab:abd-composition`) and block **[4]** (both figures)
   after that paragraph.
4. **Validation §4** — append blocks **[5]** (two paragraphs) after the existing
   human-baseline paragraph. One `\ph{}` remains, deliberately: the
   native-speaker study does not yet cover ABD spans.
5. **Limitations §6** — merge the three commented sentences in block **[6]**.
6. **Appendix C** — replace the ABD half of the `\ph{}` with block **[7]**.
7. Add block **[8]** as a new appendix (`app:abd-cleaning`), referenced from
   block [2].
8. Add to the preamble (Sinhala does not render in TeX Gyre Termes):
   ```latex
   \newfontfamily\sinhalafont{Noto Serif Sinhala}[Script=Sinhala, Scale=0.95]
   \newcommand{\si}[1]{{\sinhalafont #1}}
   ```
   Blocks [7] and [8] use `\si{...}`. Swap the font name for one installed on
   your build machine (`Iskoola Pota` on Windows).
9. Appendix: `\input{said_abd_prompts}` after `said_user_prompts`.
10. **No bibliography changes.** The ABD sections cite nothing, so there is
    nothing to add to `\bibliography{...}`. An earlier draft of this bundle
    shipped a `said_abd_refs.bib`; it was removed because three of its four
    entries duplicated keys already in your `.bib` files, which would cause
    BibTeX repeated-entry errors for no benefit.

## Length

Roughly **1 page** of body text (2 paragraphs + 2 tables + 2 figures) and
**2.5 pages** of appendix. If the body is tight, the figures are the flexible
part: Figure 2 can be cut with its content folded into one sentence of block
[2], but Figure 1 is worth keeping — it explains the three types faster than
the prose does.

## Numbers, verified against the released corpus

| | |
|---|---|
| documents | 4,244 |
| labelled sentences | 39,358 |
| machine sentences | 14,318 (36.4%) |
| source articles | 2,630 |
| authorship boundaries | 10,379 (2.45/doc) |
| continuation / single-span / multi-span | 1,675 / 1,367 / 1,202 |
| GPT-4o / DeepSeek V3 / Gemini 2.5 Pro | 1,898 / 1,761 / 585 |
| rejected generations | 5,109 (5,951 individual check failures) |
| accepted on first attempt | 76.8% |

All prompts in `said_abd_prompts.tex` were checked programmatically against
`prompts/task2_prompts.yaml` — all six blocks are byte-identical.

---

## Three things to decide before submitting

### 1. The Wikipedia date cutoff — this one affects a claim in your abstract

Your abstract says *"Human texts predate the release of ChatGPT"*, and §3 says
the Wikipedia portion of Sinhala-HAT is taken **as it stood before 1 January
2015**.

**Sinhala-ABD has no such cutoff.** It was built from the current
`wikipedia-monthly` snapshot, which ships no revision timestamp — the columns
are `id`, `url`, `title`, `raw_mediawiki`, `text` only, so the filter could not
have been applied. The human sentences in ABD therefore cannot be guaranteed to
predate widespread LLM availability.

Options:

- **Re-derive ABD sources from the same pre-2015 revision snapshot used for
  HAT.** Cleanest, and keeps the abstract claim true for both datasets, but it
  means regenerating the corpus (~$27 and about a day).
- **Scope the claim.** Change the abstract to say human texts for
  *Sinhala-HAT* predate ChatGPT, and state the ABD limitation explicitly. The
  limitation sentence is written for you in block [6].

I would not leave the claim as-is covering both datasets.

### 2. Generator counts are uneven, and deliberately so

Gemini 2.5 Pro has 585 documents against 1,898 and 1,761. This is not an
accident of collection: Gemini was generated only for a disjoint subset of
source articles so that it can serve as a **held-out generator**, which is what
makes leave-one-generator-out evaluation possible without a user having to
construct it.

Since §5 says you release no fixed splits, this needs stating plainly — a reader
building their own splits will otherwise find Gemini absent from most sources
and think it a defect. Block [3]'s caption says it; block [6] adds the sample-size
caveat.

### 3. The English constraint block in the ABD prompts

The ABD prompts append a short English block after the Sinhala instructions,
whereas your HAT prompts are Sinhala-only, and §3 argues for native prompting
specifically to avoid translation artifacts.

This is defensible — the English block constrains *format*, not content, and the
output-language requirement is stated in both languages — but a reviewer will
notice the inconsistency if it is not addressed. The prompts appendix explains
it in its opening paragraphs. Read that and decide whether you are comfortable
with the argument, since it is a genuine deviation from the paper's stated
principle.
