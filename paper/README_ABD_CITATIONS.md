# Citations added to the Sinhala-ABD sections

12 citations across the construction and validation material, comparable to the
HAT sections. **Every key is already in your `.bib` files** — nothing new to
add, no duplicate-key risk.

## What each citation is doing

Check each against the paper you actually have. I am confident about the role
each plays, but you know your bib entries better than I do.

| Citation | Placed at | Claim it supports |
|---|---|---|
| `wang2024semeval` | sentence-level is stricter than document-level | SemEval-2024 Task 8 Subtask C established boundary location as a task distinct from document classification |
| `zeng2024aaai` | same sentence, and the replacement types | sentence-level detection inside hybrid essays |
| `gao2024llm` | per-sentence release; replacement types | mixed human/machine documents as the object of study |
| `zeng2024ijcai` | per-sentence release | hybrid-essay setting; you already cite it in Limitations for constructed vs genuine boundaries |
| `dugan2022roft` | **continuation** | RoFT is the origin of the human-prefix → machine-continuation setting. The single most load-bearing citation in these sections |
| `kushnareva2024ai` | continuation | boundary detection studied as a task on RoFT data |
| `wang2024m4` | replacement prevents topic separating classes; three generators | M4's topic-controlled parallel design, which you already invoke in §3 |
| `khairallah2025alhd` | three generators | multi-generator design, same pairing you use in HAT Generation |
| `doughman2024limitations` | length matching (×2, construction and validation) | length as a shortcut feature — the exact use you already make of it in HAT |
| `macko2023multitude` | held-out generator | detection degrades across unseen generators, which is why holding one out matters |
| `lucas2026bluff` | held-out generator | same, and it is your own motivating citation for low-resource degradation |
| `mcgovern2024fingerprints` | symmetric cleaning in validation | surface form carries authorship signal, so cleaning must be minimal — mirrors your HAT Cleaning paragraph |

## Structural notes

**`dugan2022roft` and `zeng2024aaai` now appear in both §2 and §3.** That is
normal and desirable — §2 situates them in the literature, §3 says which design
decision each one motivated. It is what the HAT sections already do with
`wang2024m4` and `doughman2024limitations`.

**I added a third paragraph, `\paragraph{Design constraints.}`** Your two-heading
structure left the design rationale (why 20–80%, why not the first/last
sentence, why replacement rather than insertion, why a held-out generator)
without a home, and that is exactly the material that needs citing. Fold it back
into the second paragraph if you prefer two headings.

**Appendix cross-references restored.** Your version dropped
`(Appendix~\ref{app:abd-cleaning})` from the first paragraph and the
`Appendix~\ref{app:abd-prompts}` sentence at the end. Both appendices exist in
your `main.tex`, so without these the appendices are never referenced from the
body.

## What I deliberately did not cite

- **The seven validation checks.** These are our own construction, not drawn
  from prior work. Citing something here would be decorative.
- **The generator-screening paragraph** (Mistral Nemo rejection). It is an
  empirical observation about our own pilot.
- **The source-filtering appendix.** Mechanical description of what the code
  does.

Padding these with citations would make the section look better-grounded than
it is. The HAT sections cite heavily because they make design claims that echo
specific prior work; the ABD sections should cite where that is true and not
elsewhere.

## One caveat on `kushnareva2024ai`

Your bib entry carries `note = {Verify venue and page numbers before
submission.}` from my earlier draft. Resolve that note before submitting, or
drop the citation — it is the least essential of the twelve, since
`dugan2022roft` already covers the continuation setting.
