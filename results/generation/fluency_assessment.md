Judgements below are from reading the generated Sinhala directly, not from
the automatic checks. This matters because **the validator cannot detect
fluency**: its script check only measures what fraction of letters are in the
Sinhala block, so text made of Sinhala characters that is not Sinhala
*language* passes it with `sinhala_ratio = 1.0`.

### DeepSeek V3 — usable

Fluent, natural encyclopedic Sinhala. Register matches Wikipedia, facts carried
over from context are handled correctly, and honorifics and grammatical endings
are used properly. Sentences are longer than the human originals on average,
which is why `length` is its most common rejection, but the prose itself is
sound. No instruction leakage, no repetition degeneration observed.

**Verdict: keep as a seen generator.**

### GPT-4o — usable, best format compliance

Fluent and coherent, with the strongest instruction-following of the three.
Technical vocabulary is handled well: in the *ක්වොන්ටම් පරිගණනය* (quantum
computing) sample it correctly produced `කුබිට්ස්` (qubits) and
`සුපර්පොසිෂන්` (superposition) rather than falling back to English or inventing
calques. Register is consistently encyclopedic. Highest call-level pass rate of
the seen generators.

**Verdict: keep as a seen generator — the substitute for Mistral Nemo.**

### Gemini 2.5 Pro — usable, highest quality

The most natural Sinhala of the three. Correct honorifics (`පියතුමා`, `මහතා`,
`මහත්මිය`), accurate Sri Lankan place names and historical detail, and
idiomatic connectives. As a reasoning model it was configured with
`reasoning: {effort: minimal, exclude: true}`; **zero of the 116 accepted
records contain reasoning-trace markers**, so the suppression plus
content-only parsing worked.

**Verdict: keep as the held-out generator (dev/test only).**

### Mistral Nemo — REJECTED, removed from the pipeline

Mistral Nemo's Sinhala is not usable, and the failure is qualitative rather
than a matter of tuning. Its call-level pass rate was **9.2%** against
54–67% for the others, but the pass rate understates the problem.

Observed failure modes:

1. **Degenerate word-salad.** Malformed, non-existent word forms
   (`මාත්‍රිකාවකිනීම්`, `කැමින්නටුවක්`) assembled into sentences with no
   recoverable meaning.
2. **Prompt-instruction leakage into the output text.** Generations contained
   my own instructions rendered as content — e.g. `වාක්‍ය 18 සිට 30 අතර විය
   යුතුය` ("sentences must be between 18 and 30") and `පෙළ පමණක් ලබා දෙන්න`
   ("provide only the text"). One output was simply `මම පිළිපීඩා කරමි`
   ("I comply").
3. **Repetition degeneration** — `පත්‍රිකාවක් පත්‍රිකාවක් ගියේය`.
4. **Meta-commentary as content** — `ජැමෙයිකාවේ රජය සහ දේශපාලනයක් විකිපීඩියා
   ලිපියකි` ("Jamaica's government and politics is a Wikipedia article").

The decisive point: **the 7 documents Mistral Nemo did get accepted are also
unusable.** Failure modes 2, 3 and 4 all appear in accepted output. They passed
only because they happened to land on the right sentence and word counts, and
because every check that could have caught them is blind to meaning. Training a
boundary detector on them would teach it to spot degenerate text, not an
authorial change.

Those 7 records have been removed from the dataset and archived to
`results/generation/evidence/mistral_nemo_pilot_records.jsonl`, with the 128 rejections in
`results/generation/evidence/mistral_nemo_rejections.jsonl`.

**Verdict: dropped. Replaced by GPT-4o (`openai/gpt-4o-2024-11-20`), which was
probed on the same prompts and passes both the automatic checks and reading.**

### Consequence for the validator

Mistral Nemo exposed a real gap: nothing in the 7 checks tests whether the
output *means* anything. That gap is harmless for the three retained
generators, all of which produce genuine Sinhala, but it means the validator
must not be trusted as a fluency gate if a new generator is added later. Any
future generator should be read before it is trusted, exactly as done here.
