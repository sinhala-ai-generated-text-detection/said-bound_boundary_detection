"""The 7 automatic validation checks.

Pipeline contract: generate -> clean -> segment -> validate ->
PASS to dataset / FAIL retry / 3 failures log+skip. Model output is NEVER
hand-repaired; `clean_output` only strips packaging the model added around its
answer (fences, wrapping quotes), it does not fix the Sinhala.

Checks:
  1 sentence_count  - matches the requested count exactly
  2 length          - within +/-25% of the target word count
  3 no_preamble     - no "Sure...", "මෙන්න...", "Certainly..."
  4 no_markdown     - no headings, bullets, bold, fences, links
  5 sinhala_script  - predominantly Sinhala (Latin proper nouns tolerated)
  6 not_a_copy      - not a normalised copy of the original span
  7 numbers_consistent - numeric tokens must not mutate (2025 -> 2027)
"""
from __future__ import annotations

import re
import sys
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segment import latin_ratio, segment_sentences, sinhala_ratio, word_count  # noqa: E402

# Wrapping the model sometimes adds around an otherwise-fine answer.
_CODE_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*\n?|\n?```\s*$")
_WRAPPING_QUOTES = ('"', "'", "“", "”", "«", "»")

# Sinhala + ASCII digits; used by the number-consistency check.
_NUM_RE = re.compile(r"\d[\d,.٫٬]*")


@dataclass
class ValidationResult:
    passed: bool
    errors: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.passed


def clean_output(raw: str) -> str:
    """Strip packaging around the answer. Does not alter the Sinhala itself."""
    t = (raw or "").strip()
    t = _CODE_FENCE.sub("", t).strip()
    # Some models wrap the whole answer in quotes.
    for q in _WRAPPING_QUOTES:
        if len(t) > 2 and t.startswith(q) and t.endswith(q):
            t = t[1:-1].strip()
            break
    # A single leading '---' separator echoed from the prompt.
    t = re.sub(r"^\s*-{3,}\s*\n", "", t)
    t = re.sub(r"\n\s*-{3,}\s*$", "", t)
    t = re.sub(r"[ \t ]+", " ", t)
    t = re.sub(r"\n{2,}", "\n\n", t)
    return t.strip()


def _normalise_for_compare(text: str) -> str:
    """NFC, lowercase, strip punctuation/whitespace - for copy detection."""
    t = unicodedata.normalize("NFC", text or "").lower()
    t = "".join(ch for ch in t if not unicodedata.category(ch).startswith("P"))
    return re.sub(r"\s+", " ", t).strip()


def _numbers(text: str) -> list[str]:
    """Numeric tokens, normalised so 1,250 and 1250 compare equal."""
    out = []
    for m in _NUM_RE.findall(text or ""):
        tok = m.rstrip(".,").replace(",", "")
        if tok:
            out.append(tok)
    return out


class Validator:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        v = cfg["validate"]
        self.v = v
        self.preamble = [re.compile(p) for p in v["preamble_patterns"]]
        self.markdown = [re.compile(p) for p in v["markdown_patterns"]]
        self.tol = v["length_tolerance"]
        self.sent_tol = v["sentence_count_tolerance"]
        self.min_sinhala = v["min_sinhala_char_ratio"]
        self.max_latin = v["max_latin_char_ratio"]
        self.max_copy = v["max_copy_similarity"]

    # -- individual checks ------------------------------------------------
    def check_sentence_count(self, sents: list[str], target: int) -> str | None:
        if abs(len(sents) - target) > self.sent_tol:
            return f"sentence_count: got {len(sents)}, expected {target}"
        return None

    def check_length(self, text: str, target_words: int) -> str | None:
        n = word_count(text)
        lo = target_words * (1 - self.tol)
        hi = target_words * (1 + self.tol)
        if not (lo <= n <= hi):
            return (f"length: {n} words outside "
                    f"{lo:.0f}-{hi:.0f} (target {target_words})")
        return None

    def check_no_preamble(self, text: str) -> str | None:
        for p in self.preamble:
            if p.search(text):
                return f"preamble: matched {p.pattern!r}"
        return None

    def check_no_markdown(self, text: str) -> str | None:
        for p in self.markdown:
            if p.search(text):
                return f"markdown: matched {p.pattern!r}"
        return None

    def check_sinhala_script(self, text: str) -> str | None:
        sr = sinhala_ratio(text)
        lr = latin_ratio(text)
        if sr < self.min_sinhala:
            return f"script: sinhala_ratio {sr:.2f} < {self.min_sinhala}"
        if lr > self.max_latin:
            return f"script: latin_ratio {lr:.2f} > {self.max_latin}"
        return None

    def check_not_a_copy(self, text: str, original: str | None) -> str | None:
        if not original:
            return None
        a, b = _normalise_for_compare(text), _normalise_for_compare(original)
        if not a or not b:
            return None
        if a == b:
            return "copy: output identical to original span"
        sim = SequenceMatcher(None, a, b).ratio()
        if sim > self.max_copy:
            return f"copy: similarity {sim:.3f} > {self.max_copy}"
        return None

    def check_numbers_consistent(self, text: str,
                                 original: str | None) -> str | None:
        """Numeric tokens in the target must survive the rewrite unchanged.

        Only applies to span replacement, where the original span is known.
        Extra numbers in the output are a fabrication; missing ones are a
        dropped fact. Both are rejected.
        """
        if original is None:
            return None
        src, out = _numbers(original), _numbers(text)
        missing = [n for n in set(src) if src.count(n) > out.count(n)]
        added = [n for n in set(out) if out.count(n) > src.count(n)]
        if missing or added:
            return (f"numbers: missing={sorted(missing)} added={sorted(added)}")
        return None

    # -- driver -----------------------------------------------------------
    def validate(
        self,
        text: str,
        *,
        target_sentence_count: int,
        target_word_count: int,
        original_span: str | None = None,
        check_numbers: bool = True,
    ) -> ValidationResult:
        """Run all checks. `original_span` enables checks 6 and 7."""
        errors: list[str] = []
        if not text or not text.strip():
            return ValidationResult(False, ["empty: model returned no text"])

        sents = segment_sentences(text, self.cfg)

        for err in (
            self.check_sentence_count(sents, target_sentence_count),
            self.check_length(text, target_word_count),
            self.check_no_preamble(text),
            self.check_no_markdown(text),
            self.check_sinhala_script(text),
            self.check_not_a_copy(text, original_span),
            self.check_numbers_consistent(
                text, original_span if check_numbers else None),
        ):
            if err:
                errors.append(err)

        return ValidationResult(
            passed=not errors,
            errors=errors,
            details={
                "n_sentences": len(sents),
                "n_words": word_count(text),
                "sinhala_ratio": round(sinhala_ratio(text), 3),
                "latin_ratio": round(latin_ratio(text), 3),
            },
        )
