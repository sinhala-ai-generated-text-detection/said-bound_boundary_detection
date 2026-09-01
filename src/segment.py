"""Deterministic Sinhala sentence segmenter.

This is THE segmenter. Source prep, generation length targets, labelling and
evaluation all call `segment_sentences` so that sentence indices mean the same
thing everywhere in the pipeline. It is pure, deterministic and dependency-free.

Splitting happens on '.', '?', '!', '෴' (kunddaliya) and '…', with guards for:
  - abbreviations   (ආචාර්ය.  Dr.  ක්‍රි.ව.)
  - decimals        (3.14, 1,250.50)
  - initials        (A. B. Silva)
  - ellipsis runs   (...)
  - closing quotes/brackets that trail the terminator
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_config  # noqa: E402

# Sinhala block; used for script-ratio checks across the pipeline.
SINHALA_RANGE = (0x0D80, 0x0DFF)
_SINHALA_RE = re.compile(r"[඀-෿]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)

# Characters allowed to trail a terminator while still closing the sentence.
_TRAILING = "\"'’”)]}»"

_DEFAULT_TERMINATORS = [".", "?", "!", "෴", "…"]
_DEFAULT_ABBREV: list[str] = [
    "ආචාර්ය", "මහාචාර්ය", "පූජ්‍ය", "ශ්‍රී", "මයා", "මිය", "අංක",
    "ක්‍රි.ව", "ක්‍රි.පූ", "ඉ.පෙ", "පි.ව",
    "Dr", "Mr", "Mrs", "Ms", "Prof", "St", "Jr", "Sr",
    "vs", "etc", "No", "Fig", "Vol", "pp", "ed", "eds", "cf", "al",
]


def sinhala_ratio(text: str) -> float:
    """Share of *letters* that are Sinhala. Digits/punctuation ignored."""
    letters = _LETTER_RE.findall(text)
    if not letters:
        return 0.0
    sin = sum(1 for c in letters if SINHALA_RANGE[0] <= ord(c) <= SINHALA_RANGE[1])
    return sin / len(letters)


def latin_ratio(text: str) -> float:
    letters = _LETTER_RE.findall(text)
    if not letters:
        return 0.0
    return sum(1 for c in letters if _LATIN_RE.match(c)) / len(letters)


def word_count(text: str) -> int:
    """Whitespace word count; the pipeline's single definition of 'word'."""
    return len(text.split())


class SinhalaSegmenter:
    """Configured segmenter. Prefer the module-level `segment_sentences`."""

    def __init__(
        self,
        terminators: Sequence[str] | None = None,
        abbreviations: Sequence[str] | None = None,
        min_sentence_chars: int = 15,
    ) -> None:
        self.terminators = list(terminators or _DEFAULT_TERMINATORS)
        self.abbreviations = set(abbreviations or _DEFAULT_ABBREV)
        self.min_sentence_chars = min_sentence_chars
        # Longest-first so 'ක්‍රි.ව' wins over 'ක්‍රි'.
        self._abbrev_sorted = sorted(self.abbreviations, key=len, reverse=True)
        self._term_set = set(self.terminators)

    # -- guards ----------------------------------------------------------
    def _is_decimal_point(self, text: str, i: int) -> bool:
        """True if text[i]=='.' sits between digits, e.g. 3.14 (also 1.5%)."""
        if text[i] != ".":
            return False
        prev_digit = i > 0 and text[i - 1].isdigit()
        nxt = i + 1
        next_digit = nxt < len(text) and text[nxt].isdigit()
        return prev_digit and next_digit

    def _preceding_token(self, text: str, i: int) -> str:
        """The run of non-space chars ending just before index i."""
        j = i
        while j > 0 and not text[j - 1].isspace():
            j -= 1
        return text[j:i]

    def _is_abbreviation(self, text: str, i: int) -> bool:
        """True if the '.' at i terminates a known abbreviation or initial."""
        if text[i] != ".":
            return False
        tok = self._preceding_token(text, i)
        if tok in self.abbreviations:
            return True
        # Dotted abbreviation already containing periods: 'ක්‍රි.ව.' -> tok='ක්‍රි.ව'
        for ab in self._abbrev_sorted:
            if tok == ab or tok.endswith("." + ab) or tok == ab.replace(".", ""):
                return True
        # Single-letter Latin initial: 'A. B. Silva'.
        # ASCII-only on purpose: in Sinhala a lone letter before a full stop is
        # usually the sentence-final particle (".. සම්මාන ය.", ".. වේ ය."), so
        # treating it as an initial silently merges sentences.
        if len(tok) == 1 and tok.isascii() and tok.isalpha():
            return True
        return False

    def _is_ellipsis_interior(self, text: str, i: int) -> bool:
        """Inside a run of dots ('...'), only the final dot may terminate."""
        if text[i] != ".":
            return False
        return i + 1 < len(text) and text[i + 1] == "."

    @staticmethod
    def _quote_positions(text: str) -> set[int]:
        """Indices that sit inside a balanced double-quoted span.

        A terminator inside a quotation ('ඔහු "මම යනවා." කීවේය.') closes the
        quote, not the sentence. Only applied when quotes are balanced, so an
        unpaired quote cannot suppress every split in a document.
        """
        inside: set[int] = set()
        for quote in ('"', "“”", "‘’"):
            if len(quote) == 2:  # directional pair
                opens = [k for k, c in enumerate(text) if c == quote[0]]
                closes = [k for k, c in enumerate(text) if c == quote[1]]
                if len(opens) != len(closes):
                    continue
                for o, c in zip(opens, closes):
                    if o < c:
                        inside.update(range(o + 1, c))
            else:
                pos = [k for k, c in enumerate(text) if c == quote]
                if len(pos) % 2 != 0:
                    continue
                for o, c in zip(pos[0::2], pos[1::2]):
                    inside.update(range(o + 1, c))
        return inside

    # -- main ------------------------------------------------------------
    def segment(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        raw_pieces: list[str] = []
        quoted = self._quote_positions(text)
        start = 0
        i = 0
        n = len(text)
        while i < n:
            ch = text[i]
            if ch in self._term_set:
                if (
                    i in quoted
                    or self._is_decimal_point(text, i)
                    or self._is_abbreviation(text, i)
                    or self._is_ellipsis_interior(text, i)
                ):
                    i += 1
                    continue
                # Absorb repeated terminators ('!!', '?!', '...') and trailers.
                j = i + 1
                while j < n and (text[j] in self._term_set or text[j] in _TRAILING):
                    j += 1
                # A terminator only ends a sentence at end-of-text or before
                # whitespace; 'e.g' style mid-token dots do not split.
                if j < n and not text[j].isspace():
                    i += 1
                    continue
                piece = text[start:j].strip()
                if piece:
                    raw_pieces.append(piece)
                start = j
                i = j
                continue
            # A newline run also ends a sentence (headings, list residue).
            if ch == "\n":
                j = i
                while j < n and text[j] in "\r\n":
                    j += 1
                piece = text[start:i].strip()
                if piece:
                    raw_pieces.append(piece)
                start = j
                i = j
                continue
            i += 1

        tail = text[start:].strip()
        if tail:
            raw_pieces.append(tail)

        return self._merge_fragments(raw_pieces)

    def _is_fragment(self, piece: str) -> bool:
        """Residue, not a sentence.

        A *properly terminated* short string is a real sentence ("ඔව්!") and is
        kept. What we merge away is unterminated residue left by headings, table
        cells and caption remnants, plus anything carrying no letters at all.
        """
        if not _LETTER_RE.search(piece):
            return True
        ends_terminated = piece.rstrip(_TRAILING).endswith(tuple(self._term_set))
        return not ends_terminated and len(piece) < self.min_sentence_chars

    def _merge_fragments(self, pieces: list[str]) -> list[str]:
        """Glue residue fragments onto a neighbouring sentence."""
        out: list[str] = []
        for p in pieces:
            if out and self._is_fragment(p):
                out[-1] = out[-1] + " " + p
            else:
                out.append(p)
        # A leading fragment has no predecessor; fold it forward instead.
        if len(out) > 1 and self._is_fragment(out[0]):
            out[1] = out[0] + " " + out[1]
            out.pop(0)
        return [re.sub(r"\s+", " ", s).strip() for s in out if s.strip()]


_SEGMENTER: SinhalaSegmenter | None = None


def get_segmenter(config: dict | None = None) -> SinhalaSegmenter:
    """Process-wide singleton built from config.yaml (cached)."""
    global _SEGMENTER
    if _SEGMENTER is None or config is not None:
        cfg = config if config is not None else load_config()
        s = cfg.get("segment", {})
        seg = SinhalaSegmenter(
            terminators=s.get("terminators"),
            abbreviations=s.get("abbreviations"),
            min_sentence_chars=s.get("min_sentence_chars", 15),
        )
        if config is None:
            _SEGMENTER = seg
        return seg
    return _SEGMENTER


def segment_sentences(text: str, config: dict | None = None) -> list[str]:
    """THE segmentation entry point. Use this everywhere."""
    return get_segmenter(config).segment(text)


if __name__ == "__main__":
    from utils import force_utf8_stdout

    force_utf8_stdout()
    demo = (
        "ආචාර්ය. සිල්වා මහතා 1972 දී උපත ලැබීය. "
        "ඔහුගේ උස 1.75 කි. එය ක්‍රි.ව. 200 දී සිදු විය෴ "
        "ඔබ එය දන්නවාද? ඔව්!"
    )
    for k, s in enumerate(segment_sentences(demo)):
        print(f"{k}: {s}")
