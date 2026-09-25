"""Clean the raw dump: namespace filter -> page-type filter -> markup strip
-> prose gate.

Cleaning runs on `raw_mediawiki` (see config.schema.text_col). The shipped
`text` column is not trustworthy: it still contains Category: lines,
{{#ifexpr}} residue and __NOTOC__ markers, and it flattens list bullets into
leading spaces, which corrupts sentence segmentation.

Every filter increments a counter so the drop breakdown can be reported.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from segment import segment_sentences, sinhala_ratio, word_count  # noqa: E402
from utils import force_utf8_stdout, load_config, write_jsonl  # noqa: E402

# --------------------------------------------------------------- markup ----

_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_REF_PAIRED = re.compile(r"<ref[^>]*>.*?</ref\s*>", re.DOTALL | re.IGNORECASE)
_REF_SELF = re.compile(r"<ref[^>]*/\s*>", re.IGNORECASE)
_HTML_TAG = re.compile(r"</?[A-Za-z][A-Za-z0-9]*(?:\s[^<>]*)?/?>")
_NOWIKI = re.compile(r"<nowiki>(.*?)</nowiki>", re.DOTALL | re.IGNORECASE)
_MATH = re.compile(r"<math[^>]*>.*?</math\s*>", re.DOTALL | re.IGNORECASE)
_GALLERY = re.compile(r"<gallery[^>]*>.*?</gallery\s*>", re.DOTALL | re.IGNORECASE)
_TIMELINE = re.compile(r"<timeline[^>]*>.*?</timeline\s*>", re.DOTALL | re.IGNORECASE)

_HEADING = re.compile(r"(?m)^\s*={2,6}\s*(.*?)\s*={2,6}\s*$")
_BOLD_ITALIC = re.compile(r"'{2,5}")
_BEHAVIOUR_SWITCH = re.compile(r"__[A-Z_]+__")
_HORIZ_RULE = re.compile(r"(?m)^-{4,}\s*$")

# Link namespaces whose whole link is dropped (media/categories/interwiki).
_FILE_NS = r"(?:File|Image|Media|ගොනුව|ගොනු|රූපය|මාධ්‍ය)"
_CAT_NS = r"(?:Category|ප්‍රවර්ගය|ප්‍රවර්ග)"

_CATEGORY_LINK = re.compile(r"\[\[\s*" + _CAT_NS + r"\s*:[^\]]*\]\]", re.IGNORECASE)
# Bare 'Category:x' / 'ප්‍රවර්ගය:x' lines (residue seen in the shipped text col).
_CATEGORY_LINE = re.compile(
    r"(?m)^\s*(?:" + _CAT_NS + r")\s*:.*$", re.IGNORECASE)
# Interwiki language links: [[en:Foo]], [[ta:Foo]] ...
_INTERWIKI = re.compile(r"\[\[\s*[a-z]{2,3}(?:-[a-z0-9-]+)?\s*:[^\]]*\]\]")

_EXTERNAL_LINK = re.compile(r"\[(?:https?:|ftp:|//)[^\s\]]+(?:\s+([^\]]*))?\]")
_BARE_URL = re.compile(r"https?://\S+")

_LIST_MARK = re.compile(r"^[*#:;]+[ \t]*")


def _reflow(text: str) -> str:
    """Rebuild paragraphs with MediaWiki newline semantics.

    In wikitext a *single* newline inside a paragraph is a soft wrap rendered
    as a space; only a blank line starts a new paragraph. Joining soft wraps
    matters because the segmenter treats a newline as a sentence break, so
    leaving them in splits sentences mid-clause.

    List items are emitted as their own paragraphs so they cannot run together
    into fake prose; short ones are then removed by drop_short_paragraphs.
    """
    paras: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            paras.append(" ".join(buf))
            buf.clear()

    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            flush()
            continue
        m = _LIST_MARK.match(line)
        if m:
            flush()
            item = line[m.end():].strip()
            if item:
                paras.append(item)
        else:
            buf.append(line)
    flush()
    return "\n\n".join(p for p in paras if p)


def _strip_balanced(text: str, open_tok: str, close_tok: str, max_iter: int = 200) -> str:
    """Remove balanced {{...}} / {|...|} spans, innermost first.

    A regex cannot match nested templates, so we repeatedly delete the
    innermost span (one containing no further opener) until none remain.
    """
    pattern = re.compile(
        re.escape(open_tok)
        + r"(?:(?!"
        + re.escape(open_tok)
        + r"|"
        + re.escape(close_tok)
        + r").)*"
        + re.escape(close_tok),
        re.DOTALL,
    )
    for _ in range(max_iter):
        new = pattern.sub(" ", text)
        if new == text:
            break
        text = new
    # Drop any unbalanced leftovers so they cannot leak into prose.
    text = text.replace(open_tok, " ").replace(close_tok, " ")
    return text


def _strip_file_links(text: str, max_iter: int = 60) -> str:
    """Remove [[File:...|...[[nested]]...]] image links including captions.

    Image captions frequently contain nested links, so peel innermost first.
    """
    inner = re.compile(r"\[\[\s*" + _FILE_NS + r"\s*:(?:(?!\[\[|\]\]).)*\]\]",
                       re.IGNORECASE | re.DOTALL)
    for _ in range(max_iter):
        new = inner.sub(" ", text)
        if new == text:
            break
        text = new
    # Nested case: a File link whose caption held links now has stray content.
    outer = re.compile(r"\[\[\s*" + _FILE_NS + r"\s*:[^\[\]]*(?:\[\[[^\]]*\]\][^\[\]]*)*\]\]",
                       re.IGNORECASE | re.DOTALL)
    for _ in range(max_iter):
        new = outer.sub(" ", text)
        if new == text:
            break
        text = new
    return text


def _resolve_wikilinks(text: str) -> str:
    """[[link|text]] -> text ; [[link]] -> link ; innermost first."""
    link = re.compile(r"\[\[(?:(?!\[\[|\]\]).)*?\]\]", re.DOTALL)

    def repl(m: re.Match) -> str:
        body = m.group(0)[2:-2]
        if "|" in body:
            body = body.split("|")[-1]
        # A surviving 'ns:target' with no display text is not prose.
        if ":" in body and "|" not in m.group(0)[2:-2]:
            head = body.split(":", 1)[0]
            if len(head) <= 20 and " " not in head:
                return " "
        return body

    for _ in range(40):
        new = link.sub(repl, text)
        if new == text:
            break
        text = new
    return text


def strip_markup(raw: str) -> str:
    """Full wikitext -> plain prose."""
    t = raw or ""
    t = _HTML_COMMENT.sub(" ", t)
    t = _NOWIKI.sub(r"\1", t)
    t = _MATH.sub(" ", t)
    t = _GALLERY.sub(" ", t)
    t = _TIMELINE.sub(" ", t)
    t = _REF_PAIRED.sub(" ", t)
    t = _REF_SELF.sub(" ", t)

    # Tables first (they embed templates), then templates.
    t = _strip_balanced(t, "{|", "|}")
    t = _strip_balanced(t, "{{", "}}")

    t = _strip_file_links(t)
    t = _CATEGORY_LINK.sub(" ", t)
    t = _INTERWIKI.sub(" ", t)
    t = _resolve_wikilinks(t)
    t = _CATEGORY_LINE.sub(" ", t)

    t = _EXTERNAL_LINK.sub(lambda m: (m.group(1) or " "), t)
    t = _BARE_URL.sub(" ", t)

    # A heading becomes a blank line, i.e. a paragraph break for _reflow.
    t = _HEADING.sub("\n", t)
    t = _BOLD_ITALIC.sub("", t)
    t = _BEHAVIOUR_SWITCH.sub(" ", t)
    t = _HORIZ_RULE.sub(" ", t)
    t = _HTML_TAG.sub(" ", t)

    # HTML entities that survive in older dumps.
    t = (t.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<")
          .replace("&gt;", ">").replace("&quot;", '"').replace("&#160;", " "))

    # Collapse horizontal whitespace, then rebuild paragraphs (this is where
    # soft-wrapped lines get joined and list items become own paragraphs).
    t = re.sub(r"[ \t\u00a0]+", " ", t)
    t = _reflow(t)
    t = re.sub(r"(?m)^ +| +$", "", t)
    return t.strip()


def drop_short_paragraphs(text: str, min_chars: int) -> str:
    """Remove caption/heading residue left as very short standalone blocks."""
    paras = [p.strip() for p in text.split("\n\n")]
    keep = [p for p in paras if len(p) >= min_chars]
    return "\n\n".join(keep).strip()


# ----------------------------------------------------------- page filters ---


class PageFilter:
    """Namespace + page-type + prose gate. Counts every drop reason."""

    def __init__(self, cfg: dict) -> None:
        c = cfg["clean"]
        self.cfg = cfg
        self.c = c
        self.prefixes = tuple(c["drop_title_prefixes"])
        self.exact_titles = set(c["drop_exact_titles"])
        self.redirect = [re.compile(p) for p in c["redirect_patterns"]]
        self.disambig = [re.compile(p) for p in c["disambiguation_patterns"]]
        self.stub = [re.compile(p) for p in c["stub_patterns"]]
        self.list_title = [re.compile(p) for p in c["list_title_patterns"]]
        self.counts: Counter = Counter()

    def _any(self, pats: list[re.Pattern], text: str) -> bool:
        return any(p.search(text) for p in pats)

    def reject_reason(self, title: str, raw: str) -> str | None:
        """Reason this page is not an eligible article, or None if it is."""
        title = (title or "").strip()
        raw = raw or ""

        if not raw.strip():
            return "empty_raw"
        if title in self.exact_titles:
            return "main_page"
        if title.startswith(self.prefixes):
            return "namespace"
        if self._any(self.redirect, raw[:200]):
            return "redirect"
        if self._any(self.disambig, raw):
            return "disambiguation"
        if self._any(self.stub, raw):
            return "stub"
        if self._any(self.list_title, title):
            return "list_page_title"

        lines = [ln for ln in raw.splitlines() if ln.strip()]
        if lines:
            bullets = sum(1 for ln in lines if re.match(r"^\s*[*#]", ln))
            if bullets / len(lines) > self.c["max_bullet_line_ratio"]:
                return "list_page_content"
        return None

    def prose_gate(self, cleaned: str) -> tuple[str | None, list[str]]:
        """Reason the cleaned prose fails the gate (or None), plus sentences."""
        if len(cleaned) < self.c["min_chars"]:
            return "too_few_chars", []
        if word_count(cleaned) < self.c["min_words"]:
            return "too_few_words", []
        if sinhala_ratio(cleaned) < self.c["min_sinhala_char_ratio"]:
            return "not_sinhala", []
        sents = segment_sentences(cleaned, self.cfg)
        if len(sents) < self.c["min_sentences"]:
            return "too_few_sentences", sents
        return None, sents


# ------------------------------------------------------------------ main ---


def run(cfg: dict, limit: int | None = None, out_path: str | None = None,
        batch_size: int = 2000) -> dict:
    force_utf8_stdout()
    sc = cfg["schema"]
    pf = pq.ParquetFile(cfg["paths"]["raw_parquet"])
    filt = PageFilter(cfg)
    min_para = cfg["clean"]["min_paragraph_chars"]

    kept: list[dict] = []
    total = 0
    stop = False
    cols = [sc["id_col"], sc["title_col"], sc["url_col"], sc["text_col"]]

    for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
        d = batch.to_pydict()
        for pid, title, url, raw in zip(
            d[sc["id_col"]], d[sc["title_col"]], d[sc["url_col"]], d[sc["text_col"]]
        ):
            total += 1
            reason = filt.reject_reason(title, raw)
            if reason:
                filt.counts[reason] += 1
                continue
            cleaned = strip_markup(raw)
            cleaned = drop_short_paragraphs(cleaned, min_para)
            gate, sents = filt.prose_gate(cleaned)
            if gate:
                filt.counts[gate] += 1
                continue
            filt.counts["kept"] += 1
            kept.append({
                "source_id": str(pid),
                "title": title,
                "url": url,
                "domain": "wikipedia",
                "text": cleaned,
                "sentences": sents,
                "n_sentences": len(sents),
                "n_words": word_count(cleaned),
            })
            if limit and len(kept) >= limit:
                stop = True
                break
        if stop:
            break

    out = out_path or cfg["paths"]["sources"]
    write_jsonl(out, kept)

    # ---- report -----------------------------------------------------------
    order = ["empty_raw", "main_page", "namespace", "redirect", "disambiguation",
             "stub", "list_page_title", "list_page_content", "too_few_chars",
             "too_few_words", "not_sinhala", "too_few_sentences", "kept"]
    print("=" * 66)
    print(f"CLEANING REPORT   (scanned {total} pages)")
    print("=" * 66)
    running = total
    for k in order:
        v = filt.counts.get(k, 0)
        if k == "kept":
            continue
        if v:
            running -= v
            print(f"  drop {k:<20s} {v:7d}  ({v/total:6.2%})   remaining {running}")
    kept_n = filt.counts.get("kept", 0)
    print("-" * 66)
    print(f"  KEPT {kept_n} / {total}  ({kept_n/total:.2%})")
    if kept:
        ns = sorted(r["n_sentences"] for r in kept)
        nw = sorted(r["n_words"] for r in kept)
        pct = lambda a, p: a[min(len(a) - 1, int(len(a) * p / 100))]  # noqa: E731
        print(f"  sentences/page  p25={pct(ns,25)} p50={pct(ns,50)} "
              f"p75={pct(ns,75)} p90={pct(ns,90)} max={ns[-1]}")
        print(f"  words/page      p25={pct(nw,25)} p50={pct(nw,50)} "
              f"p75={pct(nw,75)} p90={pct(nw,90)} max={nw[-1]}")
        # Eligibility by construction type.
        cons = cfg["constructions"]
        for name in ("type1_single_boundary", "type2_single_internal_segment",
                     "type3_multiple_internal_segments"):
            m = cons[name]["min_sentences"]
            n = sum(1 for r in kept if r["n_sentences"] >= m)
            print(f"  eligible {name:<34s} (>={m} sents): {n}")
    print(f"\n  wrote -> {out}")
    return dict(filt.counts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--limit", type=int, default=None, help="stop after N kept pages")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(load_config(a.config), limit=a.limit, out_path=a.out)


if __name__ == "__main__":
    main()
