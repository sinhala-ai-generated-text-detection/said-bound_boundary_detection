"""Unit tests for wikitext stripping and page-type filtering."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from clean import (  # noqa: E402
    PageFilter,
    drop_short_paragraphs,
    strip_markup,
)
from utils import load_config  # noqa: E402

CFG = load_config()
FILT = PageFilter(CFG)


# ------------------------------------------------------------- markup ------

def test_wikilink_piped_keeps_display_text():
    assert strip_markup("[[සිංහල භාෂාව|සිංහල]] කථා කරයි.") == "සිංහල කථා කරයි."


def test_wikilink_plain_keeps_target():
    assert strip_markup("[[ගණිතය]] යනු විෂයකි.") == "ගණිතය යනු විෂයකි."


def test_category_link_removed():
    out = strip_markup("පෙළ මෙයයි.\n[[Category:සිංහල]]\n[[ප්‍රවර්ගය:ලංකාව]]")
    assert "Category" not in out and "ප්‍රවර්ගය" not in out
    assert "පෙළ මෙයයි." in out


def test_bare_category_line_removed():
    """Residue seen in the shipped `text` column."""
    out = strip_markup("පෙළ මෙයයි.\nCategory:සිංහල\nප්‍රවර්ගය:ලාංකික")
    assert "Category:" not in out and "ප්‍රවර්ගය:" not in out


def test_interwiki_removed():
    out = strip_markup("පෙළ.\n[[en:Mathematics]]\n[[ta:கணிதம்]]")
    assert "en:" not in out and "Mathematics" not in out


def test_template_removed():
    assert "Infobox" not in strip_markup("{{Infobox country|name=Sri Lanka}}\nපෙළ මෙයයි.")


def test_nested_template_removed():
    out = strip_markup("{{outer|{{inner|x}}|y}}පෙළ මෙයයි.")
    assert "{" not in out and "}" not in out
    assert "පෙළ මෙයයි." in out


def test_table_removed():
    raw = '{| class="wikitable"\n! header\n| cell\n|}\nපෙළ මෙයයි.'
    out = strip_markup(raw)
    assert "wikitable" not in out and "header" not in out
    assert "පෙළ මෙයයි." in out


def test_html_comment_removed():
    assert "hidden" not in strip_markup("<!-- hidden note -->පෙළ මෙයයි.")


def test_ref_tags_removed():
    out = strip_markup("පෙළ<ref>Smith 2001, p. 4</ref> මෙයයි.<ref name=x/>")
    assert "Smith" not in out and "<ref" not in out


def test_file_link_removed_with_caption():
    raw = "[[ගොනුව:Euclid.jpg|right|thumb|220px|යුක්ලීඩ් ගේ සිතුවමක්]]\nපෙළ මෙයයි."
    out = strip_markup(raw)
    assert "Euclid.jpg" not in out and "thumb" not in out
    assert "පෙළ මෙයයි." in out


def test_file_link_with_nested_link_in_caption():
    raw = "[[File:X.jpg|thumb|[[Euclid]], රෆායල් විසින්]]\nපෙළ මෙයයි."
    out = strip_markup(raw)
    assert "X.jpg" not in out and "රෆායල්" not in out


def test_heading_removed():
    out = strip_markup("== ඉතිහාසය ==\nපෙළ මෙයයි.")
    assert "=" not in out
    assert "ඉතිහාසය" not in out


def test_bold_italic_markers_removed():
    assert strip_markup("'''ගණිතය''' යනු ''විෂයකි''.") == "ගණිතය යනු විෂයකි."


def test_external_link_keeps_label():
    assert strip_markup("[http://example.com උදාහරණය] බලන්න.") == "උදාහරණය බලන්න."


def test_bare_url_removed():
    assert "http" not in strip_markup("බලන්න http://example.com/x යන්න.")


def test_behaviour_switch_removed():
    assert "NOTOC" not in strip_markup("පෙළ.__NOTOC____NOEDITSECTION__")


def test_list_markers_removed():
    out = strip_markup("* එකයි\n* දෙකයි\n# තුනයි")
    assert "*" not in out and "#" not in out


def test_paragraph_breaks_preserved():
    out = strip_markup("පළමු ඡේදය මෙයයි.\n\nදෙවන ඡේදය මෙයයි.")
    assert "\n\n" in out


def test_empty_input():
    assert strip_markup("") == ""
    assert strip_markup(None) == ""


def test_drop_short_paragraphs():
    text = "කෙටි.\n\n" + ("දිගු ඡේදයක් මෙයයි එය බොහෝ අකුරු ඇතුළත් වේ. " * 3)
    out = drop_short_paragraphs(text, min_chars=60)
    assert "කෙටි." not in out
    assert "දිගු" in out


# -------------------------------------------------------- page filters -----

def test_accepts_normal_article():
    assert FILT.reject_reason("ගණිතය", "'''ගණිතය''' යනු විෂයකි. එය පුළුල් වේ.") is None


def test_rejects_main_page():
    assert FILT.reject_reason("මුල් පිටුව", "content") == "main_page"


def test_rejects_namespace_prefixes():
    for t in ["Talk:X", "Template:Y", "ප්‍රවර්ගය:Z", "WT:A", "Draft:B", "සාමාජික:C"]:
        assert FILT.reject_reason(t, "content") == "namespace", t


def test_rejects_redirect():
    assert FILT.reject_reason("X", "#REDIRECT [[Y]]") == "redirect"
    assert FILT.reject_reason("X", "#redirect [[Y]]") == "redirect"


def test_rejects_disambiguation():
    raw = "සිංහල යන්නෙන් අදහස් කෙරෙනුයේ,\n{{බහුරුත්හරණය}}"
    assert FILT.reject_reason("සිංහල (බහුරුත්හරණය)", raw) == "disambiguation"


def test_rejects_stub():
    assert FILT.reject_reason("X", "පෙළ.\n{{දියුණු කරන්න}}") == "stub"
    assert FILT.reject_reason("X", "text {{geo-stub}}") == "stub"


def test_rejects_list_page_by_title():
    assert FILT.reject_reason("ලංකාවේ ගංගා ලැයිස්තුව", "content") == "list_page_title"
    assert FILT.reject_reason("List of rivers", "content") == "list_page_title"


def test_rejects_list_page_by_content():
    raw = "හැඳින්වීම\n* එකයි\n* දෙකයි\n* තුනයි\n* හතරයි"
    assert FILT.reject_reason("X", raw) == "list_page_content"


def test_rejects_empty():
    assert FILT.reject_reason("X", "   ") == "empty_raw"


# ------------------------------------------------------------ prose gate ---

def test_prose_gate_rejects_short():
    reason, _ = FILT.prose_gate("කෙටි පෙළක්.")
    assert reason == "too_few_chars"


def test_prose_gate_rejects_non_sinhala():
    text = ("This is a long English passage repeated to pass the length gate. " * 12)
    reason, _ = FILT.prose_gate(text)
    assert reason in {"not_sinhala", "too_few_words", "too_few_chars"}


def test_prose_gate_accepts_real_prose():
    s = ("ශ්‍රී ලංකාව දකුණු ආසියාවේ පිහිටි දූපත් රාජ්‍යයකි එහි ජනගහනය මිලියන විසි දෙකකි. ")
    reason, sents = FILT.prose_gate(s * 12)
    assert reason is None
    assert len(sents) >= CFG["clean"]["min_sentences"]
