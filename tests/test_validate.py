"""Unit tests for the 7 validation checks."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from utils import load_config  # noqa: E402
from validate import Validator, clean_output  # noqa: E402

CFG = load_config()
V = Validator(CFG)

# Three well-formed Sinhala sentences, ~8 words each.
GOOD = ("ශ්‍රී ලංකාව දකුණු ආසියාවේ පිහිටි සුන්දර දූපත් රාජ්‍යයකි. "
        "එහි ජනගහනය මිලියන විසි දෙකක් පමණ වන බව සඳහන් වේ. "
        "ප්‍රධාන අපනයන භෝග අතර තේ සහ රබර් ප්‍රමුඛ වේ.")
GOOD_SENTS = 3
GOOD_WORDS = len(GOOD.split())


def v(text, sents=GOOD_SENTS, words=GOOD_WORDS, original=None):
    return V.validate(text, target_sentence_count=sents,
                      target_word_count=words, original_span=original)


# ------------------------------------------------------------- happy path --

def test_good_output_passes():
    r = v(GOOD)
    assert r.passed, r.errors


def test_result_is_truthy():
    assert bool(v(GOOD)) is True


# ------------------------------------------------- 1. sentence count -------

def test_wrong_sentence_count_fails():
    r = v(GOOD, sents=5)
    assert not r.passed
    assert any(e.startswith("sentence_count") for e in r.errors)


# ------------------------------------------------------- 2. length ---------

def test_too_short_fails():
    r = v(GOOD, words=200)
    assert not r.passed
    assert any(e.startswith("length") for e in r.errors)


def test_length_within_tolerance_passes():
    r = v(GOOD, words=int(GOOD_WORDS * 1.1))
    assert not any(e.startswith("length") for e in r.errors)


# ----------------------------------------------------- 3. preamble ---------

def test_english_preamble_fails():
    r = v("Sure! " + GOOD)
    assert any(e.startswith("preamble") for e in r.errors)


def test_here_is_preamble_fails():
    r = v("Here is the continuation: " + GOOD)
    assert any(e.startswith("preamble") for e in r.errors)


def test_sinhala_preamble_fails():
    r = v("මෙන්න ඔබට අවශ්‍ය පෙළ: " + GOOD)
    assert any(e.startswith("preamble") for e in r.errors)


# ----------------------------------------------------- 4. markdown ---------

def test_heading_fails():
    r = v("## ශීර්ෂය\n" + GOOD)
    assert any(e.startswith("markdown") for e in r.errors)


def test_bold_fails():
    r = v(GOOD.replace("ශ්‍රී ලංකාව", "**ශ්‍රී ලංකාව**"))
    assert any(e.startswith("markdown") for e in r.errors)


def test_bullets_fail():
    r = v("- එකයි\n- දෙකයි\n- තුනයි")
    assert any(e.startswith("markdown") for e in r.errors)


def test_numbered_list_fails():
    r = v("1. එකයි\n2. දෙකයි\n3. තුනයි")
    assert any(e.startswith("markdown") for e in r.errors)


# ------------------------------------------------------- 5. script ---------

def test_english_output_fails_script():
    en = ("Sri Lanka is an island nation located in South Asia today. "
          "Its population is about twenty two million people in total. "
          "The main export crops are tea and rubber products.")
    r = v(en)
    assert any(e.startswith("script") for e in r.errors)


def test_latin_proper_noun_tolerated():
    """A legitimate Latin proper noun must not trip the script check."""
    t = ("ශ්‍රී ලංකාව දකුණු ආසියාවේ පිහිටි සුන්දර දූපත් රාජ්‍යයකි. "
         "එහි ජනගහනය මිලියන විසි දෙකක් පමණ වන බව Wikipedia සඳහන් කරයි. "
         "ප්‍රධාන අපනයන භෝග අතර තේ සහ රබර් ප්‍රමුඛ වේ.")
    r = V.validate(t, target_sentence_count=3,
                   target_word_count=len(t.split()))
    assert not any(e.startswith("script") for e in r.errors), r.errors


# --------------------------------------------------------- 6. copy ---------

def test_exact_copy_fails():
    r = v(GOOD, original=GOOD)
    assert any(e.startswith("copy") for e in r.errors)


def test_copy_detected_despite_punctuation_and_case():
    r = v(GOOD, original=GOOD.replace(".", " !").upper())
    assert any(e.startswith("copy") for e in r.errors)


def test_genuine_rewrite_passes_copy_check():
    original = ("ලංකාවේ ජනගහනය මිලියන විසි දෙකකි. "
                "එරට ප්‍රධාන අපනයනය තේ වේ. "
                "දිවයින ඉන්දියානු සාගරයේ පිහිටා ඇත.")
    rewrite = ("එම දිවයිනේ වැසියන් සංඛ්‍යාව මිලියන විසි දෙකක් වේ. "
               "ප්‍රධාන වශයෙන් අපනයනය කරනුයේ තේ ය. "
               "එය ඉන්දියානු සාගරය මැද පිහිටියකි.")
    r = V.validate(rewrite, target_sentence_count=3,
                   target_word_count=len(rewrite.split()),
                   original_span=original)
    assert not any(e.startswith("copy") for e in r.errors), r.errors


# ------------------------------------------------------ 7. numbers ---------

def test_mutated_number_fails():
    original = "එය 2025 වර්ෂයේ දී සිදු විය. එහි වටිනාකම රුපියල් 500 කි."
    bad = "එය 2027 වර්ෂයේ දී සිදු විය. එහි වටිනාකම රුපියල් 500 කි."
    r = V.validate(bad, target_sentence_count=2,
                   target_word_count=len(bad.split()), original_span=original)
    assert any(e.startswith("numbers") for e in r.errors), r.errors


def test_dropped_number_fails():
    original = "එය 1972 වර්ෂයේ දී ආරම්භ විය. සාමාජිකයන් 45 ක් විය."
    bad = "එය 1972 වර්ෂයේ දී ආරම්භ කරන ලදී. සාමාජිකයන් රැසක් සිටියහ."
    r = V.validate(bad, target_sentence_count=2,
                   target_word_count=len(bad.split()), original_span=original)
    assert any(e.startswith("numbers") for e in r.errors)


def test_preserved_numbers_pass():
    original = "එය 1972 වර්ෂයේ දී ආරම්භ විය. සාමාජිකයන් 45 ක් විය."
    ok = "එම ආයතනය 1972 වර්ෂයේ දී පිහිටුවන ලදී. එහි සාමාජිකයන් 45 දෙනෙකි."
    r = V.validate(ok, target_sentence_count=2,
                   target_word_count=len(ok.split()), original_span=original)
    assert not any(e.startswith("numbers") for e in r.errors), r.errors


def test_thousands_separator_equivalent():
    """1,250 and 1250 are the same number, not a mutation."""
    original = "වටිනාකම රුපියල් 1,250 කි. එය වැඩි විය."
    ok = "එහි වටිනාකම රුපියල් 1250 ක් වන අතර එය ඉහළ ය. පසුව එය තවත් වැඩි විය."
    r = V.validate(ok, target_sentence_count=2,
                   target_word_count=len(ok.split()), original_span=original)
    assert not any(e.startswith("numbers") for e in r.errors), r.errors


def test_numbers_skipped_for_continuation():
    """Type 1 has no original span, so number checking must not fire."""
    r = V.validate(GOOD, target_sentence_count=3, target_word_count=GOOD_WORDS,
                   original_span=None)
    assert not any(e.startswith("numbers") for e in r.errors)


# ------------------------------------------------------- clean_output ------

def test_clean_output_strips_code_fence():
    assert clean_output("```\n" + GOOD + "\n```") == GOOD


def test_clean_output_strips_wrapping_quotes():
    assert clean_output('"' + GOOD + '"') == GOOD


def test_clean_output_strips_prompt_separators():
    assert clean_output("---\n" + GOOD + "\n---") == GOOD


def test_clean_output_preserves_inner_text():
    assert clean_output(GOOD) == GOOD


def test_empty_output_fails():
    r = v("")
    assert not r.passed
    assert any(e.startswith("empty") for e in r.errors)
