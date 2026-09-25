"""Hand-checked segmentation cases. Expected counts verified by reading."""
from __future__ import annotations


import pytest

from segment import (  # noqa: E402
    SinhalaSegmenter,
    latin_ratio,
    segment_sentences,
    sinhala_ratio,
    word_count,
)

SEG = SinhalaSegmenter()


def seg(t):
    return SEG.segment(t)


def test_simple_two_sentences():
    t = "ශ්‍රී ලංකාව දකුණු ආසියාවේ පිහිටි දූපත් රාජ්‍යයකි. එහි අගනුවර ශ්‍රී ජයවර්ධනපුර කෝට්ටේ වේ."
    assert len(seg(t)) == 2


def test_question_and_exclamation():
    t = "ඔබ කොහෙද යන්නේ? මම ගමට යනවා. එය පුදුමයි!"
    out = seg(t)
    assert len(out) == 3
    assert out[0].endswith("?")
    assert out[2].endswith("!")


def test_kunddaliya_terminator():
    t = "මෙය පැරණි ලේඛනයකි෴ එය පුස්කොළ පොතක ලියා ඇත෴"
    assert len(seg(t)) == 2


def test_decimal_not_split():
    t = "ඔහුගේ උස මීටර් 1.75 කි. බර කිලෝග්‍රෑම් 68.5 කි."
    out = seg(t)
    assert len(out) == 2
    assert "1.75" in out[0]
    assert "68.5" in out[1]


def test_thousands_and_decimal_combo():
    t = "එම ගොඩනැගිල්ලේ වටිනාකම රුපියල් 1,250.50 කි. එය 2020 දී ඉදිකරන ලදී."
    assert len(seg(t)) == 2


def test_sinhala_abbreviation_not_split():
    t = "ආචාර්ය. සිල්වා මහතා විශ්වවිද්‍යාලයේ කථිකාචාර්යවරයෙකි. ඔහු ගණිතය උගන්වයි."
    out = seg(t)
    assert len(out) == 2
    assert out[0].startswith("ආචාර්ය.")


def test_era_abbreviation_not_split():
    t = "එම සිද්ධිය ක්‍රි.පූ. 300 දී සිදු විය. පසුව එය වෙනස් විය."
    out = seg(t)
    assert len(out) == 2
    assert "ක්‍රි.පූ." in out[0]


def test_latin_initials_not_split():
    t = "A. B. සිල්වා මහතා පැමිණියේය. ඔහු ගුරුවරයෙකි."
    assert len(seg(t)) == 2


def test_english_abbreviation_not_split():
    t = "Dr. Perera ලංකාවට පැමිණියේය. ඔහු වෛද්‍යවරයෙකි."
    assert len(seg(t)) == 2


def test_ellipsis_single_sentence():
    t = "ඔහු කීවේ එපමණක් පමණි... නමුත් කිසිවෙක් විශ්වාස කළේ නැත."
    assert len(seg(t)) == 2


def test_empty_and_whitespace():
    assert seg("") == []
    assert seg("     ") == []
    assert seg("\n\n\n") == []


def test_no_terminator_single_sentence():
    t = "මෙය අවසන් තිත නොමැති වාක්‍යයකි"
    assert len(seg(t)) == 1


def test_newline_separates():
    t = "පළමු වාක්‍යය මෙයයි\nදෙවන වාක්‍යය මෙයයි"
    assert len(seg(t)) == 2


def test_unterminated_residue_merged():
    """Heading/caption residue (short, no terminator) is not a sentence."""
    t = "ඉතිහාසය\nමෙය දිගු වාක්‍යයක් වන අතර එය කරුණු කිහිපයක් විස්තර කරයි."
    assert len(seg(t)) == 1


def test_short_terminated_sentence_kept():
    """A short but properly terminated sentence stays its own sentence."""
    t = "මෙය දිගු වාක්‍යයක් වන අතර එය කරුණු කිහිපයක් විස්තර කරයි. හරි."
    assert len(seg(t)) == 2


def test_punctuation_only_residue_merged():
    t = "පළමු වාක්‍යය මෙයයි වන අතර එය දිගු වේ.\n|}"
    assert len(seg(t)) == 1


def test_determinism():
    t = "පළමු වාක්‍යය. දෙවන වාක්‍යය. තෙවන වාක්‍යය."
    assert seg(t) == seg(t) == segment_sentences(t)


def test_no_text_lost():
    """Concatenating segments must preserve every non-space character."""
    t = ("ශ්‍රී ලංකාව දූපතකි. එහි ජනගහනය මිලියන 22 කි෴ "
         "ආචාර්ය. පෙරේරා එය පවසයි! ඇත්තද?")
    joined = "".join(seg(t)).replace(" ", "")
    assert joined == t.replace(" ", "").replace("\n", "")


def test_trailing_quote_closes_sentence():
    t = "ඔහු \"මම යනවා.\" කීවේය. පසුව ඔහු ගියේය."
    assert len(seg(t)) == 2


@pytest.mark.parametrize(
    "text,expected",
    [
        ("එකයි. දෙකයි. තුනයි. හතරයි. පහයි.", 5),   # short but terminated = real sentences
        ("ලංකාවේ ප්‍රධාන නගරය කොළඹයි. එය වරාය නගරයකි.", 2),
    ],
)
def test_parametrised_counts(text, expected):
    assert len(seg(text)) == expected


# ---- script / counting helpers -------------------------------------------

def test_sinhala_ratio():
    assert sinhala_ratio("ශ්‍රී ලංකාව") == 1.0
    assert sinhala_ratio("hello world") == 0.0
    assert 0.0 < sinhala_ratio("ලංකාව Sri Lanka") < 1.0


def test_sinhala_ratio_ignores_digits_and_punct():
    """Digits and punctuation must not dilute the script ratio."""
    assert sinhala_ratio("ලංකාව 1972, 45.5%") == 1.0


def test_latin_ratio():
    assert latin_ratio("hello") == 1.0
    assert latin_ratio("ලංකාව") == 0.0


def test_word_count():
    assert word_count("එකයි දෙකයි තුනයි") == 3
    assert word_count("") == 0


def test_sinhala_final_particle_ya_ends_sentence():
    """'ය.' is a sentence-final particle, not a Latin-style initial."""
    t = "ඒවා නම් සිනමාව, රූපවාහිනිය සහ විශේෂ සම්මාන ය. තේරීම ඡන්දය මගිනි."
    assert len(seg(t)) == 2


def test_sinhala_final_particle_wey_ya():
    t = "එය වැදගත් කරුණක් වන බව පැහැදිලි වේ ය. එහෙත් එය තවම තහවුරු වී නොමැත."
    assert len(seg(t)) == 2


def test_latin_initial_still_guarded():
    t = "A. B. සිල්වා මහතා පැමිණියේය. ඔහු ගුරුවරයෙකි."
    assert len(seg(t)) == 2
