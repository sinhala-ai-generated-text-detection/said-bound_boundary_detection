"""Construction-plan and assembly invariants.

These guard the properties the dataset's labels depend on: spans never touch
the document edges, spans never overlap or abut, and labels/boundaries always
describe the assembled text exactly.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from generate import (  # noqa: E402
    TYPE1,
    TYPE2,
    TYPE3,
    assemble,
    plan_type1,
    plan_type2,
    plan_type3,
)
from utils import load_config  # noqa: E402

CFG = load_config()


def make_window(n: int) -> dict:
    sents = [f"මෙය අංක {i} දරන පරීක්ෂණ වාක්‍යය වන අතර එය දිගු වේ." for i in range(n)]
    return {
        "window_id": f"w{n}", "source_id": f"s{n}", "title": "පරීක්ෂණය",
        "domain": "wikipedia", "sentences": sents, "n_sentences": n,
    }


def rngs(k=40):
    return [random.Random(i) for i in range(k)]


# ------------------------------------------------------------- Type 1 -----

def test_type1_respects_min_sentences():
    c = CFG["constructions"][TYPE1]
    assert plan_type1(make_window(c["min_sentences"] - 1), CFG, random.Random(0)) is None


def test_type1_min_human_and_ai_sentences():
    c = CFG["constructions"][TYPE1]
    for n in range(c["min_sentences"], 14):
        for r in rngs(20):
            p = plan_type1(make_window(n), CFG, r)
            if p is None:
                continue
            b = p["boundary_index"]
            assert b >= c["min_human_sentences"], (n, b)
            assert n - b >= c["min_ai_sentences"], (n, b)


def test_type1_boundary_inside_configured_band():
    c = CFG["constructions"][TYPE1]
    n = 10
    for r in rngs(60):
        p = plan_type1(make_window(n), CFG, r)
        assert p is not None
        frac = p["boundary_index"] / n
        # Band is applied then clamped by the min-sentence rules.
        assert 0.15 <= frac <= 0.85, frac


def test_type1_boundary_varies():
    """Guards against positional bias: the split must not be constant."""
    seen = {plan_type1(make_window(12), CFG, r)["boundary_index"] for r in rngs(60)}
    assert len(seen) > 1


def test_type1_targets_match_removed_text():
    p = plan_type1(make_window(10), CFG, random.Random(3))
    assert p["target_sentence_count"] == len(p["removed"])
    assert p["target_word_count"] == len(" ".join(p["removed"]).split())


# ------------------------------------------------------------- Type 2 -----

def test_type2_respects_min_sentences():
    c = CFG["constructions"][TYPE2]
    assert plan_type2(make_window(c["min_sentences"] - 1), CFG, random.Random(0)) is None


def test_type2_span_never_touches_edges():
    for n in range(CFG["constructions"][TYPE2]["min_sentences"], 14):
        for r in rngs(20):
            p = plan_type2(make_window(n), CFG, r)
            if p is None:
                continue
            (s, e), = p["spans"]
            assert s >= 1, f"span touches sentence 0 (n={n}, s={s})"
            assert e <= n - 1, f"span touches final sentence (n={n}, e={e})"


def test_type2_single_span():
    p = plan_type2(make_window(10), CFG, random.Random(1))
    assert len(p["spans"]) == 1


def test_type2_span_size_in_range():
    c = CFG["constructions"][TYPE2]
    for r in rngs(30):
        p = plan_type2(make_window(10), CFG, r)
        if p is None:
            continue
        (s, e), = p["spans"]
        assert c["span_min_sentences"] <= e - s <= c["span_max_sentences"]


def test_type2_ai_ratio_within_band():
    c = CFG["constructions"][TYPE2]
    n = 10
    for r in rngs(30):
        p = plan_type2(make_window(n), CFG, r)
        if p is None:
            continue
        (s, e), = p["spans"]
        assert c["ai_ratio_min"] <= (e - s) / n <= c["ai_ratio_max"]


# ------------------------------------------------------------- Type 3 -----

def test_type3_respects_min_sentences():
    c = CFG["constructions"][TYPE3]
    assert plan_type3(make_window(c["min_sentences"] - 1), CFG, random.Random(0)) is None


def test_type3_span_count_in_range():
    c = CFG["constructions"][TYPE3]
    for r in rngs(30):
        p = plan_type3(make_window(12), CFG, r)
        if p is None:
            continue
        assert c["num_spans_min"] <= len(p["spans"]) <= c["num_spans_max"]


def test_type3_spans_non_overlapping_and_gapped():
    gap = CFG["constructions"][TYPE3]["min_gap_sentences"]
    for n in range(CFG["constructions"][TYPE3]["min_sentences"], 16):
        for r in rngs(25):
            p = plan_type3(make_window(n), CFG, r)
            if p is None:
                continue
            spans = p["spans"]
            for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
                assert e1 <= s2, f"overlap {spans}"
                assert s2 - e1 >= gap, f"spans abut without human gap: {spans}"


def test_type3_spans_never_touch_edges():
    for n in range(CFG["constructions"][TYPE3]["min_sentences"], 16):
        for r in rngs(25):
            p = plan_type3(make_window(n), CFG, r)
            if p is None:
                continue
            assert p["spans"][0][0] >= 1
            assert p["spans"][-1][1] <= n - 1


def test_type3_ai_ratio_capped():
    cap = CFG["constructions"][TYPE3]["ai_ratio_max"]
    for n in range(8, 16):
        for r in rngs(25):
            p = plan_type3(make_window(n), CFG, r)
            if p is None:
                continue
            assert sum(e - s for s, e in p["spans"]) / n <= cap


# ------------------------------------------------------------ assembly ----

def test_assemble_type1_labels():
    w = make_window(10)
    p = plan_type1(w, CFG, random.Random(5))
    ai = ["කෘත්‍රිම වාක්‍යය එකයි.", "කෘත්‍රිම වාක්‍යය දෙකයි."]
    a = assemble(w, p, [ai])
    b = p["boundary_index"]
    assert a["labels"] == [0] * b + [1] * len(ai)
    assert a["boundaries"] == [b]
    assert len(a["sentences"]) == len(a["labels"])


def test_assemble_type1_single_boundary():
    w = make_window(10)
    p = plan_type1(w, CFG, random.Random(6))
    a = assemble(w, p, [["ක.", "ඛ."]])
    assert len(a["boundaries"]) == 1


def test_assemble_type2_two_boundaries():
    w = make_window(10)
    p = plan_type2(w, CFG, random.Random(2))
    (s, e), = p["spans"]
    ai = ["කෘත්‍රිම වාක්‍යයකි."] * (e - s)
    a = assemble(w, p, [ai])
    assert len(a["boundaries"]) == 2, a["boundaries"]
    assert a["labels"][s] == 1 and a["labels"][s - 1] == 0
    assert a["labels"][e] == 0


def test_assemble_type3_boundary_count():
    w = make_window(14)
    p = plan_type3(w, CFG, random.Random(4))
    assert p is not None
    ai = [["කෘත්‍රිම වාක්‍යයකි."] * (e - s) for s, e in p["spans"]]
    a = assemble(w, p, ai)
    # Each internal span contributes exactly two boundaries.
    assert len(a["boundaries"]) == 2 * len(p["spans"])


def test_assemble_labels_align_with_sentences():
    for ctype, planner in ((TYPE1, plan_type1), (TYPE2, plan_type2),
                           (TYPE3, plan_type3)):
        w = make_window(14)
        p = planner(w, CFG, random.Random(9))
        assert p is not None, ctype
        if ctype == TYPE1:
            ai = [["ක වාක්‍යය."] * p["target_sentence_count"]]
        else:
            ai = [["ක වාක්‍යය."] * (e - s) for s, e in p["spans"]]
        a = assemble(w, p, ai)
        assert len(a["sentences"]) == len(a["labels"]), ctype
        assert a["text"] == " ".join(a["sentences"]), ctype


def test_assemble_preserves_human_sentences_outside_spans():
    w = make_window(12)
    p = plan_type2(w, CFG, random.Random(11))
    (s, e), = p["spans"]
    a = assemble(w, p, [["කෘත්‍රිම."] * (e - s)])
    assert a["sentences"][:s] == w["sentences"][:s]
    assert a["sentences"][s + (e - s):] == w["sentences"][e:]


def test_boundary_position_normalized_range():
    w = make_window(10)
    p = plan_type1(w, CFG, random.Random(7))
    a = assemble(w, p, [["ක.", "ඛ."]])
    assert 0.0 < a["boundary_position_normalized"] < 1.0


def test_type3_generated_independently_from_original():
    """Each span's target must come from the ORIGINAL human sentences."""
    w = make_window(14)
    p = plan_type3(w, CFG, random.Random(8))
    assert p is not None
    for (s, e), target in zip(p["spans"], p["targets"]):
        assert target == w["sentences"][s:e]
