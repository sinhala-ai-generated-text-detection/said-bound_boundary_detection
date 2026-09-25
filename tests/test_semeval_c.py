"""English replication (SemEval-2024 Task 8 Subtask C): words, labels,
chunking and decoding."""
import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "replication" / "semeval_c"))

from corpus import (boundary_from_labels, fill_labels, gold_labels,  # noqa: E402
                    length_match, ngrams, tokenless, truncate_to, words_of)
from scoring import change_point, first_word, human_metrics  # noqa: E402


def test_words_follow_the_official_single_space_split():
    text = "Human  prefix.\nNext machine words"
    w = words_of(text)
    assert w == ["Human", "", "prefix.\nNext", "machine", "words"]
    # The gold boundary is the word count of the human prefix.
    assert len(words_of("Human  prefix.\nNext")) == 3
    assert [tokenless(x) for x in w] == [False, True, False, False, False]
    assert tokenless("\n") and tokenless("  ")


def test_gold_labels_and_boundary_rule():
    assert gold_labels(5, 2) == [0, 0, 1, 1, 1]
    assert gold_labels(3, 0) == [1, 1, 1]           # fully machine
    assert gold_labels(3, 7) == [0, 0, 0]           # boundary past the end
    assert boundary_from_labels([0, 0, 1, 0, 1]) == 2   # first machine word
    assert boundary_from_labels([0, 0, 0]) == 3     # none = n


def test_tokenless_words_inherit_a_neighbour():
    lab, has = [0, 9, 1, 9, 1], [True, False, True, False, True]
    assert fill_labels(lab, has, "prev") == [0, 0, 1, 1, 1]
    assert fill_labels(lab, has, "next") == [0, 1, 1, 1, 1]
    # Leading tokenless words take the first real word under "prev".
    assert fill_labels([9, 1, 0], [False, True, True], "prev") == [1, 1, 0]
    assert fill_labels([9, 9], [False, False], "prev") == [0, 0]


def test_first_word_and_change_point_decoders():
    nan = np.nan
    p = np.array([0.1, nan, 0.2, 0.9, 0.8, 0.95], dtype=np.float32)
    assert first_word(p, 0.5) == 3
    assert change_point(p) == 3
    # A single confident word flags a human document under the first-word
    # rule, but is not enough evidence for the change-point decoder.
    q = np.array([0.1] * 10 + [0.9] + [0.1] * 10, dtype=np.float32)
    assert first_word(q, 0.5) == 10
    assert change_point(q) == len(q)
    assert first_word(np.array([0.1, 0.2]), 0.5) == 2
    assert change_point(np.array([0.9, 0.9])) == 0


def test_human_metrics():
    docs = [{"text": "a b c d"}, {"text": "a b"}]
    probs = [np.array([0.1, 0.1, 0.9, 0.1]), np.array([0.1, 0.1])]
    m = human_metrics(docs, probs, [2, 2], thr=0.5)
    assert m["false_alarm"] == pytest.approx(0.5)
    # "prev" fill does not spread a flag: 1 of 4 words, then 0 of 2.
    assert m["frac_flagged"] == pytest.approx((0.25 + 0.0) / 2)


def test_leakage_ngrams_and_length_matching():
    a = "one two three four five six seven eight nine"
    assert ngrams(a) & ngrams("zero " + a)
    assert not ngrams("one two three four five six seven") & ngrams(a)
    t = "First sentence here. Second one is a bit longer. Third."
    assert truncate_to(t, 4) == "First sentence here."
    assert truncate_to(t, 9) == "First sentence here. Second one is a bit longer."
    pool = [(i, " ".join(["w"] * n)) for i, n in enumerate([10, 20, 30, 40])]
    got, st = length_match([20, 40], pool, random.Random(0))
    assert [len(words_of(t)) for _, t in got] == [20, 40]
    assert len({i for i, _ in got}) == 2 and st == {"whole": 2}


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    except (OSError, ImportError, ValueError):
        pytest.skip("no DeBERTa-v3 tokenizer available")


def test_first_subword_labels_and_chunk_reassembly(tok):
    from tagger import collate, encode_doc
    words = words_of("alpha  beta.\ngamma deltaepsilonzeta eta theta iota "
                     "kappa lambda mu")
    labels = gold_labels(len(words), 5)
    chunks = encode_doc(tok, words, labels, max_length=10)
    assert len(chunks) > 1                              # forced to chunk
    seen = {}
    for ids, pos, widx, lab in chunks:
        assert len(ids) <= 10
        for p, w, y in zip(pos, widx, lab):
            # Each scored position is the first subword of its word.
            first = tok(words[w], add_special_tokens=False)["input_ids"][0]
            assert ids[p] == first
            assert w not in seen
            seen[w] = y
    # Every word with a token appears once, with its gold label; the empty
    # word does not.
    assert sorted(seen) == [i for i, x in enumerate(words) if not tokenless(x)]
    assert all(seen[w] == labels[w] for w in seen)

    # Reassembly by (document, word index) through collate.
    ids, _, pos, lab, meta = collate([(*c, 0) for c in chunks],
                                     tok.pad_token_id)
    back = {}
    for b, (di, widx) in enumerate(meta):
        for j, w in enumerate(widx):
            back[w] = int(lab[b, j])
            assert int(ids[b, pos[b, j]]) == chunks[b][0][chunks[b][1][j]]
    assert back == seen


def test_vectorised_sweep_matches_the_first_word_rule():
    from sweep import GRID, first_word_curve
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(1, 30))
        p = rng.random(n).astype(np.float32)
        p[rng.random(n) < 0.2] = np.nan        # tokenless words
        got = first_word_curve(p, GRID)
        want = [first_word(p, t) for t in GRID]
        assert got.tolist() == want
