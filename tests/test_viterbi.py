"""Viterbi decoder: does the structured decode behave as intended."""

import numpy as np
import pytest


torch = pytest.importorskip("torch")
from transformer import viterbi_decode  # noqa: E402


def lp(probs):
    """(n,) P(AI) -> (n,2) log-probs."""
    p = np.asarray(probs, dtype=float)
    return np.log(np.stack([1 - p, p], axis=1) + 1e-12)


def test_follows_sentence_head_when_pairs_are_neutral():
    # Neutral pair logits (0.0 => P(change)=0.5) must not override a confident
    # sentence head.
    out = viterbi_decode(lp([0.01, 0.01, 0.99, 0.99]), np.zeros(3))
    assert out == [0, 0, 1, 1]


def test_pair_head_can_place_a_boundary():
    # Sentence head is uncertain everywhere; a strong change signal at index 2
    # should still produce exactly one boundary there.
    out = viterbi_decode(lp([0.45, 0.45, 0.55, 0.55]),
                         np.array([-6.0, 6.0, -6.0]))
    assert out in ([0, 0, 1, 1], [1, 1, 0, 0])
    changes = [i for i in range(1, len(out)) if out[i] != out[i - 1]]
    assert changes == [2]


def test_admits_single_sentence_spans():
    # The failure of run-length smoothing was deleting 1-sentence spans.
    # Viterbi must be able to emit one.
    out = viterbi_decode(lp([0.01, 0.99, 0.01, 0.01]),
                         np.array([8.0, 8.0, -8.0]))
    assert out == [0, 1, 0, 0]


def test_negative_bias_suppresses_boundaries():
    # A large negative bias makes changes expensive: collapse to one author.
    out = viterbi_decode(lp([0.2, 0.8, 0.2, 0.8]), np.zeros(3),
                         boundary_bias=-50.0)
    assert len(set(out)) == 1


def test_positive_bias_encourages_boundaries():
    out = viterbi_decode(lp([0.45, 0.55, 0.45, 0.55]), np.zeros(3),
                         boundary_bias=50.0)
    changes = [i for i in range(1, len(out)) if out[i] != out[i - 1]]
    assert len(changes) == 3


def test_length_and_edge_cases():
    assert viterbi_decode(lp([]), np.zeros(0)) == []
    assert viterbi_decode(lp([0.9]), np.zeros(0)) == [1]
    assert viterbi_decode(lp([0.1]), np.zeros(0)) == [0]
    out = viterbi_decode(lp([0.3] * 9), np.zeros(8))
    assert len(out) == 9
