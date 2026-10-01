"""Hugging Face export: span derivation, twin alignment, unrelated windows."""
import json

import pytest

from export_huggingface import ai_spans, human_rows  # noqa: E402


@pytest.mark.parametrize("labels, spans", [
    ([0, 0, 1, 1], [[2, 4]]),                       # continuation runs to the end
    ([0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0], [[1, 2], [3, 4], [8, 9]]),
    ([0, 1, 1, 1, 0], [[1, 4]]),
    ([0, 0, 0], []),
    ([1, 0], [[0, 1]]),
])
def test_ai_spans_are_half_open_runs_of_ai_sentences(labels, spans):
    assert ai_spans(labels) == spans


def window(wid, sid, sents):
    return {"window_id": wid, "source_id": sid, "title": "t", "url": "u",
            "text": " ".join(sents), "sentences": sents, "n_words": len(sents)}


def mixed(pid, wid, sid, sents, labels, split="train"):
    return {"passage_id": pid, "window_id": wid, "source_id": sid,
            "sentences": sents, "labels": labels, "n_sentences": len(sents), "split": split}


@pytest.fixture
def cfg(tmp_path):
    sources = tmp_path / "sources"
    sources.mkdir()
    windows = [window("a_w0", "a", ["h1.", "h2.", "h3."]),
               window("b_w0", "b", ["x1.", "x2."]),       # article no passage uses
               window("c_w0", "c", ["y1.", "y2."])]       # article in no split
    (sources / "windows.jsonl").write_text(
        "\n".join(json.dumps(w) for w in windows), encoding="utf-8")
    splits = tmp_path / "splits"
    splits.mkdir()
    (splits / "train_source_ids.txt").write_text("a\nb\n", encoding="utf-8")
    (splits / "dev_source_ids.txt").write_text("", encoding="utf-8")
    (splits / "test_source_ids.txt").write_text("", encoding="utf-8")
    return {"paths": {"sources": str(sources / "human_sources.jsonl"),
                      "splits_dir": str(splits)}}


def test_originals_group_passages_and_unrelated_skip_used_articles(cfg):
    passages = [mixed("p1", "a_w0", "a", ["h1.", "AI.", "h3."], [0, 1, 0]),
                mixed("p2", "a_w0", "a", ["AI.", "h2.", "h3."], [1, 0, 0])]
    originals, unrelated = human_rows(cfg, passages)
    assert len(originals) == 1
    assert originals[0]["sentences"] == ["h1.", "h2.", "h3."]
    assert originals[0]["labels"] == [0, 0, 0]
    assert originals[0]["passage_ids"] == ["p1", "p2"]
    assert [u["window_id"] for u in unrelated] == ["b_w0"]
    assert unrelated[0]["split"] == "train"


def test_misaligned_human_sentence_is_rejected(cfg):
    bad = [mixed("p1", "a_w0", "a", ["h1.", "AI.", "edited h3."], [0, 1, 0])]
    with pytest.raises(ValueError, match="does not align"):
        human_rows(cfg, bad)


def test_window_split_across_splits_is_rejected(cfg):
    passages = [mixed("p1", "a_w0", "a", ["h1.", "AI.", "h3."], [0, 1, 0], "train"),
                mixed("p2", "a_w0", "a", ["AI.", "h2.", "h3."], [1, 0, 0], "test")]
    with pytest.raises(ValueError, match="spans splits"):
        human_rows(cfg, passages)
