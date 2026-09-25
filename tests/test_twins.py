"""Counterfactual twins: construction, pairing and the paired loss."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

torch = pytest.importorskip("torch")
from data import Doc, load_twins, load_unrelated_twins  # noqa: E402
from transformer import collate_pairs, twin_loss, TwinPairDataset  # noqa: E402

WINDOW = ["human one.", "human two.", "human three.", "human four."]


def doc(rid, sents, labels, wid="w0"):
    return Doc(rid, "s0", "train", "gen", "type2", sents, labels, window_id=wid)


def write_windows(tmp_path, windows):
    p = tmp_path / "windows.jsonl"
    p.write_text("\n".join(json.dumps({"window_id": k, "sentences": v})
                           for k, v in windows.items()), encoding="utf-8")
    return p


def test_twin_restores_replaced_sentences_and_weights_shared_windows(tmp_path):
    path = write_windows(tmp_path, {"w0": WINDOW, "w1": WINDOW[:3]})
    a = doc("a", ["human one.", "AI two.", "human three.", "human four."],
            [0, 1, 0, 0])
    b = doc("b", ["human one.", "human two.", "AI three.", "AI four."],
            [0, 0, 1, 1])
    c = doc("c", ["human one.", "AI x.", "human three."], [0, 1, 0], "w1")
    twins, w = load_twins([a, b, c], {}, path)
    assert [t.sentences for t in twins] == [WINDOW, WINDOW, WINDOW[:3]]
    assert all(set(t.labels) == {0} for t in twins)
    assert w == [0.5, 0.5, 1.0]


def test_misaligned_human_sentence_is_refused(tmp_path):
    path = write_windows(tmp_path, {"w0": WINDOW})
    bad = doc("bad", ["human one.", "AI two.", "EDITED three.", "human four."],
              [0, 1, 0, 0])
    with pytest.raises(ValueError, match="do not align"):
        load_twins([bad], {}, path)


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    local = ROOT / "models" / "xlmr_tagger"
    try:
        return AutoTokenizer.from_pretrained(str(local) if local.exists()
                                             else "xlm-roberta-base")
    except OSError:
        pytest.skip("no XLM-R tokenizer available")


def test_pair_indices_point_at_the_same_sentence_in_both_documents(tok):
    # A short window forces chunking, and the twin's long human sentence makes
    # the two documents chunk at different places.
    long_h = "this human sentence is much longer than the machine one " * 3
    mixed = doc("m", ["first human.", "short ai.", "third human.", "last."],
                [0, 1, 0, 0])
    twin = doc("t", ["first human.", long_h, "third human.", "last."],
               [0, 0, 0, 0])
    ds = TwinPairDataset([mixed], [twin], [1.0], tok, max_length=40)
    assert len(ds.mixed[0]) != len(ds.twin[0])      # chunked differently
    ids, _, marker, _, gm, gt, y, _ = collate_pairs([ds[0]], tok.pad_token_id)
    width = marker.size(1)

    def first_token(flat):
        r, j = divmod(int(flat), width)
        return int(ids[r, marker[r, j] + 1])

    for si in range(4):
        want_m = tok.encode(mixed.sentences[si], add_special_tokens=False)[0]
        want_t = tok.encode(twin.sentences[si], add_special_tokens=False)[0]
        assert first_token(gm[si]) == want_m
        assert first_token(gt[si]) == want_t
    assert y.tolist() == mixed.labels


def test_loss_terms():
    # One chunk row, 3 sentences in each document: flat rows 0-2 mixed, 3-5 twin.
    logits = torch.tensor([[[2.0, 0.0], [0.0, 3.0], [1.0, 0.0],
                            [2.0, 0.0], [1.0, 0.0], [1.0, 0.0]]])
    gm, gt = torch.tensor([0, 1, 2]), torch.tensor([3, 4, 5])
    y, tw = torch.tensor([0, 1, 0]), torch.ones(3)
    ce, mrg, cons = twin_loss(logits, gm, gt, y, tw, margin=2.0)
    # Human context identical across the pair -> no consistency penalty.
    assert cons.item() == pytest.approx(0.0, abs=1e-7)
    # AI score gap: (3-0) - (0-1) = 4 >= margin -> no margin penalty.
    assert mrg.item() == pytest.approx(0.0, abs=1e-7)
    ref = torch.nn.functional.cross_entropy(
        logits[0], torch.tensor([0, 1, 0, 0, 0, 0]))
    assert ce.item() == pytest.approx(ref.item(), rel=1e-6)

    # Twin scores the replaced position as AI too: margin must bite.
    logits[0, 4] = torch.tensor([0.0, 3.0])
    _, mrg, _ = twin_loss(logits, gm, gt, y, tw, margin=2.0)
    assert mrg.item() == pytest.approx(2.0)


# ------------------------------------------------ unrelated human windows ---
def unrelated_setup(tmp_path, n_spare=20):
    """A tiny corpus: two used train articles, spare train articles of
    several lengths, a train article used only by a test-split record, and
    unused test articles."""
    windows = [("A", "A_w0", 4), ("C", "C_w0", 3), ("X", "X_w0", 4),
               ("T1", "T1_w0", 4), ("T2", "T2_w0", 3)]
    windows += [(f"U4_{i:02d}", f"U4_{i:02d}_w0", 4) for i in range(n_spare)]
    windows += [(f"U3_{i:02d}", f"U3_{i:02d}_w0", 3) for i in range(n_spare)]
    windows += [("U5_00", "U5_00_w0", 5)]
    rows = [{"window_id": w, "source_id": s,
             "sentences": [f"{s} sentence {k}." for k in range(n)]}
            for s, w, n in windows]
    (tmp_path / "windows.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    records = [{"source_id": "A", "split": "train"},
               {"source_id": "C", "split": "train"},
               # A record in another split still rules its article out.
               {"source_id": "X", "split": "test"}]
    (tmp_path / "combined.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records), encoding="utf-8")
    train_ids = [s for s, _, _ in windows if not s.startswith("T")]
    (tmp_path / "train_source_ids.txt").write_text("\n".join(train_ids))
    (tmp_path / "test_source_ids.txt").write_text("T1\nT2\n")
    cfg = {"paths": {"sources": str(tmp_path / "human_sources.jsonl"),
                     "generated_dir": str(tmp_path),
                     "splits_dir": str(tmp_path)}}
    wa = [f"A sentence {k}." for k in range(4)]
    wc = [f"C sentence {k}." for k in range(3)]
    docs = [doc("a", wa[:1] + ["AI."] + wa[2:], [0, 1, 0, 0], "A_w0"),
            doc("b", wa[:2] + ["AI.", "AI."], [0, 0, 1, 1], "A_w0"),
            doc("c", wc[:1] + ["AI."] + wc[2:], [0, 1, 0], "C_w0")]
    return cfg, docs, set(train_ids), {r["source_id"] for r in records}


def test_unrelated_candidates_are_unused_train_articles(tmp_path):
    from data import unused_windows
    cfg, docs, train_ids, used = unrelated_setup(tmp_path)
    cands = unused_windows(cfg, "train")
    assert cands and all(w["source_id"] in train_ids for w in cands)
    assert not {w["source_id"] for w in cands} & used
    twins, _ = load_unrelated_twins(docs, cfg, seed=1)
    assert all(t.source_id in train_ids and t.source_id not in used
               for t in twins)


def test_unrelated_twins_match_length_and_are_all_human(tmp_path):
    cfg, docs, _, _ = unrelated_setup(tmp_path)
    twins, _ = load_unrelated_twins(docs, cfg, seed=1)
    assert [t.n for t in twins] == [d.n for d in docs]
    assert all(t.labels == [0] * t.n for t in twins)
    assert all(t.window_id.startswith("U") for t in twins)


def test_unrelated_mapping_is_deterministic_per_seed(tmp_path):
    cfg, docs, _, _ = unrelated_setup(tmp_path)
    ids = lambda s: [t.window_id for t in load_unrelated_twins(docs, cfg, s)[0]]
    assert ids(7) == ids(7)
    assert any(ids(s) != ids(7) for s in range(8, 20))


def test_shared_window_maps_to_one_unrelated_window(tmp_path):
    cfg, docs, _, _ = unrelated_setup(tmp_path)
    twins, _ = load_unrelated_twins(docs, cfg, seed=3)
    assert twins[0].window_id == twins[1].window_id       # a, b share A_w0
    assert twins[0].sentences == twins[1].sentences
    assert twins[0].window_id != twins[2].window_id


def test_unrelated_weights_equal_matched_weights(tmp_path):
    cfg, docs, _, _ = unrelated_setup(tmp_path)
    _, w_unrel = load_unrelated_twins(docs, cfg, seed=3)
    _, w_twin = load_twins(docs, cfg)
    assert w_unrel == w_twin == [0.5, 0.5, 1.0]


def test_unrelated_shortage_is_an_error_not_a_length_change(tmp_path):
    cfg, _, _, _ = unrelated_setup(tmp_path, n_spare=0)
    five = [doc(f"d{i}", ["x."] * 5, [0, 1, 0, 0, 0], f"W{i}")
            for i in range(3)]
    with pytest.raises(ValueError, match="2 missing"):
        load_unrelated_twins(five, cfg, seed=0)


def test_unrelated_human_test_documents(tmp_path):
    from data import load_unrelated_human
    cfg, _, _, _ = unrelated_setup(tmp_path)
    human = load_unrelated_human(cfg, "test")
    assert [d.source_id for d in human] == ["T1", "T2"]
    assert all(d.labels == [0] * d.n and d.n > 0 for d in human)


def test_unrelated_source_needs_zero_paired_weights():
    import subprocess
    import sys
    run = ROOT / "src" / "detect" / "run_transformer.py"
    r = subprocess.run([sys.executable, str(run), "--twins",
                        "--twin-source", "unrelated"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 2 and "--margin-weight 0" in r.stderr


# ------------------------------------------------ metrics and normalization
def test_human_doc_metrics():
    from metrics import human_doc_metrics
    m = human_doc_metrics([[0, 0, 0], [0, 1, 0], [1, 1, 0, 0]])
    assert m["doc_false_alarm"] == pytest.approx(2 / 3)
    assert m["boundaries_per_doc"] == pytest.approx(3 / 3)
    assert m["sentence_fpr"] == pytest.approx(3 / 10)


def test_normalize_surface():
    from data import normalize_surface
    zwj = "ශ්‍රී"                       # conjunct spelling: ZWJ stays
    assert normalize_surface(zwj + " .") == zwj + "."
    assert normalize_surface("a​‌⁠﻿­b ,c ; d !") \
        == "ab,c; d!"
    assert normalize_surface("“x” ‘y’") == "\"x\" 'y'."
    for end in ("?", "!", "෴", "…", "."):
        assert normalize_surface("x" + end) == "x" + end
    assert normalize_surface("no stop ") == "no stop."
    assert normalize_surface("(1990) Latin") == "(1990) Latin."
