"""Counterfactual twins: construction, pairing and the paired loss."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

torch = pytest.importorskip("torch")
from data import Doc, load_twins  # noqa: E402
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
