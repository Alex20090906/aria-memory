"""aria index: keys, block-orthogonal codes, query, save/load, forget; parity with the research where the real
model vocabulary is present (data/aria_vocab, or ARIA_VOCAB)."""
import os
from pathlib import Path

import numpy as np
import pytest

from aria_memory import NetIndex, Vocab, key_text
from conftest import LLM_SMALL, small_vocab
from aria_memory.eval import ScriptSet
from aria_memory.replay import index_recall

FIX = Path(__file__).parent / "fixtures"
REAL = Path(os.environ.get("ARIA_VOCAB", Path(__file__).resolve().parents[1] / "data" / "aria_vocab"))


@pytest.fixture(scope="module")
def vocab():
    return small_vocab()


def rec(i, thing, aspect, value, heard, day=1):
    return {"id": i, "thing": thing, "aspect": aspect, "value": value, "heard": heard, "day": day, "src": "owner"}


RECS = [
    rec(0, "護照", "location", "書房防潮箱", "我護照收進書房防潮箱了"),
    rec(1, "太陽眼鏡", "location", "玄關鞋櫃最上面", "太陽眼鏡改放玄關鞋櫃最上面"),
    rec(2, "牙醫", "time", "第3天下午2點半", "我約後天下午兩點半看牙醫", 1),
    rec(3, "咖啡", "preference", "冰燕麥拿鐵", "我現在喝咖啡都改冰燕麥拿鐵", 2),
    rec(4, "護照", "location", "黑色後背包夾層", "護照我改放黑色後背包夾層了", 3),
]


def build(vocab, recs=RECS):
    ix = NetIndex(vocab)
    for r in recs:
        ix.add(r["id"], r)
    return ix


@pytest.mark.skipif(not LLM_SMALL, reason="model vocabulary sample not present")
def test_vocab_fixture_loads(vocab):
    ids = vocab.token_ids("我，護照咧？")
    assert len(ids) == 6
    assert len(vocab.units(ids[0])) == 4            # 我 is one model token with real meaning units
    assert len(vocab.units(ids[1])) == 0            # punctuation has none


def test_key_has_meaning_and_sound_units(vocab):
    ix = NetIndex(vocab)
    cnt = ix.raw("護照 放在哪 書房防潮箱")
    assert any(u < 32768 for u in cnt) and any(u >= 32768 for u in cnt)
    u, w = ix.sparse(cnt)
    assert len(u) <= 128 and abs(float(np.linalg.norm(w)) - 1) < 1e-5


def test_codes_are_block_orthogonal(vocab):
    ix = NetIndex(vocab, dim=64)
    a = ix.codes(range(64))
    assert np.allclose(a @ a.T, np.eye(64), atol=1e-5)
    b = ix.codes(range(64, 128))
    assert np.allclose(b @ b.T, np.eye(64), atol=1e-5)
    assert not np.allclose(a, b)
    ix2 = NetIndex(vocab, dim=64)
    assert np.array_equal(ix2.code(70), ix.code(70))       # codes depend only on position


def test_query_finds_the_record_and_the_newest_version(vocab):
    ix = build(vocab)
    top = [i for i, _ in ix.query("我太陽眼鏡放哪", k=1)]
    assert top == [1]
    top5 = [i for i, _ in ix.query("Aria我護照咧", k=5)]
    assert set(top5[:2]) == {0, 4}
    assert [i for i, _ in ix.query("Aria我護照咧", k=5, exclude={0, 4})][0] not in (0, 4)
    sc = dict(ix.scores("護照", [0, 3, 99]))
    assert set(sc) == {0, 3} and sc[0] > sc[3]


def test_save_load_is_lossless(vocab, tmp_path):
    ix = build(vocab, RECS[:3])
    p = tmp_path / "index.npz"
    ix.save(p)
    ix2 = NetIndex(vocab).load(p)
    for r in RECS[3:]:
        ix.add(r["id"], r)
        ix2.add(r["id"], r)
    assert np.array_equal(ix.W, ix2.W) and ix.ids == ix2.ids
    for q in ("我護照咧", "咖啡喝什麼", "牙醫幾點"):
        assert ix.query(q) == ix2.query(q)
    assert not list(tmp_path.glob("*.tmp.npz"))


def test_string_ids_survive_save_load(vocab, tmp_path):
    ix = NetIndex(vocab)
    ix.add("r-1", RECS[0])
    ix.save(tmp_path / "i.npz")
    assert NetIndex(vocab).load(tmp_path / "i.npz").ids == ["r-1"]


def test_forget_really_deletes(vocab):
    ix = build(vocab)
    ix.forget([1], RECS)
    fresh = build(vocab, [r for r in RECS if r["id"] != 1])
    assert np.array_equal(ix.W, fresh.W) and np.array_equal(ix.n_use, fresh.n_use)
    assert 1 not in ix.ids and all(i != 1 for i, _ in ix.query("太陽眼鏡", k=10))


def test_add_twice_is_refused(vocab):
    ix = build(vocab, RECS[:1])
    with pytest.raises(ValueError):
        ix.add(0, RECS[0])


def test_key_text_is_e49s():
    assert key_text(RECS[0]) == "護照 放在哪 書房防潮箱 我護照收進書房防潮箱了"
    assert key_text(dict(RECS[0], heard="")) == "護照 放在哪 書房防潮箱 (sleep)"


@pytest.mark.skipif(not (REAL / "token_units.npy").exists(), reason="real model vocabulary not present")
class TestRealVocab:
    def test_tokenizer_matches_the_research(self):
        v = Vocab.load(REAL)
        ref = set(np.load(FIX / "e49_token_ids.npy").tolist())
        s = ScriptSet(FIX / "set_old")
        ids = set()
        for w in s.cache["writes"]:
            ids.update(v.token_ids(f"{w['fact'][0]} {w['fact'][2]} {w.get('heard') or ''}"))
        # the public fixtures name the owner role 使用者, a word the research text did not use: leave its tokens out
        assert ids and ids - set(v.token_ids("使用者")) <= ref

    def test_index_recall_matches_the_research(self):
        v = Vocab.load(REAL)
        assert index_recall(ScriptSet(FIX / "set_old"), v)["top5"] == 35        # H41: 35/38
        assert index_recall(ScriptSet(FIX / "set2"), v)["top5"] == 34           # H41: 34/38
