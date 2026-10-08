"""Word-chain extraction for the MULTIWORD pass."""

from collections import Counter

import numpy as np
import pytest

from supratok import chains
from supratok.chains import runs, unique_runs


def _reference_runs(joins, ok):
    out, run = [], []
    for i in range(len(ok)):
        if ok[i] and run and joins[i]:
            run.append(i)
            continue
        if len(run) > 1:
            out.append((run[0], len(run)))
        run = [i] if ok[i] else []
    if len(run) > 1:
        out.append((run[0], len(run)))
    return out


@pytest.mark.parametrize("seed", range(6))
def test_runs_match_a_reference_loop(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(0, 500))
    joins, ok = rng.random(n) < 0.7, rng.random(n) < 0.8
    start, length = runs(joins, ok)
    assert list(zip(start.tolist(), length.tolist())) == _reference_runs(joins, ok)


def test_a_lone_word_is_not_a_run():
    assert [len(a) for a in runs(np.zeros(0, bool), np.zeros(0, bool))] == [0, 0]
    start, _ = runs(np.array([False, True, True]), np.array([True, False, True]))
    assert len(start) == 0


def test_unique_runs_count_every_distinct_run_exactly():
    rng = np.random.default_rng(0)
    ids = rng.integers(0, 4, 5000).astype(np.int32)
    start, length = runs(rng.random(5000) < 0.8, rng.random(5000) < 0.85)
    rep, rep_len, count = unique_runs(ids, start, length)
    expected = Counter(tuple(ids[a:a + n].tolist()) for a, n in zip(start, length))
    got = {tuple(ids[a:a + n].tolist()): int(c) for a, n, c in zip(rep, rep_len, count)}
    assert got == expected and int(count.sum()) == len(start)


def test_unique_runs_tell_order_and_length_apart():
    ids = np.array([1, 2, 2, 1, 1, 2, 1, 1, 2, 2], np.int32)
    rep, rep_len, count = unique_runs(ids, np.array([0, 2, 4, 6]), np.array([2, 2, 2, 4]))
    found = sorted((tuple(ids[a:a + n].tolist()), int(c)) for a, n, c in zip(rep, rep_len, count))
    assert found == [((1, 1, 2, 2), 1), ((1, 2), 2), ((2, 1), 1)]


def test_unique_runs_of_nothing():
    empty = np.zeros(0, np.int64)
    assert [len(a) for a in unique_runs(np.zeros(3, np.int32), empty, empty)] == [0, 0, 0]


def test_hashes_tell_every_short_run_apart():
    """Every run of two ids below 1024 gets its own value under each hash: a weak mix of
    small ids is what lets two distinct chains collide."""
    a, b = np.meshgrid(np.arange(1024, dtype=np.int32), np.arange(1024, dtype=np.int32),
                       indexing="ij")
    ids = np.stack([a.ravel(), b.ravel()], 1).ravel()
    start, length = np.arange(0, len(ids), 2), np.full(len(ids) // 2, 2)
    for mult, salt in ((chains._M1, chains._S1), (chains._M2, chains._S2),
                       (chains._M3, chains._S3)):
        assert len(np.unique(chains._hash(ids, start, length, mult, salt))) == len(start)


def _forced(monkeypatch, *constant):
    real = chains._hash

    def hashed(ids, start, length, mult, salt):
        if mult in constant:
            return np.zeros(len(start), np.uint64)
        return real(ids, start, length, mult, salt)

    monkeypatch.setattr(chains, "_hash", hashed)


def test_unique_runs_survive_a_collision_in_one_hash(monkeypatch):
    """Runs are grouped by two hashes together, so one colliding still counts exactly."""
    _forced(monkeypatch, chains._M1)
    ids = np.array([1, 2, 3, 4, 1, 2], np.int32)
    rep, rep_len, count = unique_runs(ids, np.array([0, 2, 4]), np.array([2, 2, 2]))
    found = sorted((tuple(ids[a:a + n].tolist()), int(c)) for a, n, c in zip(rep, rep_len, count))
    assert found == [((1, 2), 2), ((3, 4), 1)]


def test_unique_runs_detect_a_collision_in_both_hashes(monkeypatch):
    """A third hash must agree inside every group, so a collision in both grouping hashes
    raises instead of counting two runs as one."""
    _forced(monkeypatch, chains._M1, chains._M2)
    with pytest.raises(RuntimeError):
        unique_runs(np.array([1, 2, 3, 4], np.int32), np.array([0, 2]), np.array([2, 2]))


def test_windows_cut_long_runs_and_keep_short_ones():
    start, length = np.array([0, 20]), np.array([3, 14])
    w_start, w_len = chains.windows(start, length, 6)
    assert w_start.tolist() == [0, 20, 26, 32]
    assert w_len.tolist() == [3, 6, 6, 2]
    assert w_len.sum() == length.sum()


def test_windows_of_nothing():
    w_start, w_len = chains.windows(np.zeros(0, np.int64), np.zeros(0, np.int64), 6)
    assert len(w_start) == len(w_len) == 0
