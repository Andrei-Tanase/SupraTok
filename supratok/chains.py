"""Runs of corpus positions for MULTIWORD and the sizing run: maximal runs, windows, and
the count of each distinct run.

Identical runs are grouped by two 64-bit hashes together; a third must agree inside every
group, or training stops with an error, so two distinct runs are counted as one only if all
three hashes collide.
"""
from __future__ import annotations

import numpy as np

# Arbitrary odd multipliers and salts, one pair per hash.
_M1 = np.uint64(0x9E3779B97F4A7C15)
_M2 = np.uint64(0xC2B2AE3D27D4EB4F)
_M3 = np.uint64(0xD6E8FEB86659FD93)
_S1 = np.uint64(0x5BD1E995)
_S2 = np.uint64(0x27D4EB2F)
_S3 = np.uint64(0x165667B1)
# The splitmix64 finalizer's constants.
_F1 = np.uint64(0xBF58476D1CE4E5B9)
_F2 = np.uint64(0x94D049BB133111EB)


def runs(joins: np.ndarray, ok: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start positions and lengths of the maximal runs of two or more `ok` positions in
    which each position after the first `joins` its predecessor."""
    link = np.zeros(len(ok), bool)
    link[1:] = joins[1:] & ok[1:] & ok[:-1]
    start = np.flatnonzero(ok & ~link)
    breaks = np.flatnonzero(~link)
    end = np.append(breaks, len(ok))[np.searchsorted(breaks, start, side="right")]
    length = end - start
    keep = length >= 2
    return start[keep], length[keep]


def windows(start: np.ndarray, length: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Runs cut into consecutive windows of at most `size` positions."""
    count = (length + size - 1) // size
    step = size * (np.arange(int(count.sum())) - np.repeat(np.cumsum(count) - count, count))
    return np.repeat(start, count) + step, np.minimum(size, np.repeat(length, count) - step)


def _hash(ids: np.ndarray, start: np.ndarray, length: np.ndarray, mult: np.uint64,
          salt: np.uint64) -> np.ndarray:
    """A 64-bit hash of each run: every id is combined with its position and passed
    through the splitmix64 finalizer, and the results are summed with a term for the run's
    length."""
    offset = np.cumsum(length) - length
    pos = (np.arange(int(length.sum())) - np.repeat(offset, length)).astype(np.uint64)
    values = ids[np.repeat(start, length) + pos.astype(np.int64)].astype(np.uint64)
    with np.errstate(over="ignore"):
        mixed = values * mult + (pos + np.uint64(1)) * (mult ^ salt) + salt
        mixed ^= mixed >> np.uint64(30)
        mixed *= _F1
        mixed ^= mixed >> np.uint64(27)
        mixed *= _F2
        mixed ^= mixed >> np.uint64(31)
        return np.add.reduceat(mixed, offset) + length.astype(np.uint64) * salt


def unique_runs(ids: np.ndarray, start: np.ndarray,
                length: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One representative start and length per distinct run, and how often it occurs."""
    if not len(start):
        return start, length, np.zeros(0, np.int64)
    first = _hash(ids, start, length, _M1, _S1)
    second = _hash(ids, start, length, _M2, _S2)
    order = np.lexsort((second, first))
    first, second = first[order], second[order]
    new = np.ones(len(order), bool)
    new[1:] = (first[1:] != first[:-1]) | (second[1:] != second[:-1])
    head = np.flatnonzero(new)
    count = np.diff(np.append(head, len(order)))
    third = _hash(ids, start, length, _M3, _S3)[order]
    if not np.array_equal(third, np.repeat(third[head], count)):
        raise RuntimeError("two distinct word chains share a 128-bit hash")
    rep = order[head]
    return start[rep], length[rep], count
