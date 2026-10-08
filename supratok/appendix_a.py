"""Appendix A span scoring and leaf-surgery acceptance (Eqs. 5-14).

Pure functions over already-collected counts; the trainer computes the counts and calls
these for the EXPRESSION pass. `pmi`, `entropy` and `peak` are the scalar forms of
Eqs. 5-7 that the tests check the vectorized statistics against.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

# Eq. 8 boundary gate g_B = sigma(BOUNDARY_GATE_SLOPE, H_B(s) - TAU_BE).
TAU_BE = 1.15
BOUNDARY_GATE_SLOPE = 1.75
# Hard rejections before leaf surgery.
PMI_FLOOR = 4.4
MIN_COUNT = 3
BOUNDARY_GATE_FLOOR = 0.18
STOPWORD_CEILING = 0.75
SCORE_FLOOR = 0.40
# Eq. 12, Eq. 13 and Eq. 14 weights.
NO_HOLDOUT_PENALTY = 0.45
LEAF_DF_WEIGHT = 0.35
PRIORITY_SCORE_WEIGHT = 0.03


def sigmoid(k: float, x: float) -> float:
    """sigma(k, x) of Eq. 8."""
    return 1.0 / (1.0 + math.exp(-k * x))


def pmi(count: int, total_ngrams: int, unigram_probs: Sequence[float]) -> tuple[float, float]:
    """Eq. 5: PMI and NPMI of a span, base 2."""
    joint = count / max(1, total_ngrams)
    denom = 1.0
    for p in unigram_probs:
        if p <= 0:
            return -math.inf, -math.inf
        denom *= p
    if denom <= 0 or joint <= 0:
        return -math.inf, -math.inf
    value = math.log2(joint / denom)
    # A span that is every n-gram occurrence (joint = 1) is perfectly cohesive: NPMI 1.
    return value, value / -math.log2(joint) if joint < 1 else 1.0


def entropy(counter: dict[str, int]) -> float:
    """Eq. 6 branching entropy, base 2."""
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    out = 0.0
    for c in counter.values():
        p = c / total
        if p > 0:
            out -= p * math.log2(p)
    return out


def peak(counter: dict[str, int]) -> float:
    """Eq. 7: the largest context probability on one side; c_peak takes the larger side."""
    total = sum(counter.values())
    return (max(counter.values()) / total) if total > 0 else 0.0


def score(*, gain: int, count: int, holdout_count: int, holdout_df: int,
          npmi: float, boundary_entropy: float, avg_idf: float,
          internal_npmi: float, context_peak: float, stopword_ratio: float,
          n_words: int, avg_word_len: float, char_len: int) -> float:
    """Eqs. 8-12: the multiplicative gated score S(s), with the rejections g_B < 0.18 and
    S(s) <= 0.40; 0.0 means rejected. `gain` is g_tok(s) = l_tok(s) - 1 (Eq. 7)."""
    # Eq. 8
    g_b = sigmoid(BOUNDARY_GATE_SLOPE, boundary_entropy - TAU_BE)
    if g_b < BOUNDARY_GATE_FLOOR:
        return 0.0
    g_n = sigmoid(6.0, npmi - 0.22)
    b_spec = 1.0 + 0.14 * max(0.0, avg_idf - 1.0)
    b_con = 1.0 + 0.50 * max(0.0, internal_npmi - context_peak)
    # Eq. 9
    p_stop = max(0.30, 1.0 - 0.78 * (stopword_ratio ** 1.35))
    b_len = 1.0 + 0.08 * max(0, n_words - 2)
    b_cont = 1.0 + 0.04 * max(0.0, avg_word_len - 5.0)
    # Eqs. 10-12
    p_surf = 0.88 if char_len > 42 else 1.0
    g_hold = gain * (holdout_count + 0.60 * holdout_df + 0.18 * math.sqrt(max(count, 0)))
    s = g_hold * g_n * g_b * b_spec * b_con * p_stop * b_len * b_cont * p_surf
    if holdout_count <= 0:
        s *= NO_HOLDOUT_PENALTY
    return 0.0 if s <= SCORE_FLOOR else s


def addition_gain(holdout_count: int, token_length: int, holdout_df: int) -> float:
    """Eq. 13: A(s)."""
    return holdout_count * max(1, token_length - 1) + LEAF_DF_WEIGHT * holdout_df


def leaf_utility(eval_count: int, eval_df: int) -> float:
    """Eq. 13: U(m)."""
    return eval_count + LEAF_DF_WEIGHT * eval_df


def priority(net: float, s: float) -> float:
    """Eq. 14: Priority(s)."""
    return net + PRIORITY_SCORE_WEIGHT * s
