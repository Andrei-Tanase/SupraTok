"""Script-aware processing (paper Section 3.4 / 3.4.1 / Appendix E).

Three mechanisms, one per language group in the 3.4.1 ablation:

  zh/ja/th : thresholded character-bigram gating (min count 50) at merge junctions
  tr/fi    : suffix-aware weighting at boundaries matched against 200 mined suffixes
  ar       : hamza normalization before frequency counting (re-exported from arabic.py)

plus the five Unicode supercategories (LM / PS / N / Z / C) they share.

Lines marked INTERPRETIVE are choices the paper leaves undefined.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from collections.abc import Iterable, Sequence
from itertools import repeat

import regex

from .arabic import HAMZA_MAP, hamza_normalizer, normalize_hamza

__all__ = [
    "BOUNDARY_FREE", "HAMZA_MAP", "MIN_BIGRAM_COUNT", "SUFFIX_LANGUAGES", "SUFFIX_TOP_K",
    "SUPERCATEGORIES", "BigramGate", "SuffixMatcher", "classify", "fold", "gatable",
    "hamza_normalizer", "has_boundary_free", "is_boundary_free", "mine_suffixes",
    "normalize_hamza", "script_of", "suffixes_from_types",
]

SUPERCATEGORIES = ("LM", "PS", "N", "Z", "C")

# ---------------------------------------------------------------- supercategories

# General categories L,M -> LM; P,S -> PS; N; Z; everything else C. They are read through
# `regex`, the library and Unicode version of the trainer's pre-tokenizer, not Python's
# unicodedata, whose tables lag behind and vary with the Python version.
# INTERPRETIVE: the controls with the Unicode White_Space property are separators for a
# tokenizer, so they map to Z, as the trainer's pre-tokenizer treats them.
_SUPERCATEGORY_RES = (("LM", regex.compile(r"[\p{L}\p{M}]")),
                      ("PS", regex.compile(r"[\p{P}\p{S}]")),
                      ("N", regex.compile(r"\p{N}")),
                      ("Z", regex.compile(r"[\p{Z}\t\n\v\f\r\x85]")))


def _supercategory(ch: str) -> str:
    for name, pattern in _SUPERCATEGORY_RES:
        if pattern.fullmatch(ch):
            return name
    return "C"


# Memo table: seeded with ASCII, grown on demand. Bounded by the Unicode space.
_CAT_CACHE: dict[str, str] = {chr(i): _supercategory(chr(i)) for i in range(128)}


def classify(ch: str) -> str:
    """Supercategory of a single character: LM, PS, N, Z or C."""
    try:
        return _CAT_CACHE[ch]
    except KeyError:
        value = _CAT_CACHE[ch] = _supercategory(ch)
        return value


# ---------------------------------------------------------------- scripts

# INTERPRETIVE: script detection is scoped to the gated scripts and Hangul; everything
# else is "Other". Common/Inherited characters that only occur inside a host script
# (chouonpu U+30FC / U+FF70, iteration marks, kana voicing marks) sit in its range.
_SCRIPT_RANGES = (
    (0x0E01, 0x0E5B, "Thai"),
    (0x1100, 0x11FF, "Hangul"),
    (0x2E80, 0x2EFF, "Han"), (0x2F00, 0x2FDF, "Han"),
    (0x3005, 0x3005, "Han"), (0x3007, 0x3007, "Han"),
    (0x3021, 0x3029, "Han"), (0x3038, 0x303B, "Han"),
    (0x3041, 0x309F, "Hiragana"),
    (0x30A0, 0x30FF, "Katakana"),
    (0x3130, 0x318F, "Hangul"),
    (0x31F0, 0x31FF, "Katakana"),
    (0x3400, 0x4DBF, "Han"), (0x4E00, 0x9FFF, "Han"),
    (0xA960, 0xA97F, "Hangul"), (0xAC00, 0xD7A3, "Hangul"), (0xD7B0, 0xD7FF, "Hangul"),
    (0xF900, 0xFAFF, "Han"),
    (0xFF66, 0xFF9F, "Katakana"),
    (0x20000, 0x2A6DF, "Han"), (0x2A700, 0x2EE5F, "Han"),
    (0x2F800, 0x2FA1F, "Han"), (0x30000, 0x323AF, "Han"),
)
_SCRIPT_STARTS = [start for start, _, _ in _SCRIPT_RANGES]

# INTERPRETIVE: Korean is in the 3.4.1 ablation, but Section 3.4 and Appendix E describe
# gating only for zh/ja/th, and Hangul text is space-delimited into eojeol, so Hangul is
# not boundary-free and Korean is not gated.
BOUNDARY_FREE = frozenset({"Han", "Hiragana", "Katakana", "Thai"})

_BOUNDARY_FREE_RE = regex.compile("[" + "".join(
    f"{chr(start)}-{chr(end)}" for start, end, name in _SCRIPT_RANGES
    if name in BOUNDARY_FREE) + "]")


def script_of(ch: str) -> str:
    """Script name of a single character, or 'Other' outside the tracked scripts."""
    cp = ord(ch)
    i = bisect_right(_SCRIPT_STARTS, cp) - 1
    if i >= 0:
        _, end, name = _SCRIPT_RANGES[i]
        if cp <= end:
            return name
    return "Other"


_BF_CACHE: dict[str, bool] = {}


def is_boundary_free(script: str) -> bool:
    """True for scripts written without explicit word boundaries.

    Accepts a script name ('Han', 'thai') or, for convenience in hot loops, a
    single character, whose script is resolved first.
    """
    if len(script) == 1:
        try:
            return _BF_CACHE[script]
        except KeyError:
            value = _BF_CACHE[script] = script_of(script) in BOUNDARY_FREE
            return value
    return script.title() in BOUNDARY_FREE


def has_boundary_free(text: str) -> bool:
    """True if any character of `text` belongs to a boundary-free script."""
    return _BOUNDARY_FREE_RE.search(text) is not None


def gatable(ch: str) -> bool:
    """True if a character takes part in bigram gating.

    INTERPRETIVE: restricted to LM characters of boundary-free scripts, so script
    punctuation and digits (U+30FB, U+0E4F, Thai digits) stay ungated.
    """
    return is_boundary_free(ch) and classify(ch) == "LM"


# ---------------------------------------------------------------- bigram gating

MIN_BIGRAM_COUNT = 50  # Section 3.4.1 / Appendix E


class BigramGate:
    """Thresholded character-bigram gate for boundary-free scripts.

    Counts adjacent character pairs seen inside boundary-free runs, then refuses a
    merge whose junction bigram was seen fewer than `min_count` times.
    """

    def __init__(self, counts: dict[str, int] | None = None,
                 min_count: int = MIN_BIGRAM_COUNT):
        self.counts: Counter = Counter(counts or {})
        self.min_count = min_count

    def fit(self, texts: Iterable[str],
            weights: Iterable[int] | None = None) -> BigramGate:
        """Count character bigrams over a corpus pass, each text counted `weight` times.
        Returns self."""
        counts = self.counts
        for text, weight in zip(texts, repeat(1) if weights is None else weights):
            prev, prev_ok = "", False
            for ch in text:
                ok = gatable(ch)
                # INTERPRETIVE: a bigram is counted only when both sides are gatable;
                # scripts are not required to match, since Japanese mixes kana and han
                # word-internally.
                if ok and prev_ok:
                    counts[prev + ch] += weight
                prev, prev_ok = ch, ok
        return self

    def allows(self, left_token: str, right_token: str) -> bool:
        """Whether the merge left + right may fire: its junction bigram, the last character
        of `left_token` and the first of `right_token`, must have been seen `min_count`
        times. A junction with a character that is not gatable passes, so a right-hand
        token opening with a space is never gated."""
        if not left_token or not right_token:
            return True
        a, b = left_token[-1], right_token[0]
        if not (gatable(a) and gatable(b)):
            return True
        return self.counts.get(a + b, 0) >= self.min_count

    def split(self, text: str) -> list[str]:
        """Cut `text` at every junction the gate refuses, so no merge can cross it."""
        out, start = [], 0
        for i in range(1, len(text)):
            if not self.allows(text[i - 1], text[i]):
                out.append(text[start:i])
                start = i
        out.append(text[start:])
        return out

    def __len__(self) -> int:
        return len(self.counts)


# ---------------------------------------------------------------- suffix mining

SUFFIX_TOP_K = 200  # Section 3.4.1 / Appendix E
SUFFIX_LANGUAGES = frozenset({"tr", "fi"})
SUFFIX_LENS = (2, 3, 4, 5)  # INTERPRETIVE: tr/fi suffixes are 2-5 chars; 1 is noise.
MIN_STEM = 2  # INTERPRETIVE: shortest residue that can still be a root ("ev", "ne").
_WORD_RE = regex.compile(r"\p{L}[\p{L}\p{M}]*")


def fold(word: str, lang: str) -> str:
    """Lowercase `word`, with Turkish dotted/dotless I, which str.lower gets wrong."""
    if lang == "tr":
        word = word.replace("İ", "i").replace("I\u0307", "i").replace("I", "ı")
    return word.lower()


def _word_types(texts: Iterable[str], lang: str) -> Counter:
    types: Counter = Counter()
    for text in texts:
        for word in _WORD_RE.findall(text):
            types[fold(word, lang)] += 1
    return types


def _drop_redundant(ranked: Sequence[str], totals: Counter, cover: float) -> list[str]:
    """Drop a suffix whose types are almost all explained by a longer kept one."""
    kept: list[str] = []
    for s in ranked:
        if any(len(t) > len(s) and t.endswith(s) and totals[t] >= cover * totals[s]
               for t in kept):
            continue
        kept.append(s)
    return kept


def suffixes_from_types(words: Iterable[str], top_k: int = SUFFIX_TOP_K,
                        min_types: int = 2, cover: float = 0.9) -> list[str]:
    """Up to `top_k` suffix candidates from folded word types.

    A suffix is counted once per word type, not per token, so its count measures
    productivity rather than the frequency of a few function words. Candidates rank first
    by how many types leave an attested word when stripped of them (Harris-style boundary
    evidence), which also demotes over-long endings such as -alar; a suffix whose types are
    almost all covered by a longer kept one is dropped.
    """
    vocab = set(words)
    stems: Counter = Counter()   # types whose residue is itself a word
    totals: Counter = Counter()  # types ending in the suffix at all
    for word in vocab:
        for length in SUFFIX_LENS:
            if len(word) - length < MIN_STEM:
                break
            suffix = word[-length:]
            totals[suffix] += 1
            if word[:-length] in vocab:
                stems[suffix] += 1
    ranked = sorted((s for s, n in totals.items() if n >= min_types),
                    key=lambda s: (-stems[s], -totals[s], s))
    return _drop_redundant(ranked, totals, cover)[:top_k]


def mine_suffixes(texts: Iterable[str], lang: str, top_k: int = SUFFIX_TOP_K,
                  min_types: int = 2, cover: float = 0.9) -> list[str]:
    """Heuristic suffix candidates for Turkish / Finnish mined from raw texts."""
    return suffixes_from_types(_word_types(texts, lang), top_k, min_types, cover)


class SuffixMatcher:
    """Finds the likely morphological boundary of Section 3.4: the longest mined suffix
    ending a word with a stem of at least MIN_STEM characters."""

    def __init__(self, suffixes: Iterable[str], lang: str):
        self.suffixes = frozenset(suffixes)
        self.lang = lang
        self._lens = sorted({len(s) for s in self.suffixes}, reverse=True)

    def __call__(self, word: str) -> str:
        """The matched suffix in the word's own spelling, or '' if none matches."""
        core = word.lstrip()
        folded = fold(core, self.lang)
        if len(folded) != len(core):
            return ""
        for length in self._lens:
            if len(folded) - length >= MIN_STEM and folded[-length:] in self.suffixes:
                return core[-length:]
        return ""
