"""Arabic hamza normalization (paper Section 3.4.1 / Appendix E).

Folds hamza-carrier letters to their bare carriers before frequency counting.
This is deliberately lossy: normalized text does not round-trip to its original
orthography.

INTERPRETIVE: the paper names hamza normalization without defining it, so the map covers
hamza forms only: hamzated alef forms (including madda and wasla) fold to bare alef, and
hamza on waw or yeh folds to its carrier. Teh marbuta and alef maksura folding are separate
normalizations and are not applied, and the standalone hamza letter is kept.
"""

from __future__ import annotations

import regex
from tokenizers import Regex, normalizers

# (variant, canonical form)
HAMZA_MAP = [
    ("أ", "ا"),  # أ ALEF WITH HAMZA ABOVE -> ا ALEF
    ("إ", "ا"),  # إ ALEF WITH HAMZA BELOW -> ا ALEF
    ("آ", "ا"),  # آ ALEF WITH MADDA ABOVE -> ا ALEF
    ("ٱ", "ا"),  # ٱ ALEF WASLA            -> ا ALEF
    ("ؤ", "و"),  # ؤ WAW WITH HAMZA ABOVE  -> و WAW
    ("ئ", "ي"),  # ئ YEH WITH HAMZA ABOVE  -> ي YEH
]

# The same letters written decomposed (NFD): the carrier, then the combining hamza or madda.
# Canonical order puts other Arabic marks of a lower combining class (harakat, superscript
# alef, Quranic marks) before it, so the hamza mark is dropped after up to four of them.
# Python's `regex` and the tokenizers' Oniguruma read these patterns alike.
_MARKS = ("[\u0610-\u061A\u064B-\u0652\u0656-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4"
          "\u06E7\u06E8\u06EA-\u06ED\u08D3-\u08E1\u08E3-\u08FF]")
DECOMPOSED = [
    f"(?<=\u0627{_MARKS}{{0,4}})[\u0653-\u0655]",  # alef + madda / hamza above / below
    f"(?<=[\u0648\u064A]{_MARKS}{{0,4}})\u0654",   # waw, yeh + hamza above
]
_DECOMPOSED = [regex.compile(pattern) for pattern in DECOMPOSED]


def normalize_hamza(text: str) -> str:
    """Apply the hamza folding to a string."""
    for variant, canonical in HAMZA_MAP:
        text = text.replace(variant, canonical)
    for pattern in _DECOMPOSED:
        text = pattern.sub("", text)
    return text


def hamza_normalizer() -> normalizers.Sequence:
    """A tokenizers normalizer applying the same folding."""
    return normalizers.Sequence(
        [normalizers.Replace(variant, canonical) for variant, canonical in HAMZA_MAP]
        + [normalizers.Replace(Regex(pattern), "") for pattern in DECOMPOSED]
    )
