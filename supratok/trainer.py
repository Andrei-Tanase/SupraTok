"""Figure 1 training procedure.

T starts from Unicode characters, byte-fallback symbols and the seed phrases mined on D'
that fit its cross-word share; three ordered passes then each continue from the vocabulary
the previous one left:

  TRADITIONAL  BPE inside one pre-token, a word with the space before it
  MULTIWORD    BPE resumed across Section 3.5-screened or PMI-seeded junctions, and
               between a word and what is attached to it, joining the pieces on either side
  EXPRESSION   2-5 word spans scored by Appendix A and accepted by leaf surgery

Corpus statistics are exact word n-gram counts; TRADITIONAL and MULTIWORD merges are learned
by the `tokenizers` Rust trainer.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import numbers
import operator
from array import array
from bisect import bisect_right
from collections import Counter
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from itertools import accumulate, chain
from typing import NamedTuple

import numpy as np
import regex as re
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

from . import appendix_a as A
from . import chains
from . import script_aware as S
from .arabic import hamza_normalizer
from .arabic import normalize_hamza as _fold_hamza
from .data_loader import ThreeBinSampler
from .hf_build import (
    _BYTE_SPELLING,
    _BYTE_SPELLING_CHARS,
    BYTE_TOKENS,
    build_bpe_tokenizer,
    build_shortest_tokenizer,
)

logger = logging.getLogger(__name__)

# Section 3.4's supercategories as pre-tokens: a run of letters and marks (LM), numbers
# (N), punctuation and symbols (PS) or other characters (C), each with the space before
# it, or a whitespace run (Z). A no-break space (U+00A0, U+202F), which French and Russian
# text put before some punctuation, leads a pre-token as a space does. A PS run keeps the
# line breaks right after it (".\n\n"), as GPT-4's and SuperBPE's pre-tokenizers do. An
# English contraction with a straight apostrophe ('s, 't, 're, 've, 'm, 'll, 'd) ending a
# word is one pre-token; any other apostrophe between letters stays inside the word
# ("l'homme", "dell'anno", "Ankara'da"). Line breaks end a whitespace run, and otherwise a
# run leaves its last space to the word after it. Lossless: the alternation tiles the input.
_SPACES = frozenset(" \u00a0\u202f")
_LEAD = r"[ \u00a0\u202f]?"
_CONTRACTION = r"'(?:s|t|re|ve|m|ll|d)(?![\p{L}\p{M}])"
_CHUNK = re.compile(_CONTRACTION
                    + "|" + _LEAD + r"[\p{L}\p{M}]+(?:['\u2019](?!" + _CONTRACTION[1:]
                    + r")[\p{L}\p{M}]+)*"
                    + "|" + _LEAD + r"\p{N}+"
                    + "|" + _LEAD + r"[\p{P}\p{S}]+[\r\n]*"
                    + "|" + _LEAD + r"[^\s\p{L}\p{M}\p{N}\p{P}\p{S}]+"
                    + r"|\s*[\r\n]+|\s+(?!\S)|\s+")
# A word is any pre-token but whitespace, with the space before it: letters, numbers,
# punctuation or other characters; punctuation holding line breaks is not one, since a line
# break ends every phrase. Phrases join words separated by single spaces, so they follow
# Appendix A's whitespace-delimited words, except that a contraction or punctuation
# attached to a word is a word of its own ("York," is " York" and ","). Suffix mining keeps
# to letter runs.
_WORD = re.compile(_LEAD + r"\S+")
_LETTERS = re.compile(_LEAD + r"[\p{L}\p{M}]+")
_PUNCT = re.compile(_LEAD + r"[\p{P}\p{S}]+")
_CJK_PUNCT = re.compile(r"[\u3000-\u303F\uFF01-\uFF0F\uFF1A-\uFF20\uFF3B-\uFF40\uFF5B-\uFF65]")
# No pre-token holds whitespace other than line breaks after other material, so a token
# that does spans a pre-token boundary.
_CROSSES = re.compile(r"\S[^\S\r\n]")
_BLANK = re.compile(r"\s+")
_BYTE_SET = frozenset(BYTE_TOKENS)
# Ids that are not pre-tokens: SEP ends a document and BREAK stands where a special token
# was cut out of one. No merge or phrase crosses either.
SEP, BREAK = 0, 1


def _document(doc) -> tuple[str, str | None]:
    """A document's text and language tag; a document is a str or a (str, language) pair."""
    if isinstance(doc, str):
        return doc, None
    if (isinstance(doc, (tuple, list)) and len(doc) == 2 and isinstance(doc[0], str)
            and (doc[1] is None or isinstance(doc[1], str))):
        return doc[0], doc[1]
    raise TypeError(f"each document must be a str or a (str, language) pair, not "
                    f"{type(doc).__name__}")


def shard_of(text: str, fraction: float) -> str:
    """The Appendix A shard of a document, by a 64-bit hash of its text: the lowest
    `fraction` of the hash range is the held-out shard and the next `fraction` the
    verification shard. Membership depends on the text alone, so it ignores document order
    and duplicates share a shard."""
    if not fraction:
        return "train"
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    q = int.from_bytes(digest, "big") / (fraction * 2 ** 64)
    return "held-out" if q < 1 else "verification" if q < 2 else "train"

# EXPRESSION checks the verification shard once per this many spans.
EXPRESSION_CHUNK = 2000

# MULTIWORD trains a chain longer than this many words (a space-free run of punctuated
# words, such as a URL) in windows of that length: a token spans far fewer words, and the
# trainer's cost per occurrence grows with the length of the sequence it sits in.
MAX_CHAIN = 64

# A character enters T when it is at least one in this many training characters: a rarer
# one saves fewer tokens in T than a merge would in its slot, and byte fallback spells it.
RARE_CHARACTER = 5_000_000

STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has", "have", "he",
    "in", "is", "it", "its", "of", "on", "or", "that", "the", "to", "was", "were", "will",
    "with", "this", "these", "those", "they", "not", "no", "so", "if", "then", "than", "there",
    "their", "we", "you", "his", "her", "our", "your", "all", "can", "had", "do", "does", "did",
    "been", "being", "would", "could", "should",
})


class _Span(NamedTuple):
    """Appendix A statistics of one candidate span."""
    words: tuple[str, ...]
    train_count: int
    holdout_count: int
    holdout_df: int
    npmi: float
    boundary_entropy: float
    context_peak: float
    avg_idf: float
    internal_npmi: float
    stopword_ratio: float
    avg_word_len: float
    char_len: int


class _Mined(NamedTuple):
    seeds: dict[tuple[str, ...], int]
    bigrams: dict[tuple[str, ...], int]
    every: list[tuple[str, int]]
    plain: list[tuple[str, int]]
    units: list[tuple[str, int]]
    extras: list[tuple[str, int]]
    spans: list[_Span]
    # Position i joins the word before it across a junction MULTIWORD may cross.
    junctions: np.ndarray
    # Whether any word has something attached to it.
    has_attached: bool


class _Ids(dict):
    """Pre-token ids, numbered after SEP and BREAK in order of first lookup."""

    def __missing__(self, chunk: str) -> int:
        self[chunk] = i = len(self) + BREAK + 1
        return i


class SupraTokTrainer:
    """Builds a character-level, byte-fallback BPE vocabulary through the Figure 1
    curriculum. Documents are `str`, or `(str, language)` pairs for language-specific
    processing."""

    def __init__(
        self,
        vocab_size: int = 100000,
        min_frequency: int = 5,
        screening_pmi: float = 5.0,
        screening_frequency: int = 5,
        max_span: int = 5,
        special_tokens: Sequence[str] | None = None,
        seed_pmi_threshold: float = 2.0,
        seed_min_frequency: int = 100,
        max_seed_ngram: int = 6,
        seed_overlap: float = 0.5,
        holdout_fraction: float = 0.1,
        enable_seeds: bool = True,
        enable_multiword: bool = True,
        enable_expression: bool = True,
        use_entropy_filtering: bool = False,
        space_separated_stream: bool = False,
        script_aware: bool = True,
        suffix_weight: int = 1,
        normalize_hamza: bool = False,
        stopwords: Sequence[str] | None = None,
    ):
        if isinstance(special_tokens, (str, bytes)) or isinstance(stopwords, (str, bytes)):
            raise TypeError("special_tokens and stopwords take a sequence of strings")
        specials = list(special_tokens or [])
        if not all(isinstance(t, str) and t for t in specials):
            raise ValueError("special tokens must be non-empty strings")
        if " " in specials or any(_BYTE_SPELLING.fullmatch(t) or t in _BYTE_SPELLING_CHARS
                                  for t in specials):
            raise ValueError("a special token cannot be the space, one of the characters "
                             "that spell a byte-fallback token, or spelled like one")
        self.special_tokens = list(dict.fromkeys(specials))
        vocab_size = operator.index(vocab_size)
        base = len(BYTE_TOKENS) + len(_BYTE_SPELLING_CHARS) + 1
        if vocab_size - len(self.special_tokens) <= base:
            raise ValueError("vocab_size must leave room beyond the special tokens, the 256 "
                             "byte-fallback tokens and the 20 characters always in T")
        if not 2 <= max_seed_ngram <= 64 or not 2 <= max_span <= 64:
            raise ValueError("max_seed_ngram and max_span must be between 2 and 64")
        if isinstance(suffix_weight, bool) or not isinstance(suffix_weight, int):
            raise TypeError("suffix_weight must be an int")
        if suffix_weight < 0:
            raise ValueError("suffix_weight must not be negative")
        counts = {"min_frequency": min_frequency, "screening_frequency": screening_frequency,
                  "seed_min_frequency": seed_min_frequency}
        for name, value in counts.items():
            if isinstance(value, bool):
                raise TypeError(f"{name} must be an int")
            if operator.index(value) < 0:
                raise ValueError(f"{name} must not be negative")
        for name, value in (("screening_pmi", screening_pmi),
                            ("seed_pmi_threshold", seed_pmi_threshold)):
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise TypeError(f"{name} must be a number")
            if math.isnan(value):
                raise ValueError(f"{name} must not be NaN")
        if not 0.0 <= holdout_fraction < 0.5:
            raise ValueError("holdout_fraction must be in [0, 0.5)")
        if enable_expression and not holdout_fraction:
            raise ValueError("EXPRESSION measures utility on held-out shards; "
                             "holdout_fraction must be above 0")
        if not 0.0 < seed_overlap <= 1.0:
            raise ValueError("seed_overlap must be in (0, 1]")
        # The input is split at special tokens before the model runs, so the corpus is
        # split at them the same way; the group keeps them in the split.
        longest_first = sorted(self.special_tokens, key=len, reverse=True)
        self._specials = (re.compile("(" + "|".join(map(re.escape, longest_first)) + ")")
                          if longest_first else None)
        self.vocab_size = vocab_size
        self.min_frequency = operator.index(min_frequency)
        self.screening_pmi = screening_pmi
        self.screening_frequency = operator.index(screening_frequency)
        self.max_span = max_span
        self.seed_pmi_threshold = seed_pmi_threshold
        self.seed_min_frequency = operator.index(seed_min_frequency)
        self.max_seed_ngram = max_seed_ngram
        self.seed_overlap = seed_overlap
        self.holdout_fraction = holdout_fraction
        self.enable_seeds = enable_seeds
        self.enable_multiword = enable_multiword
        self.enable_expression = enable_expression
        self.use_entropy_filtering = use_entropy_filtering
        self.space_separated_stream = space_separated_stream
        self.script_aware = script_aware
        self.suffix_weight = suffix_weight
        self.normalize_hamza = normalize_hamza
        self._split = None
        self.stopwords = (STOPWORDS if stopwords is None
                          else frozenset(w.strip().lower() for w in stopwords))
        self.seeds: dict[tuple[str, ...], int] = {}
        self.alphabet: list[str] = []
        self.suffixes: dict[str, list[str]] = {}
        self.stats: dict[str, object] = {}
        # The corpus as read by `train`: pre-token strings, their ids, document languages,
        # the Appendix A shard of each document (0 train, 1 held-out, 2 verification) and
        # the texts of the held-out and verification documents.
        self.types: list[str] = []
        self.ids = np.zeros(0, np.int32)
        self.langs: list[str | None] = []
        self._shard = np.zeros(0, np.int8)
        self.heldout: list[str] = []
        self.verification: list[str] = []

    # ---------------------------------------------------------------- corpus

    def _curate(self, source):
        """Figure 1 line 1: Section 3.2 entropy-based document selection."""
        def curated():
            sampler = ThreeBinSampler()
            for doc in source():
                if sampler.should_sample(_document(doc)[0]):
                    yield doc
        return curated

    def _cut(self, text: str) -> list[str]:
        """`text` split at special tokens, which stay at the odd positions."""
        return self._specials.split(text) if self._specials else [text]

    def _prepare(self, text: str) -> str:
        """Hamza folding, then with `space_separated_stream` one space between pre-tokens,
        which replaces line breaks too: "York, don't" becomes "York , don 't"."""
        if self.normalize_hamza:
            text = _fold_hamza(text)
        if self.space_separated_stream:
            text = " ".join(c.lstrip(" ").rstrip("\r\n")
                            for c in _CHUNK.findall(text, concurrent=False)
                            if not _BLANK.fullmatch(c))
        return text

    def _read(self, source) -> None:
        """Read D' once into pre-token ids, documents separated by SEP and the text
        between special tokens by BREAK."""
        index = _Ids()
        ids = array("i", [SEP])
        langs: list[str | None] = []
        shards: list[int] = []
        self.heldout, self.verification = [], []
        for doc in source():
            text, lang = _document(doc)
            lang = lang.lower() if isinstance(lang, str) else None
            # The input is split at special tokens before normalization; the held-out
            # text keeps them, to be split again when it is encoded.
            parts = self._cut(text)
            parts[::2] = [self._prepare(part) for part in parts[::2]]
            # Folding or respacing can spell a special token anew, so each part is cut
            # again: no learned token is ever spelled like one. Matching keeps the GIL,
            # which regex would otherwise release and retake around every match.
            pieces = [_CHUNK.findall(piece, concurrent=False) for part in parts[::2]
                      for piece in self._cut(part)[::2]]
            pieces = [chunks for chunks in pieces if chunks]
            if not pieces:
                continue
            for k, chunks in enumerate(pieces):
                if k:
                    ids.append(BREAK)
                ids.extend(map(index.__getitem__, chunks))
            ids.append(SEP)
            # The shards serve EXPRESSION only.
            shard = shard_of(text, self.holdout_fraction) if self.enable_expression else "train"
            if shard == "held-out":
                self.heldout.append("".join(parts))
            elif shard == "verification":
                self.verification.append("".join(parts))
            shards.append(("train", "held-out", "verification").index(shard))
            langs.append(lang)
        self.types = ["", ""] + list(index)
        self.ids = np.frombuffer(ids, dtype=np.int32).copy()
        self.langs = langs
        self._shard = np.array(shards, np.int8)
        self.stats["documents"] = len(langs)
        self.stats["heldout_documents"] = len(self.heldout)
        self.stats["verification_documents"] = len(self.verification)
        self._split = None
        self.stats["gated_pretokens"] = self._gate() if self.script_aware else 0
        self.stats["pretokens"] = int(np.count_nonzero(self.ids > BREAK))
        self.stats["pretoken_types"] = int(np.count_nonzero(np.bincount(self.ids)[BREAK + 1:]))

    def _gate(self) -> int:
        """Section 3.4: inside boundary-free scripts, a junction seen fewer than 50 times
        splits its pre-token, so no merge can cross it; `self._split` marks the pieces
        after such a junction. Returns the number of pre-token types split."""
        types = self.types
        gated = [t for t in range(1, len(types)) if S.has_boundary_free(types[t])]
        if not gated:
            return 0
        freq = np.bincount(self.ids, minlength=len(types))
        gate = S.BigramGate().fit([types[t] for t in gated], [int(freq[t]) for t in gated])
        index = {t: i for i, t in enumerate(types)}
        split: dict[int, list[int]] = {}
        for t in gated:
            pieces = gate.split(types[t])
            if len(pieces) > 1:
                row = []
                for piece in pieces:
                    if piece not in index:
                        index[piece] = len(types)
                        types.append(piece)
                    row.append(index[piece])
                split[t] = row
        if not split:
            return 0
        width = np.ones(len(types), np.int64)
        for t, row in split.items():
            width[t] = len(row)
        start = np.cumsum(width) - width
        flat = np.repeat(np.arange(len(types), dtype=np.int32), width)
        for t, row in split.items():
            flat[start[t]:start[t] + len(row)] = row
        w = width[self.ids]
        offset = np.arange(w.sum()) - np.repeat(np.cumsum(w) - w, w)
        self.ids = flat[np.repeat(start[self.ids], w) + offset]
        self._split = offset > 0
        return len(split)

    def _suffix_extras(self, doc: np.ndarray, letters: np.ndarray) -> list[tuple[str, int]]:
        """Section 3.4 / Appendix E suffix-aware weighting for Turkish and Finnish.

        Up to 200 suffix candidates are mined per language from its own documents; each
        occurrence of a word whose ending matches one adds `suffix_weight` occurrences of
        that ending as a standalone unit, so merges inside the suffix gain weight while
        merges across the matched boundary do not."""
        langs = np.array([lang or "" for lang in self.langs])
        extras: Counter[str] = Counter()
        self.suffixes = {}
        pos = np.nonzero(letters)[0]
        for lang in sorted(S.SUFFIX_LANGUAGES.intersection(self.langs)):
            at = pos[langs[doc[pos]] == lang]
            counts = np.bincount(self.ids[at], minlength=len(self.types))
            present = np.nonzero(counts)[0]
            suffixes = S.suffixes_from_types({S.fold(self.types[t].strip(), lang) for t in present})
            match = S.SuffixMatcher(suffixes, lang)
            self.suffixes[lang] = suffixes
            for t in present:
                ending = match(self.types[t])
                if ending:
                    extras[ending] += int(counts[t]) * self.suffix_weight
        self.stats["suffixes"] = {lang: len(found) for lang, found in self.suffixes.items()}
        return sorted(extras.items())

    def _mine(self) -> _Mined:
        """Exact word n-gram statistics, and from them the seeds, the MULTIWORD junctions,
        the sizing run's items and the Appendix A spans.

        Seeds (Eq. 2) and the Section 3.5 screen count all of D'; Appendix A counts the
        training shard only and takes c_holdout and df_holdout from the held-out shard.
        """
        types, ids = self.types, self.ids
        size, width = len(ids), len(types)
        isword = np.fromiter((_WORD.fullmatch(t) is not None for t in types), bool, width)
        lead = isword & np.fromiter((t[:1] in _SPACES for t in types), bool, width)
        base = {
            "stop": (isword & np.fromiter((t.strip().lower() in self.stopwords for t in types),
                                          bool, width)).astype(np.int64),
            "wlen": np.fromiter((len(t.strip()) for t in types), np.int64, width),
            "tlen": np.fromiter((len(t) for t in types), np.int64, width),
        }

        live = ids != SEP
        doc = np.cumsum(~live, dtype=np.int32) - 1
        shard = np.zeros(size, np.int8)
        shard[live] = self._shard[doc[live]]
        held = shard == 1
        train = live & (shard == 0)
        del shard
        words = live & isword[ids]
        letters = live & np.fromiter((_LETTERS.fullmatch(t) is not None for t in types),
                                     bool, width)[ids]
        extras = self._suffix_extras(doc, letters) if self.script_aware else []
        unigram = np.bincount(ids[words], minlength=width)
        base["lp_all"] = _log_unigram(unigram)
        base["lp_tr"] = _log_unigram(np.bincount(ids[words & train], minlength=width))
        at = np.nonzero(words & train)[0]
        df = np.bincount(np.unique(doc[at].astype(np.int64) * width + ids[at]) % width,
                         minlength=width)
        # Eq. 7 writes a bare `log`; natural log is used, unlike the explicit log2 elsewhere.
        base["idf"] = np.log((np.count_nonzero(np.bincount(doc[train])) + 1) / (1 + df))

        top = max(self.max_seed_ngram if self.enable_seeds else 0,
                  2 if self.enable_multiword else 0,
                  self.max_span if self.enable_expression else 0)
        prev = dict(base, adj=np.zeros(width), last=np.arange(width))
        row = np.where(words, ids, -1).astype(np.int32)
        valid = words
        links: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        seed_cands: list[tuple[int, int, int, tuple[str, ...], np.ndarray]] = []
        screened_occ: tuple[np.ndarray, np.ndarray] = (np.zeros(0, np.int64), np.zeros(0, np.int64))
        bigrams: dict[tuple[str, ...], int] = {}
        bigram_keys, bigram_npmi = np.zeros(0, np.int64), np.zeros(0)
        spans: list[_Span] = []
        # A word and what follows it without a space ("York" + ",", "don" + "'t") form one
        # whitespace-delimited word, Appendix A's unit, so MULTIWORD may join them without
        # the Section 3.5 screen.
        junctions = np.zeros(size, bool)
        junctions[1:] = isword[ids[1:]] & ~lead[ids[1:]] & isword[ids[:-1]]
        if self._split is not None:
            junctions &= ~self._split
        if self.script_aware:
            # Chinese, Japanese and Thai put no space after punctuation: the junction to the
            # next word is where a space would be, and the gate holds no bigram evidence
            # for it, so it is not an attached junction.
            # Full-width and CJK punctuation (，。、！) ends a clause whatever follows it.
            punct = np.fromiter((_PUNCT.fullmatch(t) is not None for t in types), bool, width)
            gated = np.fromiter((S.gatable(t[:1]) if t else False for t in types), bool, width)
            clause = punct & np.fromiter((_CJK_PUNCT.search(t) is not None for t in types),
                                         bool, width)
            junctions[1:] &= ~((punct[ids[:-1]] & gated[ids[1:]]) | clause[ids[:-1]])
        attached = junctions.copy()

        for n in range(2, top + 1):
            floors = []
            if self.enable_seeds and n <= self.max_seed_ngram:
                floors.append(self.seed_min_frequency)
            if self.enable_multiword and n == 2:
                floors.append(self.screening_frequency)
            if self.enable_expression and n <= self.max_span:
                floors.append(A.MIN_COUNT)
            nxt = np.zeros(size, bool)
            reach = max(0, size - n + 1)
            nxt[:reach] = valid[:reach] & lead[ids[n - 1:n - 1 + reach]]
            valid = nxt
            n_all, n_tr = int(valid.sum()), int((valid & train).sum())
            # An n-gram occurs no more often than its first or its last n - 1 words, and
            # the floor never falls as n grows, so only n-grams whose two (n-1)-grams were
            # kept are counted.
            if n == 2:
                row = np.where(unigram[ids] >= min(floors), row, -1)
            counted = valid & (row >= 0)
            counted[:-1] &= row[1:] >= 0
            start = np.flatnonzero(counted)
            del counted
            keys, order = _sort(row[start].astype(np.int64) * width + ids[start + n - 1])
            start = start[order]
            del order
            head = np.flatnonzero(np.diff(keys, prepend=-1))
            cnt = np.diff(head, append=len(keys))
            keep = cnt >= min(floors)
            key = keys[head[keep]]
            del keys, head
            row = np.full(size, -1, np.int32)
            row[start] = np.repeat(np.where(keep, np.cumsum(keep) - 1, -1).astype(np.int32), cnt)
            pre, last = key // width, key % width
            links[n] = (pre, last)
            rows = len(key)
            c_all = cnt[keep]
            hit = row >= 0
            c_tr = np.bincount(row[hit & train], minlength=rows)
            hp = np.nonzero(hit & held)[0]
            c_ho = np.bincount(row[hp], minlength=rows)
            df_ho = np.bincount(np.unique(doc[hp].astype(np.int64) * max(rows, 1) + row[hp])
                                % max(rows, 1),
                                minlength=rows)
            cur = {name: prev[name][pre] + base[name][last] for name in base}
            with np.errstate(divide="ignore", invalid="ignore"):
                pmi_all = np.log2(c_all / max(n_all, 1)) - cur["lp_all"]
                p_tr = c_tr / max(n_tr, 1)
                pmi_tr = np.log2(p_tr) - cur["lp_tr"]
                npmi_tr = np.where(p_tr < 1, pmi_tr / -np.log2(p_tr), 1.0)
            if n == 2:
                bigram_keys, bigram_npmi = key, npmi_tr
                cur["adj"] = npmi_tr
            elif len(bigram_keys):
                pair = prev["last"][pre] * width + last
                at = np.minimum(np.searchsorted(bigram_keys, pair), len(bigram_keys) - 1)
                cur["adj"] = prev["adj"][pre] + np.where(bigram_keys[at] == pair,
                                                         bigram_npmi[at], -np.inf)
            else:
                cur["adj"] = np.full(rows, -np.inf)
            cur["last"] = last

            if self.enable_seeds and n <= self.max_seed_ngram:
                chosen = (c_all >= self.seed_min_frequency) & (pmi_all >= self.seed_pmi_threshold)
                if chosen.any():
                    # `start` holds the occurrences of each n-gram together, in row order.
                    group = np.zeros(len(cnt), bool)
                    group[keep] = chosen
                    occ = start[np.repeat(group, cnt)]
                    picked = np.flatnonzero(chosen)
                    ends = np.cumsum(c_all[picked])
                    for r, span, lo, hi in zip(picked.tolist(), _words(links, types, n, picked),
                                               (ends - c_all[picked]).tolist(), ends.tolist()):
                        seed_cands.append((hi - lo, n, r, span, occ[lo:hi]))
            del start, cnt, keep
            if self.enable_multiword and n == 2:
                screened = (c_all >= self.screening_frequency) & (pmi_all >= self.screening_pmi)
                bigrams = {(types[pre[r]], types[last[r]]): int(c_all[r])
                           for r in np.nonzero(screened)[0]}
                if screened.any():
                    pos = np.flatnonzero(hit & screened[np.maximum(row, 0)])
                    screened_occ = (pos, row[pos])
                    junctions[pos + 1] = True

            if self.enable_expression and n <= self.max_span:
                ok = ((c_tr >= A.MIN_COUNT) & (c_ho > 0) & (pmi_tr >= A.PMI_FLOOR)
                      & (cur["stop"] < A.STOPWORD_CEILING * n))
                if ok.any():
                    pos = np.flatnonzero(hit & train & ok[np.maximum(row, 0)])
                    h_l, p_l = _branching(row[pos], ids[pos - 1], rows, width)
                    h_r, p_r = _branching(row[pos], ids[pos + n], rows, width)
                    h_b, peak = np.minimum(h_l, h_r), np.maximum(p_l, p_r)
                    with np.errstate(invalid="ignore"):
                        gate = 1.0 / (1.0 + np.exp(-A.BOUNDARY_GATE_SLOPE * (h_b - A.TAU_BE)))
                    ok &= gate >= A.BOUNDARY_GATE_FLOOR
                    r = np.flatnonzero(ok)
                    spans.extend(map(_Span._make, zip(
                        _words(links, types, n, r), c_tr[r].tolist(), c_ho[r].tolist(),
                        df_ho[r].tolist(), npmi_tr[r].tolist(), h_b[r].tolist(), peak[r].tolist(),
                        (cur["idf"][r] / n).tolist(), (cur["adj"][r] / (n - 1)).tolist(),
                        (cur["stop"][r] / n).tolist(), (cur["wlen"][r] / n).tolist(),
                        cur["tlen"][r].tolist())))
            prev = cur

        # Overlap control: candidates are taken by the text they would cover,
        # count * (n - 1) merges, and one is dropped as a near-duplicate variant when
        # `seed_overlap` of its occurrences overlap those of a seed already kept.
        seeds: dict[tuple[str, ...], int] = {}
        chosen_occ: list[tuple[int, int, np.ndarray]] = []
        covered = np.zeros(size, bool)
        for count, n, r, span, occ in sorted(seed_cands,
                                             key=lambda c: (-c[0] * (c[1] - 1), -c[1], c[2])):
            overlap = np.zeros(len(occ), bool)
            for k in range(n):
                overlap |= covered[occ + k]
            if overlap.mean() >= self.seed_overlap:
                continue
            for k in range(n):
                covered[occ + k] = True
            seeds[span] = count
            chosen_occ.append((n, r, occ))
        # Section 3.3's PMI-seeded MULTIWORD candidates: junctions inside a seed pass.
        for n, _, occ in chosen_occ:
            for k in range(1, n):
                junctions[occ + k] = True

        every = _counts(ids[live], types)
        units: list[tuple[str, int]] = []
        covered = np.zeros(size, bool)
        if chosen_occ or len(screened_occ[0]):
            # Units are taken left to right, the longest first at each position, unless they
            # overlap the last unit taken, so only the longest unit at a position matters.
            span_n = np.zeros(size, np.int8)
            span_row = np.zeros(size, np.int32)
            span_n[screened_occ[0]] = 2
            span_row[screened_occ[0]] = screened_occ[1]
            for n, r, occ in sorted(chosen_occ, key=lambda c: c[0]):
                span_n[occ] = n
                span_row[occ] = r
            pos = _take(span_n)
            span_n, span_row = span_n[pos], span_row[pos]
            # Taken units do not overlap.
            cover = np.zeros(size + 1, np.int8)
            cover[pos] = 1
            cover[pos + span_n] -= 1
            covered = np.cumsum(cover, dtype=np.int8)[:size] > 0
            stride = int(span_row.max()) + 1
            unit, count = np.unique(span_n.astype(np.int64) * stride + span_row, return_counts=True)
            for n in np.unique(span_n).tolist():
                at = unit // stride == n
                units += zip(map("".join, _words(links, types, n, unit[at] % stride)),
                             count[at].tolist())
        # Outside the units, a word and what is attached to it enter as the one
        # whitespace-delimited word they form, so the merges joining them compete too.
        free = live & ~covered
        start, length = chains.runs(attached, free & isword[ids])
        has_attached = bool(len(start))
        whole = np.zeros(size + 1, np.int32)
        np.add.at(whole, start, 1)
        np.add.at(whole, start + length, -1)
        plain = _counts(ids[free & ~(np.cumsum(whole)[:size] > 0)], types)
        # MULTIWORD joins at most max(max_seed_ngram, max_span) pieces, so a longer run (a
        # URL, unspaced CJK text) enters in windows of that many words.
        start, length = chains.windows(start, length, max(self.max_seed_ngram, self.max_span))
        start, length, count = chains.unique_runs(ids, start, length)
        plain += [("".join(types[t] for t in ids[a:a + n].tolist()), int(c))
                  for a, n, c in zip(start.tolist(), length.tolist(), count.tolist())]
        return _Mined(seeds, bigrams, every, plain, units, extras, spans, junctions,
                      has_attached)

    # ---------------------------------------------------------------- passes

    def _set_alphabet(self, items: list[tuple[str, int]]) -> None:
        """Figure 1 line 3: the character alphabet. A character enters T when it occurs
        at least `min_frequency` times and makes up at least one in RARE_CHARACTER
        training characters, and the space always does; rarer and unseen characters use
        byte fallback. The characters that spell a byte-fallback token always enter too,
        so text that spells one is read as characters."""
        text = "".join(s for s, _ in items)
        codes = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)
        weights = np.repeat(np.array([c for _, c in items], dtype=np.float64),
                            [len(s) for s, _ in items])
        counts = np.bincount(codes, weights=weights)
        least = max(1, self.min_frequency, counts.sum() / RARE_CHARACTER)
        frequent = {chr(cp) for cp in np.nonzero(counts >= least)[0]}
        self.alphabet = sorted(frequent | {" "} | _BYTE_SPELLING_CHARS)
        self._alpha = frozenset(self.alphabet)
        self._base = BYTE_TOKENS + self.alphabet

    def _vocab(self, merges: list[tuple[str, str]]) -> dict[str, int]:
        vocab = {t: i for i, t in enumerate(self._base)}
        for left, right in merges:
            vocab.setdefault(left + right, len(vocab))
        return vocab

    def _atoms(self, text: str) -> list[str]:
        """Initial symbols of `text`: its characters, or their UTF-8 bytes outside T."""
        out: list[str] = []
        for ch in text:
            if ch in self._alpha:
                out.append(ch)
            else:
                out.extend(f"<0x{b:02X}>" for b in ch.encode("utf-8"))
        return out

    def _bpe_run(self, items: list[tuple[str, int]], budget: int) -> list[tuple[str, str]]:
        trainer = trainers.BpeTrainer(
            vocab_size=budget - len(BYTE_TOKENS), min_frequency=self.min_frequency,
            show_progress=False, initial_alphabet=self.alphabet,
            limit_alphabet=len(self.alphabet))
        # The alphabet runs hold characters of T only, so any other character separates.
        sep = next(chr(c) for c in range(0xE000, 0x110000) if chr(c) not in self._alpha)
        return _train_bpe(self._alphabet_runs(items), trainer, sep)

    def _alphabet_runs(self, items: list[tuple[str, int]]) -> Iterable[tuple[str, int]]:
        """Cut items at characters outside T, which byte fallback encodes and no merge
        can touch, so BPE never pairs the characters around them."""
        alpha = self._alpha
        for text, count in items:
            if all(ch in alpha for ch in text):
                yield text, count
                continue
            run = []
            for ch in text:
                if ch in alpha:
                    run.append(ch)
                elif run:
                    yield "".join(run), count
                    run = []
            if run:
                yield "".join(run), count

    def _pass1(self, trad: list[tuple[str, str]], seeds: dict[tuple[str, ...], int],
               size: int, budget: int, fill: bool, reserve: int = 0):
        """Figure 1 line 3 and the TRADITIONAL pass: the first `size` entries of
        word-bounded BPE, then the seeds. A seed's merges need its words, so they follow
        the pass-1 merges, but the seed counts toward |T| from the start. Seeds are taken
        in order of the text they cover, count * (n - 1), as the overlap control ranks
        them, and one whose merges no longer fit in T less the `reserve` kept for MULTIWORD
        is skipped; with `fill`, pass 1 then takes the rest of T.

        Returns the merges, the pass-1 merge count, the seed merge count, the seeds that
        are tokens of T and the count of seeds left out by reason."""
        adds, known = [], set(self._base)
        for left, right in trad:
            adds.append(left + right not in known)
            known.add(left + right)
        grown = np.cumsum(adds)

        order = sorted(seeds, key=lambda s: (-seeds[s] * (len(s) - 1), s))

        def build(k: int):
            merges = list(trad[:k])
            ranks = {m: i for i, m in enumerate(merges)}
            vocab = set(self._base) | {a + b for a, b in merges}
            placed, kept = 0, []
            dropped: Counter[str] = Counter()
            for span in order:
                pieces = _bpe(self._atoms("".join(span)), ranks)
                if _BYTE_SET.intersection(pieces):
                    dropped["a character outside T"] += 1
                    continue
                plan, _ = _plan(pieces, ranks, len(merges), vocab)
                if len(vocab) + len({a + b for a, b in plan} - vocab) > budget - reserve:
                    dropped["no room in T"] += 1
                    continue
                for pair in plan:
                    ranks[pair] = len(merges)
                    merges.append(pair)
                    vocab.add(pair[0] + pair[1])
                placed += len(plan)
                kept.append(span)
            return merges, len(vocab), placed, kept, dropped

        k = int(np.searchsorted(grown, size, side="right"))
        merges, n, placed, kept, dropped = build(k)
        while fill and n < budget and k < len(trad):
            step = min(len(trad), k + budget - n)
            attempt = build(step)
            if sum(attempt[4].values()) > sum(dropped.values()):
                break
            k, (merges, n, placed, kept, dropped) = step, attempt
        return merges, k, placed, {span: seeds[span] for span in kept}, dropped

    def _multiword(self, merges: list[tuple[str, str]], budget: int,
                   junctions: np.ndarray) -> tuple[list[tuple[str, str]], int, int]:
        """MULTIWORD: resume BPE across junctions between words.

        A chain is a maximal run of words, each joined to the word before across a
        junction that passes the Section 3.5 screen, lies inside a seed, or joins a word to
        what is attached to it; line breaks, double spaces, document ends, special tokens
        and words T spells with a byte token end a chain, and a chain longer than MAX_CHAIN
        words is cut into windows. Each word enters as the tokens T spells it with, one
        symbol each, and the pieces of one word are kept apart, so every merge crosses a
        junction: a word or its last piece joins the next word or its first piece (" New" +
        " York", "ing" + ","). Every merge ranks after the merges T holds and is the most
        frequent valid one, and a token spans at most max(max_seed_ngram, max_span)
        pieces.

        Returns the merges, the number added and the number of distinct chains."""
        vocab = self._vocab(merges)
        if budget <= len(vocab):
            return merges, 0, 0
        types, ids = self.types, self.ids
        word = np.fromiter((_WORD.fullmatch(t) is not None for t in types), bool, len(types))
        present = np.flatnonzero((np.bincount(ids, minlength=len(types)) > 0) & word)
        tk = build_bpe_tokenizer(vocab, merges, [])
        encoded = tk.encode_batch([types[t] for t in present.tolist()])
        usable = np.fromiter((not _BYTE_SET.intersection(e.tokens) for e in encoded), bool,
                             len(present))
        pieces_of = {t: e.tokens for t, e, ok in zip(present.tolist(), encoded, usable) if ok}
        is_usable = np.zeros(len(types), bool)
        is_usable[list(pieces_of)] = True
        start, length = chains.runs(junctions, is_usable[ids])
        start, length = chains.windows(start, length, MAX_CHAIN)
        start, length, count = chains.unique_runs(ids, start, length)
        if not len(start):
            return merges, 0, 0
        # One code point per piece; code points follow the pieces' text, not corpus order.
        names = sorted({p for ps in pieces_of.values() for p in ps})
        code = dict(zip(names, map(chr, _code_points(len(names)).tolist())))
        spelled = {t: "\x00".join(code[p] for p in ps) for t, ps in pieces_of.items()}
        offset = np.cumsum(length) - length
        at = np.repeat(start - offset, length) + np.arange(int(length.sum()))
        words = [spelled[t] for t in ids[at].tolist()]
        seqs = [("".join(words[o:o + n]), c)
                for o, n, c in zip(offset.tolist(), length.tolist(), count.tolist())]
        token = {c: p for p, c in code.items()}
        text = "".join(t for t, _ in seqs).replace("\x00", "")
        room = budget - len(vocab)
        longest = max(self.max_seed_ngram, self.max_span)
        # Merges forming a token T already holds (seeds and their intermediates, learned
        # again over the chains) are skipped below, so the trainer is asked for the room
        # plus a margin, and again with a larger margin if it hit its cap first. One
        # symbol is one piece, so the token-length limit caps the pieces a token spans.
        margin = sum(1 for a, b in merges if len(_CHUNK.findall(a + b)) > 1) + room // 10 + 100
        symbols = len(set(text))
        for _ in range(4):
            asked = room + margin
            # The trainer's token-length limit is exclusive.
            trainer = trainers.BpeTrainer(
                vocab_size=symbols + asked, min_frequency=self.min_frequency,
                show_progress=False, max_token_length=longest + 1)
            learned = _train_bpe(seqs, trainer, "\x00")
            out, tokens = list(merges), set(vocab)
            for a, b in learned:
                if len(tokens) >= budget:
                    break
                # Enforced here as well, whatever a tokenizers version does with the limit.
                if len(a) + len(b) > longest:
                    continue
                pair = ("".join(map(token.__getitem__, a)),
                        "".join(map(token.__getitem__, b)))
                # A byte token is no merge's input, and pieces attached to one another
                # can spell one ("<0x41>", "<0xab>"), which would read as that byte.
                if (pair[0] + pair[1] in tokens or pair[0] not in tokens
                        or pair[1] not in tokens
                        or any(_BYTE_SPELLING.fullmatch(t)
                               for t in (pair[0], pair[1], pair[0] + pair[1]))):
                    continue
                out.append(pair)
                tokens.add(pair[0] + pair[1])
            if len(tokens) >= budget or len(learned) < asked:
                break
            margin = 2 * margin + budget - len(tokens)
        return out, len(out) - len(merges), len(start)

    @staticmethod
    def _leaves(merges: list[tuple[str, str]], heldout: list[str],
                tk: Tokenizer) -> list[tuple[float, int, tuple[str, str]]]:
        """Leaf merges with their held-out utility U(m) (Eq. 13), ascending.

        A leaf's output is no other merge's input; outputs made by two merges are skipped,
        since retiring one of them would free no slot."""
        # Tokens are counted by id, one to one with their strings.
        rows = [enc.ids for enc in tk.encode_batch(heldout)]
        width, index = tk.get_vocab_size(), tk.get_vocab()
        seen = np.bincount(np.fromiter(chain.from_iterable(rows), np.int64),
                           minlength=width).tolist()
        docs = np.bincount(np.fromiter(chain.from_iterable(map(set, rows)), np.int64),
                           minlength=width).tolist()
        parts = {x for m in merges for x in m}
        made = Counter(a + b for a, b in merges)
        return sorted((A.leaf_utility(seen[index[a + b]], docs[index[a + b]]), -i, (a, b))
                      for i, (a, b) in enumerate(merges)
                      if a + b not in parts and made[a + b] == 1)

    def _verification_tokens(self, merges: list[tuple[str, str]]) -> int:
        """Tokens the verification shard costs under `merges`."""
        tk = build_bpe_tokenizer(self._vocab(merges), merges, self.special_tokens)
        return sum(len(e.ids) for e in tk.encode_batch(self.verification))

    def _score(self, span: _Span, n_tok: int) -> float:
        """Eqs. 7-12 with the hard rejections that depend on the current tokenizer."""
        if n_tok <= 1 or not _valid_surface("".join(span.words)):
            return 0.0
        return A.score(gain=n_tok - 1, count=span.train_count, holdout_count=span.holdout_count,
                       holdout_df=span.holdout_df, npmi=span.npmi,
                       boundary_entropy=span.boundary_entropy, avg_idf=span.avg_idf,
                       internal_npmi=span.internal_npmi, context_peak=span.context_peak,
                       stopword_ratio=span.stopword_ratio, n_words=len(span.words),
                       avg_word_len=span.avg_word_len, char_len=span.char_len)

    def _expression(self, merges: list[tuple[str, str]], spans: list[_Span], budget: int):
        """EXPRESSION: Appendix A scores, then leaf surgery (Eqs. 13-14).

        Spans are visited in Priority order. Each is evaluated against the tokenizer as
        it stands, after the acceptances before it: l_tok, S(s), A(s) and the leaves it
        would displace are recomputed, and the span is accepted if Net(s) > 0, subject
        to the chunk check below. Its merges take any free slots of T, then displace the
        lowest-utility leaf merges. A span T spells in three or more tokens needs
        intermediate merges; its derivation reuses existing tokens and Appendix A-admitted
        spans where it can, and every new intermediate needs a slot, as the span's own
        token does.

        Utility is estimated per span on the held-out shard and checked per chunk on the
        verification shard: after every
        EXPRESSION_CHUNK spans, the verification shard is encoded, and the chunk is kept
        only if its token count fell; the pass stops at, and reverts, the first chunk that
        does not lower it."""
        vocab = self._vocab(merges)
        tk = build_bpe_tokenizer(vocab, merges, self.special_tokens)
        surfaces = ["".join(s.words) for s in spans]
        initial = [enc.tokens for enc in tk.encode_batch(surfaces)]
        scores = [self._score(span, len(p)) for span, p in zip(spans, initial)]
        live = [i for i, s in enumerate(scores) if s > 0.0]
        if not live:
            return merges, 0, 0, 0, 0
        admitted = {surfaces[i] for i in live}

        pool = self._leaves(merges, self.heldout, tk)
        free = max(0, budget - len(vocab))
        cheapest = np.cumsum(np.concatenate([np.zeros(free), [u for u, _, _ in pool]])).tolist()
        gains = [A.addition_gain(span.holdout_count, len(p), span.holdout_df)
                 for span, p in zip(spans, initial)]

        def priority(i: int):
            cost = len(initial[i]) - 1
            net = gains[i] - cheapest[cost - 1] if cost <= len(cheapest) else -np.inf
            return -A.priority(net, scores[i])

        # Priority order, ties broken by surface: two stable sorts.
        order = sorted(live, key=surfaces.__getitem__)
        order.sort(key=priority)
        ranks = {m: i for i, m in enumerate(merges)}
        # How many merges of `ranks` output each token.
        made = Counter(a + b for a, b in merges)
        tokens = set(vocab)
        protected = {"".join(seed) for seed in self.seeds}
        retired: set[tuple[str, str]] = set()
        gone: set[str] = set()
        added: list[tuple[str, str]] = []
        cursor = accepted = unscored = 0
        best = self._verification_tokens(merges)
        kept_chunks = 0
        for lo in range(0, len(order), EXPRESSION_CHUNK):
            saved = (dict(ranks), Counter(made), set(tokens), set(protected), set(retired),
                     set(gone), len(added), cursor, accepted, unscored, free)
            for i in order[lo:lo + EXPRESSION_CHUNK]:
                if not free and cursor >= len(pool):
                    break
                span, pieces, surface = spans[i], initial[i], surfaces[i]
                # With no free slot, a span that is neither a token nor a merge's output needs
                # a new token, so it displaces at least the cheapest leaf left. A(s) only falls
                # with l_tok, which merging intact initial pieces cannot raise: such a span is
                # rejected before its plan, and before BPE when its pieces are intact.
                short = not free and surface not in tokens and not made.get(surface)
                if gone.isdisjoint(pieces):
                    if short and gains[i] <= pool[cursor][0]:
                        continue
                    pieces = _bpe(pieces, ranks)
                else:
                    pieces = _bpe(self._atoms(surface), ranks)
                gain = A.addition_gain(span.holdout_count, len(pieces), span.holdout_df)
                if short and gain <= pool[cursor][0]:
                    continue
                s = scores[i] if len(pieces) == len(initial[i]) else self._score(span, len(pieces))
                if s <= 0.0 or _BYTE_SET.intersection(pieces):
                    continue
                plan, hold = _plan(pieces, ranks, len(merges) + len(added), tokens, admitted)
                if not plan:
                    continue
                fresh = {a + b for a, b in plan} - tokens
                use = min(free, len(fresh))
                take: list[tuple[float, tuple[str, str]]] = []
                j = cursor
                while len(take) < len(fresh) - use and j < len(pool):
                    u, _, m = pool[j]
                    j += 1
                    if m[0] + m[1] not in protected and m[0] + m[1] not in hold:
                        take.append((u, m))
                if len(take) < len(fresh) - use or gain - sum(u for u, _ in take) <= 0.0:
                    continue
                free -= use
                cursor = j
                for _, m in take:
                    retired.add(m)
                    del ranks[m]
                    made[m[0] + m[1]] -= 1
                    tokens.discard(m[0] + m[1])
                    gone.add(m[0] + m[1])
                for pair in plan:
                    ranks[pair] = len(merges) + len(added)
                    made[pair[0] + pair[1]] += 1
                    added.append(pair)
                    tokens.add(pair[0] + pair[1])
                protected |= hold
                accepted += 1
                unscored += sum(1 for o in fresh
                                if o != surface and o not in admitted and _CROSSES.search(o))
            if accepted > saved[8]:
                count = self._verification_tokens(
                    [m for m in merges if m not in retired] + added)
                if count >= best:
                    (ranks, made, tokens, protected, retired, gone, kept, cursor, accepted,
                     unscored, free) = saved
                    del added[kept:]
                    break
                best = count
                kept_chunks += 1
            if not free and cursor >= len(pool):
                break
        self.stats["pass3_chunks_kept"] = kept_chunks
        return ([m for m in merges if m not in retired] + added, len(live), accepted,
                len(retired), unscored)

    # ------------------------------------------------------------------- run

    def train(self, source) -> tuple[dict[str, int], list[tuple[str, str]]]:
        """Train on `source`, an iterable of documents or a callable returning one; each
        document is a `str` or a `(str, language)` pair. The corpus is read once.

        Returns the vocabulary {token: id} and the merges in rank order."""
        if isinstance(source, (str, bytes)):
            raise TypeError("source must be documents or a callable returning them, "
                            "not a single string")
        if not callable(source):
            docs = iter(source)
            source = lambda: docs
        try:
            return self._fit(source)
        finally:
            # The corpus read for training is released, whether training finished or
            # failed; the seeds, suffixes, alphabet and statistics stay.
            self.ids, self.types, self.langs = np.zeros(0, np.int32), [], []
            self.heldout, self.verification = [], []
            self._shard, self._split = np.zeros(0, np.int8), None

    def _fit(self, source) -> tuple[dict[str, int], list[tuple[str, str]]]:
        self.stats = {}
        if self.use_entropy_filtering:
            source = self._curate(source)
        self._read(source)
        if not self.stats["pretokens"]:
            raise ValueError("the corpus holds no text")
        mined = self._mine()
        budget = self.vocab_size - len(self.special_tokens)
        every = mined.every + mined.extras
        self._set_alphabet(mined.every)
        if budget <= len(self._base):
            raise ValueError(f"vocab_size leaves no room for merges beyond the "
                             f"{len(self._base)} byte and character symbols")

        # Figure 1 sets no quotas, and TRADITIONAL, run while |T| < V, would fill T alone.
        # The passes are sized by one BPE run, the sizing run, in which all their candidates
        # compete: TRADITIONAL keeps its merges inside one pre-token; those joining
        # pre-tokens without a space (" York" + ",", "ing" + ",") are the attached share,
        # which MULTIWORD forms; those crossing a space are the cross-word share, seeds
        # first. The TRADITIONAL run and the sizing run release the GIL and run side by side.
        size, attached_share = budget - len(self._base), 0
        multiword = self.enable_multiword and bool(mined.units or mined.has_attached)
        with ThreadPoolExecutor(2) as pool:
            trad_run = pool.submit(self._bpe_run, every, budget)
            sizing_run = (pool.submit(self._bpe_run, mined.plain + mined.units + mined.extras,
                                     budget)
                         if mined.units or multiword else None)
            trad = trad_run.result()
            if sizing_run is not None:
                within = {a + b for a, b in sizing_run.result() if not _CROSSES.search(a + b)}
                attached_share = sum(1 for t in within if len(_CHUNK.findall(t)) > 1)
                size = len(within) - attached_share
        reserve = attached_share if multiword else 0
        merges, pass1, placed, self.seeds, dropped = self._pass1(
            trad, mined.seeds, size, budget, fill=not multiword, reserve=reserve)
        pass2 = n_chains = 0
        if multiword:
            merges, pass2, n_chains = self._multiword(merges, budget, mined.junctions)
        # MULTIWORD, resuming from the pieces T spells words with, needs fewer merges than
        # the sizing run spent on its units. For up to three rounds the slots it leaves go
        # back to TRADITIONAL, and the attached share shrinks by as many, so every seed that
        # fitted still fits.
        for _ in range(3):
            short = budget - len(self._vocab(merges))
            if not multiword or short <= 0 or pass1 >= len(trad):
                break
            size += short
            reserve = max(0, reserve - short)
            merges, pass1, placed, self.seeds, dropped = self._pass1(
                trad, mined.seeds, size, budget, fill=False, reserve=reserve)
            merges, pass2, n_chains = self._multiword(merges, budget, mined.junctions)
        for reason, count in sorted(dropped.items()):
            logger.warning("%d of %d seed phrases left out: %s", count, len(mined.seeds), reason)
        scored = accepted = retired = unscored = 0
        if self.enable_expression and not (self.heldout and self.verification):
            logger.warning("EXPRESSION skipped: no document fell in the %s shard "
                           "(holdout_fraction=%g)",
                           "verification" if self.heldout else "held-out",
                           self.holdout_fraction)
        elif self.enable_expression:
            merges, scored, accepted, retired, unscored = self._expression(
                merges, mined.spans, budget)

        vocab = self._vocab(merges)
        self.stats.update(
            alphabet=len(self.alphabet), seeds_mined=len(mined.seeds),
            seed_units=len(self.seeds), seed_merges=placed,
            pass1_merges=pass1, attached_share=attached_share if multiword else 0,
            pass2_candidates=len(mined.units) if multiword else 0,
            pass2_chains=n_chains, pass2_merges=pass2,
            pass3_candidates=scored, pass3_accepted=accepted, surgery_retired=retired,
            pass3_chunks_kept=self.stats.get("pass3_chunks_kept", 0),
            pass3_unscored_intermediates=unscored,
            vocab=len(vocab), merges=len(merges),
            cross_boundary=sum(1 for t in vocab if spans_boundary(t)))
        if len(vocab) < budget:
            logger.warning("vocabulary reached %d of %d: the corpus ran out of merge "
                           "candidates at min_frequency=%d",
                           len(vocab) + len(self.special_tokens), self.vocab_size,
                           self.min_frequency)
        return vocab, merges


def _log_unigram(counts: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return np.log2(counts / max(int(counts.sum()), 1))


def _take(span_n: np.ndarray) -> np.ndarray:
    """Start positions of the units taken left to right, where `span_n` holds the length
    of the unit at each position (0 for none) and a unit overlapping the last one taken is
    skipped.

    A unit that no earlier unit reaches is always taken, and after a taken unit the next
    is the first that starts at or past its end: the units taken are those reached from
    the always-taken ones, found by pointer doubling."""
    pos = np.flatnonzero(span_n)
    end = pos + span_n[pos]
    # The first unit at or past each end is at most max(span_n) - 1 units further on; the last
    # entry of `jump` and `take` stands past the last unit.
    jump = np.append(np.arange(1, len(pos) + 1), len(pos))
    bound = np.append(pos, len(span_n))
    for _ in range(int(span_n.max(initial=0)) - 1):
        jump[:-1] += bound[jump[:-1]] < end
    take = np.ones(len(pos) + 1, bool)
    take[1:-1] = pos[1:] >= np.maximum.accumulate(end)[:-1]
    while True:
        hop = jump[take]
        hop = hop[~take[hop]]
        if not len(hop):
            break
        take[hop] = True
        jump = jump[jump]
    return pos[take[:-1]]


def _sort(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The non-negative int64 `keys` sorted, and the indices that sort them.

    Each key carries its index in its low bits, so the values sort directly, several times
    faster than an indirect sort; keys too wide for that are first grouped by their high
    bits with a radix sort, and with more than 2**16 such groups it falls back to
    np.argsort."""
    bits = len(keys).bit_length()
    room = 63 - bits
    high = keys >> room
    groups = int(high.max()) + 1 if len(keys) else 1
    if groups > 1 << 16:
        order = np.argsort(keys)
        return keys[order], order
    ends = np.cumsum(np.bincount(high, minlength=groups)).tolist()
    order = (np.argsort(high.astype(np.uint16), kind="stable") if groups > 1
             else np.arange(len(keys)))
    del high
    out = np.empty_like(keys)
    for group, (lo, hi) in enumerate(zip([0] + ends[:-1], ends)):
        part = order[lo:hi]
        packed = keys[part]
        packed &= (1 << room) - 1
        packed <<= bits
        packed |= np.arange(hi - lo)
        packed.sort()
        out[lo:hi] = packed >> bits | group << room
        packed &= (1 << bits) - 1
        order[lo:hi] = part[packed]
    return out, order


def _branching(rows: np.ndarray, ctx: np.ndarray, size: int, width: int):
    """Eq. 6 branching entropy and the peak context probability of each row."""
    keys, counts = np.unique(rows.astype(np.int64) * width + ctx, return_counts=True)
    owner = keys // width
    total = np.bincount(owner, weights=counts, minlength=size)
    plogp = np.bincount(owner, weights=counts * np.log2(counts), minlength=size)
    top = np.zeros(size)
    # Keys are sorted, so each row's contexts are contiguous.
    head = np.flatnonzero(np.diff(owner, prepend=-1))
    if len(head):
        top[owner[head]] = np.maximum.reduceat(counts, head)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log2(total) - plogp / total, top / total


def _words(links, types: list[str], n: int, rows: np.ndarray) -> list[tuple[str, ...]]:
    """The words of each n-gram in `rows`."""
    cols = []
    while n > 1:
        pre, last = links[n]
        cols.append(last[rows])
        rows = pre[rows]
        n -= 1
    cols.append(rows)
    return list(zip(*[[types[t] for t in col.tolist()] for col in reversed(cols)]))


def _counts(ids: np.ndarray, types: list[str]) -> list[tuple[str, int]]:
    counts = np.bincount(ids, minlength=len(types))
    present = np.flatnonzero(counts[BREAK + 1:]) + BREAK + 1
    return list(zip([types[t] for t in present.tolist()], counts[present].tolist()))


def _train_bpe(items: Iterable[tuple[str, int]], trainer: trainers.BpeTrainer,
               sep: str) -> list[tuple[str, str]]:
    """Merges learned from each text counted `count` times. The copies are joined by
    `sep`, which no text contains, and split apart again in Rust: the trainer counts the
    same words, with far fewer strings handed over from Python."""
    def joined(batch: int = 4096) -> Iterable[str]:
        for text, count in items:
            while count > 0:
                step = min(count, batch)
                yield sep.join([text] * step)
                count -= step

    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.Split(sep, behavior="removed")
    tk.train_from_iterator(joined(), trainer=trainer)
    return _extract(tk)[1]


def _code_points(n: int) -> np.ndarray:
    """n distinct code points from U+0100, skipping the surrogates; NUL stays free to
    separate texts."""
    cp = np.arange(0x100, 0x100 + n, dtype=np.int64)
    return np.where(cp >= 0xD800, cp + 0x800, cp)


def _bpe(parts: list[str], ranks: dict[tuple[str, str], int],
         after: dict[tuple[str, str], int] | None = None,
         formed: set[str] | None = None) -> list[str]:
    """Apply merges by rank, leftmost first, as the BPE model does at encode time.

    `after` holds further merges ranked after all of `ranks`; every token made on the way
    is added to `formed`."""
    parts = list(parts)
    after = after or {}

    def rank(pair: tuple[str, str]) -> float:
        r = ranks.get(pair)
        return after.get(pair, np.inf) if r is None else r

    # The rank of each adjacent pair; a merge changes only the two pairs beside it.
    pending = [rank(pair) for pair in zip(parts, parts[1:])]
    while pending:
        best = min(pending)
        if best == np.inf:
            break
        at = pending.index(best)
        parts[at:at + 2] = [parts[at] + parts[at + 1]]
        del pending[at]
        if at:
            pending[at - 1] = rank((parts[at - 1], parts[at]))
        if at < len(pending):
            pending[at] = rank((parts[at], parts[at + 1]))
        if formed is not None:
            formed.add(parts[at])
    return parts


def _plan(pieces: list[str], ranks: dict[tuple[str, str], int], start: int, known: set[str],
          admitted: set[str] | None = None) -> tuple[list[tuple[str, str]], set[str]]:
    """New merges, ranked from `start` after `ranks`, that BPE follows to join `pieces` into
    one token, and every token it forms on the way.

    The cheapest derivation is used when BPE follows it. With repeated pieces BPE can apply
    a planned merge at another occurrence first (' ha' + ' ha' fires on the left pair of
    ' ha ha ha'), and then the token is built in the order BPE applies merges: each step
    adds the leftmost pair inside a word, or else the leftmost pair of whole words."""
    plan = [p for p in dict.fromkeys(_derive(pieces, known, admitted)[0]) if p not in ranks]
    after = {pair: start + i for i, pair in enumerate(plan)}
    formed = set(pieces)
    if len(_bpe(pieces, ranks, after, formed)) == 1:
        return plan, formed
    bounds = _bounds("".join(pieces))
    plan, after, formed, parts = [], {}, set(pieces), list(pieces)
    while len(parts) > 1:
        offs = list(accumulate(map(len, parts), initial=0))
        inside = [i for i in range(len(parts) - 1) if offs[i + 1] not in bounds]
        whole = [i for i in range(len(parts) - 1) if offs[i] in bounds and offs[i + 2] in bounds]
        i = (inside or whole or [0])[0]
        after[(parts[i], parts[i + 1])] = start + len(plan)
        plan.append((parts[i], parts[i + 1]))
        parts = _bpe(parts, ranks, after, formed)
    return plan, formed


def _derive(pieces: list[str], known: set[str],
            admitted: set[str] | None = None) -> tuple[list[tuple[str, str]], int, int]:
    """Cheapest binary derivation joining `pieces` into one token.

    Returns its merges bottom-up, the number of new tokens it creates, and the number of
    new multi-word intermediates outside `admitted`. An intermediate that covers part of
    one word and part of another is never chosen, since words can always be completed
    before they are joined; then the fewest intermediates outside `admitted` win, then
    the fewest new tokens, so existing tokens are reused wherever a split allows."""
    k = len(pieces)
    offs = list(accumulate(map(len, pieces), initial=0))
    text = "".join(pieces)
    bounds = _bounds(text)
    edge = [o in bounds for o in offs]
    # The first pre-token boundary past each piece start; (i, j) spans words when it
    # comes before offs[j].
    ordered = sorted(bounds) + [np.inf]
    beyond = [ordered[bisect_right(ordered, o)] for o in offs]
    # A cost (fragments, bad, new) is one integer in base k: no count exceeds the k - 1
    # merges of a derivation, so sums compare as the triples do.
    cost = [[0] * (k + 1) for _ in range(k + 1)]
    split = [[0] * (k + 1) for _ in range(k + 1)]
    for size in range(2, k + 1):
        for i in range(k - size + 1):
            j = i + size
            s = text[offs[i]:offs[j]]
            spans_words = beyond[i] < offs[j]
            fragment = spans_words and not (edge[i] and edge[j])
            new = s not in known
            bad = (new and size < k and admitted is not None and s not in admitted
                   and spans_words)
            # The first cheapest split, as `min` keeps.
            c, m = min((cost[i][m] + cost[m][j], m) for m in range(i + 1, j))
            cost[i][j] = c + (fragment * k + bad) * k + new
            split[i][j] = m
    plan: list[tuple[str, str]] = []

    def emit(i: int, j: int) -> str:
        if j - i == 1:
            return pieces[i]
        m = split[i][j]
        left, right = emit(i, m), emit(m, j)
        plan.append((left, right))
        return left + right

    emit(0, k)
    return plan, cost[0][k] % k, cost[0][k] // k % k


def _bounds(text: str) -> set[int]:
    """Character offsets where the pre-tokens of `text` start or end."""
    return set(accumulate(map(len, _CHUNK.findall(text)), initial=0))


def _valid_surface(surface: str) -> bool:
    return not ("\n" in surface or "\t" in surface or "  " in surface
                or any(map(str.isdigit, surface)))


def _extract(tk: Tokenizer) -> tuple[dict[str, int], list[tuple[str, str]]]:
    state = json.loads(tk.to_str())["model"]
    if state["merges"] and not isinstance(state["merges"][0], list):
        raise RuntimeError("tokenizers >= 0.20 is required: older releases serialize "
                           "merges as space-joined strings, ambiguous for tokens with spaces")
    return dict(state["vocab"]), [tuple(m) for m in state["merges"]]


def spans_boundary(token: str) -> bool:
    """True when a token joins words across a single space, not merely whitespace."""
    body = token.removeprefix(" ")
    parts = body.split(" ")
    return len(parts) > 1 and all(p and not p.isspace() for p in parts)


def train_tokenizer(source, special_tokens=None, segmentation: str = "merges",
                    **kwargs) -> Tokenizer:
    """Train through the Figure 1 curriculum and return a ready tokenizer.

    `segmentation="merges"` applies the merges in rank order, as BPE does and as the paper
    evaluates; "shortest" encodes every text in the fewest tokens of the same vocabulary."""
    if segmentation not in ("merges", "shortest"):
        raise ValueError("segmentation must be 'merges' or 'shortest'")
    trainer = SupraTokTrainer(special_tokens=special_tokens, **kwargs)
    vocab, merges = trainer.train(source)
    normalizer = hamza_normalizer() if trainer.normalize_hamza else None
    if segmentation == "shortest":
        tok = build_shortest_tokenizer(vocab, trainer.special_tokens, normalizer=normalizer)
    else:
        tok = build_bpe_tokenizer(vocab, merges, trainer.special_tokens, normalizer=normalizer)
    tok.supratok_stats = trainer.stats
    return tok
