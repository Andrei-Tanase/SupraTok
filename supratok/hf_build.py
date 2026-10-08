"""build_bpe_tokenizer: emit a HuggingFace `tokenizers` BPE Tokenizer from a SupraTok
(vocab, merges) pair. build_shortest_tokenizer: encode with the same vocabulary in the
fewest tokens instead of in merge order.

The base units are those of Section 4.1: Unicode characters plus the 256 byte-fallback
symbols, so a character outside the vocabulary is encoded as its UTF-8 bytes. There is no
pre-tokenizer, so merges that span whitespace (" New" + " York") can fire, and the
ByteFallback and Fuse decoders give decode(encode(x)) == x.

Inputs are validated first because the Rust layer panics (pyo3_runtime.PanicException, a
BaseException) or silently corrupts on bad input.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from tokenizers import Tokenizer, decoders, models

__all__ = ["InvalidBpeSpec", "build_bpe_tokenizer", "build_shortest_tokenizer"]

BYTE_TOKENS = [f"<0x{b:02X}>" for b in range(256)]
# Every spelling the ByteFallback decoder rewrites as a byte: <0x41>, <0x4a>, <0x+A>.
_BYTE_SPELLING = re.compile(r"<0x(?:[0-9A-Fa-f]{2}|\+[0-9A-Fa-f])>")
# The characters that spell a byte-fallback token. A shortest-path vocabulary must hold them
# all, so that text spelling "<0x41>" can always be read as characters.
_BYTE_SPELLING_CHARS = frozenset("<>x0123456789ABCDEF")
# Scored so that text spelling a byte piece ("<0x41>") is read as characters, never as it.
_BYTE_PIECE_SCORE = -10_000.0


class InvalidBpeSpec(ValueError):
    """Raised for a (vocab, merges) pair the Rust layer would panic on or silently mishandle."""


def _validate(
    vocab: dict[str, int],
    merges: Sequence[tuple[str, str]],
    special_tokens: Sequence[str],
    require_contiguous_ids: bool,
) -> None:
    if not isinstance(vocab, dict) or not vocab or not all(isinstance(t, str) for t in vocab):
        raise InvalidBpeSpec("vocab must be a non-empty dict[str, int]")

    bad = [t for t, i in vocab.items() if not isinstance(i, int) or isinstance(i, bool) or i < 0]
    if bad:
        raise InvalidBpeSpec(f"vocab ids must be non-negative ints; offenders: {bad[:5]}")

    # The Rust layer accepts duplicate ids and its reverse map keeps one token per id, so
    # encode('a') can decode as 'ab'.
    seen: dict[int, str] = {}
    dupes = []
    for tok, i in vocab.items():
        if i in seen:
            dupes.append((seen[i], tok, i))
        else:
            seen[i] = tok
    if dupes:
        raise InvalidBpeSpec(
            f"{len(dupes)} duplicate id(s) in vocab -> silent decode corruption; e.g. {dupes[:3]}"
        )

    # A missing byte token makes the characters that need it unencodable: they are
    # silently dropped (unk_token=None), so the tokenizer stops round-tripping.
    missing = [t for t in BYTE_TOKENS if t not in vocab]
    if missing:
        raise InvalidBpeSpec(
            f"vocab is missing {len(missing)} of the 256 byte-fallback tokens; "
            f"first missing: {missing[:4]!r}"
        )

    if require_contiguous_ids:
        n = len(vocab)
        mx = max(vocab.values())
        if mx != n - 1:
            # Ids are unique, so the smallest holes lie below n.
            holes = sorted(set(range(n)) - set(vocab.values()))[:8]
            raise InvalidBpeSpec(
                f"vocab ids must be contiguous 0..{n - 1}; got {n} tokens with max id {mx}. "
                f"{mx + 1 - n} hole(s), first: {holes}"
            )

    byte_set = set(BYTE_TOKENS)
    spelled = [t for t in vocab if t not in byte_set and _BYTE_SPELLING.fullmatch(t)]
    if spelled:
        raise InvalidBpeSpec(f"tokens spelled like byte-fallback tokens: {spelled[:5]}")

    seen_pairs: dict[tuple[str, str], int] = {}
    for rank, m in enumerate(merges):
        if not (isinstance(m, (tuple, list)) and len(m) == 2):
            raise InvalidBpeSpec(f"merge #{rank} is not a 2-element pair: {m!r}")
        left, right = m
        if not isinstance(left, str) or not isinstance(right, str):
            raise InvalidBpeSpec(f"merge #{rank} components must be str: {m!r}")
        if not left or not right:
            raise InvalidBpeSpec(f"merge #{rank} has an empty component: {m!r}")
        # Rust accepts a repeated pair and lets the last occurrence overwrite the rank.
        if (left, right) in seen_pairs:
            raise InvalidBpeSpec(
                f"duplicate merge pair {(left, right)!r} at ranks "
                f"{seen_pairs[(left, right)]} and {rank}"
            )
        seen_pairs[(left, right)] = rank
        if left not in vocab or right not in vocab:
            raise InvalidBpeSpec(f"merge #{rank} {m!r}: a component is not in vocab")
        # The ByteFallback decoder rewrites any token spelled <0xNN>, so a merge may
        # neither consume one nor produce one.
        if left in byte_set or right in byte_set or left + right in byte_set:
            raise InvalidBpeSpec(f"merge #{rank} {m!r} involves a byte-fallback token")
        # A merge output missing from vocab makes BpeBuilder::build panic or abort.
        if left + right not in vocab:
            raise InvalidBpeSpec(f"merge #{rank} {m!r}: output {left + right!r} not in vocab")

    for st in special_tokens:
        if not isinstance(st, str) or not st:
            raise InvalidBpeSpec(f"special token must be a non-empty str: {st!r}")
        if _BYTE_SPELLING.fullmatch(st):
            raise InvalidBpeSpec(f"special token {st!r} is spelled like a byte-fallback token")


def build_bpe_tokenizer(
    vocab: dict[str, int],
    merges: Iterable[tuple[str, str]],
    special_tokens: Iterable[str],
    *,
    normalizer: object | None = None,
    require_contiguous_ids: bool = True,
) -> Tokenizer:
    """Build a character-level, byte-fallback, cross-boundary BPE Tokenizer.

    Args:
        vocab: {token_str: id}. Must contain the 256 byte-fallback tokens, every merge
            component and every merge output. Ids must be unique and (by default)
            contiguous from 0.
        merges: ordered [(left, right), ...]. Index == merge rank; lower wins.
        special_tokens: registered as special AddedTokens. A special token already in
            `vocab` keeps its id; otherwise it is appended after the largest id.
        normalizer: optional `tokenizers` normalizer applied before encoding.
        require_contiguous_ids: reject ids with gaps. Without it a gap is kept, and a
            token added later with `add_tokens` can be given an id already in use.

    Returns:
        A `tokenizers.Tokenizer` with exact `decode(encode(x)) == x` round-trip when no
        normalizer is set and `x` holds no special token.

    Raises:
        InvalidBpeSpec: for any input the Rust layer would panic on or mishandle.
    """
    merges = list(merges)
    if isinstance(special_tokens, (str, bytes)):
        raise TypeError("special_tokens takes a sequence of strings, not a string")
    special_tokens = list(special_tokens)
    _validate(vocab, merges, special_tokens, require_contiguous_ids)
    # Merge order emits a token only through a merge forming it.
    exempt = {left + right for left, right in merges} | set(BYTE_TOKENS) | set(special_tokens)
    unformed = [t for t in vocab if len(t) > 1 and t not in exempt]
    if unformed:
        raise InvalidBpeSpec(f"{len(unformed)} token(s) no merge forms, which merge order "
                             f"never emits; e.g. {unformed[:5]}")
    merges = [(a, b) for a, b in merges]
    special_tokens = list(dict.fromkeys(special_tokens))

    vocab = dict(vocab)
    next_id = max(vocab.values()) + 1
    for st in special_tokens:
        if st not in vocab:
            vocab[st] = next_id
            next_id += 1

    # Built from objects rather than Tokenizer.from_str(json), whose schema changes across
    # tokenizers versions. `merges` must be a list of tuples; lists raise TypeError.
    model = models.BPE(
        vocab=vocab,
        merges=merges,
        unk_token=None,      # unreachable: every character has a byte-fallback path
        fuse_unk=False,
        byte_fallback=True,
        ignore_merges=False,  # True looks the whole input up in vocab before merging
        dropout=None,
    )

    tok = Tokenizer(model)
    if normalizer is not None:
        tok.normalizer = normalizer
    tok.decoder = decoders.Sequence([decoders.ByteFallback(), decoders.Fuse()])

    if special_tokens:
        tok.add_special_tokens(special_tokens)

    return tok


def build_shortest_tokenizer(
    vocab: dict[str, int],
    special_tokens: Iterable[str],
    *,
    normalizer: object | None = None,
) -> Tokenizer:
    """Build a tokenizer that encodes every text in the fewest tokens of `vocab`.

    A Unigram model in which every piece scores -1 returns, by Viterbi search, a
    segmentation with the minimum number of tokens; ties are broken by the implementation.
    Byte-fallback pieces score -10,000, so text that spells one is read as characters. The
    model's unknown id points at the `<0x00>` piece and is never emitted, because byte
    fallback covers every character. Token ids and the vocabulary size match
    `build_bpe_tokenizer` for the same arguments.

    Raises:
        InvalidBpeSpec: for an invalid vocabulary, or one missing a character that spells a
            byte-fallback token.
    """
    if isinstance(special_tokens, (str, bytes)):
        raise TypeError("special_tokens takes a sequence of strings, not a string")
    special_tokens = list(dict.fromkeys(special_tokens))
    _validate(vocab, [], special_tokens, require_contiguous_ids=True)
    missing = sorted(_BYTE_SPELLING_CHARS - set(vocab))
    if missing:
        raise InvalidBpeSpec(f"shortest-path encoding needs the characters {missing} in vocab: "
                             "without them text spelling a byte token would not round-trip")
    pieces = [t for t, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
    pieces += [t for t in special_tokens if t not in vocab]
    byte_set = set(BYTE_TOKENS)
    table = [(p, _BYTE_PIECE_SCORE if p in byte_set else -1.0) for p in pieces]
    tok = Tokenizer(models.Unigram(table, unk_id=pieces.index(BYTE_TOKENS[0]),
                                   byte_fallback=True))
    if normalizer is not None:
        tok.normalizer = normalizer
    tok.decoder = decoders.Sequence([decoders.ByteFallback(), decoders.Fuse()])
    if special_tokens:
        tok.add_special_tokens(special_tokens)
    return tok
