"""Section 3.6: export a trained SupraTok tokenizer as a SentencePiece model.

Requires the optional `sentencepiece` and `protobuf` packages.

A merge-order (BPE) tokenizer exports as a SentencePiece BPE model, a shortest-path
(Unigram) tokenizer as a SentencePiece UNIGRAM model with the same piece scores, so every
text costs the same number of tokens; where several segmentations tie, the two may pick
different ones.

SentencePiece's BPE encoder merges the adjacent pair whose concatenation is the
highest-scoring piece, so each merge output is scored by its merge rank. This reproduces
the HuggingFace segmentation wherever a token is only ever formed by its own merges. A
token that two adjacent tokens could also spell without such a merge (" if you" + " are"
for " if you are", formed as " if" + " you are") can segment differently: SentencePiece
reaches it, the merge order may not; on Nemotron-CC text about one document in ten
segments differently, the SentencePiece model using under 0.1% fewer tokens. SentencePiece
writes the space as U+2581, so input that itself contains U+2581 is read as containing a space;
the HuggingFace tokenizer keeps that character, through byte fallback if need be.
"""

from __future__ import annotations

import json

from tokenizers import Tokenizer

from .hf_build import BYTE_TOKENS

__all__ = ["to_sentencepiece"]

SPACE = "▁"


def to_sentencepiece(tokenizer: Tokenizer, path: str | None = None) -> bytes:
    """Serialize `tokenizer` as a SentencePiece model; write it to `path` if given.

    Pieces keep the tokenizer's ids, the byte-fallback tokens become BYTE pieces, added
    tokens become USER_DEFINED pieces, and the UNKNOWN piece is appended last, named `<unk>`
    unless a token already is (a Unigram model's own unknown piece is kept instead when it
    is not a byte-fallback piece). A token containing U+2581 (SentencePiece's space)
    or NUL cannot be expressed and is kept as an UNUSED placeholder.

    SentencePiece matches added tokens in one longest-first pass, while HuggingFace
    first matches those with `normalized=False` (special tokens by default), then the
    rest, and honors `lstrip`, `rstrip` and `single_word`. An added token whose matching
    therefore differs is rejected.
    """
    try:
        from sentencepiece import sentencepiece_model_pb2 as pb
    except ImportError as exc:
        hint = "SentencePiece export needs `pip install sentencepiece protobuf`"
        raise ImportError(hint) from exc

    state = json.loads(tokenizer.to_str())
    if state.get("normalizer") is not None:
        raise ValueError("SentencePiece export supports tokenizers without a normalizer")
    model = state["model"]
    if not model.get("byte_fallback"):
        raise ValueError("expected a SupraTok byte-fallback tokenizer")
    vocab, scores, unknown = _model_pieces(model)
    if " " not in vocab:
        raise ValueError("the space must be in the vocabulary: SentencePiece writes it as "
                         "the piece U+2581")
    added, first = set(), set()
    for item in state.get("added_tokens", []):
        if item.get("lstrip") or item.get("rstrip") or item.get("single_word"):
            raise ValueError(f"added token {item['content']!r}: SentencePiece has no lstrip, "
                             "rstrip or single_word matching")
        vocab.setdefault(item["content"], item["id"])
        added.add(item["content"])
        if not item.get("normalized"):
            first.add(item["content"])
    for token in added - first:
        clash = next((t for t in first if _overlap(token, t)), None)
        if clash is not None:
            raise ValueError(f"added tokens {token!r} and {clash!r} overlap but are matched in "
                             "separate HuggingFace passes, which SentencePiece cannot reproduce")
    by_id = sorted(vocab.items(), key=lambda kv: kv[1])
    if [i for _, i in by_id] != list(range(len(by_id))):
        raise ValueError("token ids must be contiguous from 0")

    piece = pb.ModelProto.SentencePiece
    proto = pb.ModelProto()
    seen = set()
    byte_set = set(BYTE_TOKENS)
    for token, i in by_id:
        entry = proto.pieces.add()
        if SPACE in token or "\x00" in token:
            # SentencePiece reads U+2581 as the space and rejects NUL in a piece.
            entry.piece, entry.type = f"<unused{i}>", piece.UNUSED
        elif token in byte_set:
            entry.piece, entry.type = token, piece.BYTE
        elif token in added:
            entry.piece, entry.type = token.replace(" ", SPACE), piece.USER_DEFINED
        elif token == unknown:
            entry.piece, entry.type = token, piece.UNKNOWN
        else:
            entry.piece, entry.type = token.replace(" ", SPACE), piece.NORMAL
            entry.score = scores.get(token, 0.0)
        if entry.piece in seen:
            raise ValueError(f"two tokens map to the SentencePiece piece {entry.piece!r}")
        seen.add(entry.piece)
    if unknown is None:
        unknown, n = "<unk>", 0
        while unknown in seen:
            n += 1
            unknown = f"<unk{n}>"
        unk = proto.pieces.add()
        unk.piece, unk.type = unknown, piece.UNKNOWN
        unk_id = len(proto.pieces) - 1
    else:
        unk_id = vocab[unknown]

    spec = proto.trainer_spec
    spec.model_type = pb.TrainerSpec.UNIGRAM if model["type"] == "Unigram" else pb.TrainerSpec.BPE
    spec.vocab_size = len(proto.pieces)
    spec.byte_fallback = True
    spec.split_by_whitespace = False
    spec.split_by_unicode_script = False
    spec.split_by_number = False
    spec.allow_whitespace_only_pieces = True
    spec.unk_id = unk_id
    spec.unk_piece = unknown
    spec.bos_id = spec.eos_id = spec.pad_id = -1
    norm = proto.normalizer_spec
    norm.name = "identity"
    norm.add_dummy_prefix = False
    norm.remove_extra_whitespaces = False
    norm.escape_whitespaces = True

    data = proto.SerializeToString()
    if path is not None:
        with open(path, "wb") as handle:
            handle.write(data)
    return data


def _model_pieces(model: dict) -> tuple[dict[str, int], dict[str, float], str | None]:
    """The model's {piece: id}, the score of each NORMAL piece, and its UNKNOWN piece, or
    None when SentencePiece needs one appended: always for BPE, and for a Unigram model
    whose unknown id is a byte-fallback piece, as SupraTok's shortest-path model's is. A
    BPE merge output scores by rank, since SentencePiece's BPE encoder merges the
    highest-scoring pair first."""
    if model.get("type") == "BPE":
        rank: dict[str, int] = {}
        for i, (left, right) in enumerate(model["merges"]):
            rank.setdefault(left + right, i)
        return dict(model["vocab"]), {t: -float(r + 1) for t, r in rank.items()}, None
    if model.get("type") == "Unigram":
        table = model["vocab"]
        unknown = table[model["unk_id"]][0]
        return ({p: i for i, (p, _) in enumerate(table)}, {p: float(v) for p, v in table},
                None if unknown in BYTE_TOKENS else unknown)
    raise ValueError("expected a SupraTok BPE or Unigram tokenizer")


def _overlap(a: str, b: str) -> bool:
    """True when one string contains the other or a proper suffix of one starts the other."""
    return (a in b or b in a
            or any(a.endswith(b[:k]) or b.endswith(a[:k]) for k in range(1, min(len(a), len(b)))))
