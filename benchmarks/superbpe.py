"""SuperBPE baseline (Liu et al., 2025) built with stock HuggingFace `tokenizers`.

The official recipe (github.com/PythonNut/superbpe) has two stages:

  1. byte-level BPE under the stage-1 pre-tokenization regex, to the full vocabulary;
  2. keep the first int(V * transition) merges (transition 0.78125, Appendix C of the
     SupraTok paper) and resume BPE under the stage-2 regex, which no longer splits at
     whitespace, until the vocabulary is full again.

The official stage 2 resumes training in a fork of `tokenizers`. Here stage 2 is emulated
by resuming BPE over the inherited segmentation of each stage-2 pre-token, with every
stage-1 token as one symbol. On 20k Nemotron-CC documents at a 100k vocabulary this shares
99.7% of the official build's vocabulary and matches its characters per token within 0.1% on
WikiText-103 validation (raw and detokenized) and 0.7% on held-out Nemotron documents.
Training reads each document line by line, line breaks included, as the official
scripts read their text files. Like the official trainer, which is given no initial
alphabet, the tokenizer only knows the bytes it saw in training; an unseen byte is dropped
at encode time, which spends no token on it and so slightly favors this baseline.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Sequence

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

STAGE1 = (r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
          r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"
          r"|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+")
STAGE2 = r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]{2,}[\r\n/]*| +(?!\S)"


def _pretokenizer(regex: str) -> pre_tokenizers.PreTokenizer:
    return pre_tokenizers.Sequence([
        pre_tokenizers.Split(pattern=Regex(regex), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False)])


def _lines(documents: Sequence[str]) -> Iterable[str]:
    """Each document's lines, line breaks included."""
    for doc in documents:
        yield from (doc + "\n").splitlines(keepends=True)


def _symbol(i: int) -> str:
    """A distinct code point from U+0100 for symbol i, skipping the surrogates."""
    cp = 0x100 + i
    return chr(cp + 0x800 if cp >= 0xD800 else cp)


def _merges(tokenizer: Tokenizer) -> list[tuple[str, str]]:
    return [tuple(m) for m in json.loads(tokenizer.to_str())["model"]["merges"]]


def train_superbpe(documents: Sequence[str], vocab_size: int,
                   transition: float = 0.78125) -> Tokenizer:
    """Train a SuperBPE tokenizer of `vocab_size` tokens on `documents`."""
    stage1 = Tokenizer(models.BPE())
    stage1.pre_tokenizer = _pretokenizer(STAGE1)
    stage1.train_from_iterator(_lines(documents),
                               trainers.BpeTrainer(vocab_size=vocab_size, show_progress=False))
    merges = _merges(stage1)
    inherited = merges[:int(vocab_size * transition)]
    alphabet = sorted(set(stage1.get_vocab()) - {a + b for a, b in merges})
    vocab = {t: i for i, t in enumerate(alphabet)}
    for a, b in inherited:
        vocab.setdefault(a + b, len(vocab))

    # Stage 2: count the stage-2 chunks and segment each with the inherited merges.
    stage2 = _pretokenizer(STAGE2)
    chunks: Counter[str] = Counter()
    for line in _lines(documents):
        chunks.update(piece for piece, _ in stage2.pre_tokenize_str(line))
    segmenter = Tokenizer(models.BPE(vocab=dict(vocab), merges=list(inherited)))
    keys = list(chunks)
    code: dict[str, str] = {}
    seqs: list[str] = []
    for key, enc in zip(keys, segmenter.encode_batch(keys, add_special_tokens=False)):
        if len(enc.tokens) > 1:
            symbols = "".join(code.setdefault(t, _symbol(len(code))) for t in enc.tokens)
            seqs.extend([symbols] * chunks[key])
    del chunks, keys
    token = {s: t for t, s in code.items()}
    room = vocab_size - len(vocab)
    resumed = Tokenizer(models.BPE())
    resumed.pre_tokenizer = pre_tokenizers.Split("\x00", behavior="removed")
    # A margin for merges skipped below as already inherited.
    resumed.train_from_iterator(
        seqs, trainers.BpeTrainer(vocab_size=len(code) + room + room // 10 + 100,
                                  show_progress=False))
    del seqs
    merges, known = list(inherited), set(inherited)
    for a, b in _merges(resumed):
        if len(vocab) >= vocab_size:
            break
        pair = ("".join(token[s] for s in a), "".join(token[s] for s in b))
        if pair not in known:
            merges.append(pair)
            known.add(pair)
            vocab.setdefault(pair[0] + pair[1], len(vocab))

    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=merges))
    tokenizer.pre_tokenizer = stage2
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer
