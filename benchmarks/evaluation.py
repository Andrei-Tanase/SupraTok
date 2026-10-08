"""Evaluation sets and the characters-per-token metric shared by the benchmarks."""

from __future__ import annotations

import json
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(REPO, "datasets/nemotron/nemotron_scale_1.jsonl")
EVAL = os.path.join(REPO, "datasets/wikitext103/wiki.valid.txt")


def detokenize_wikitext(text: str) -> str:
    """Undo WikiText-103's tokenization spaces: 'York , don 't' becomes 'York, don't'.

    Double quotes, and the apostrophe of a plural possessive ("players '"), stay spaced:
    telling an opening quote from a closing one needs more context than a substitution."""
    text = text.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    text = re.sub(r" ([.,;:!?)\]}%])", r"\1", text)
    text = re.sub(r"([(\[{$]) ", r"\1", text)
    return re.sub(r" ('t|n't|'s|'re|'ve|'m|'ll|'d)\b", r"\1", text)


def nemotron_documents(start: int, stop: int) -> list[str]:
    """Stripped, non-empty texts of lines [start, stop) of the Nemotron-CC subset."""
    out = []
    with open(TRAIN, encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if i >= stop:
                break
            if i >= start:
                text = json.loads(line).get("text", "").strip()
                if text:
                    out.append(text)
    return out


def evaluation_sets(train_docs: int, heldout_docs: int = 5000) -> dict[str, list[str]]:
    """WikiText-103 validation as distributed and detokenized, line by line, and the
    Nemotron documents that follow the training range, when the file has them."""
    with open(EVAL, encoding="utf-8") as handle:
        raw = handle.read()
    sets = {"wiki raw": raw.splitlines(keepends=True),
            "wiki detok": detokenize_wikitext(raw).splitlines(keepends=True)}
    held = nemotron_documents(train_docs, train_docs + heldout_docs)
    if held:
        sets["held-out"] = held
    return sets


def chars_per_token(tokenizer, texts: list[str]) -> float:
    """Characters over tokens, each text encoded on its own."""
    tokens = sum(len(e.ids) for e in tokenizer.encode_batch(texts, add_special_tokens=False))
    return sum(map(len, texts)) / max(tokens, 1)
