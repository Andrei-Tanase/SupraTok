"""Multilingual benchmark for Tables 1 and 6.

CulturaX is gated, so Wikipedia stands in for it: the first 8,000 documents per language
(datasets/README.md), each cut to at most 4,000 characters. Compression is measured on
FLORES-200 devtest, sentence by sentence, at a 100k vocabulary.

    python benchmarks/multilingual_benchmark.py --table 1   # 7 languages, with vs without
                                                            # script-aware processing
    python benchmarks/multilingual_benchmark.py --table 6   # 14 languages, against
                                                            # BPE-Standard and SuperBPE
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from evaluation import chars_per_token

WIKI = os.path.join(REPO, "datasets/wikipedia")
FLORES = os.path.join(REPO, "datasets/flores200/devtest")
CODES = {"en": "eng_Latn", "de": "deu_Latn", "nl": "nld_Latn", "fr": "fra_Latn",
         "es": "spa_Latn", "it": "ita_Latn", "ru": "rus_Cyrl", "uk": "ukr_Cyrl",
         "pl": "pol_Latn", "zh": "zho_Hans", "ja": "jpn_Jpan", "ko": "kor_Hang",
         "tr": "tur_Latn", "fi": "fin_Latn", "th": "tha_Thai", "ar": "arb_Arab"}
TABLES = {1: ["zh", "ja", "th", "ko", "tr", "fi", "ar"],
          6: ["en", "de", "nl", "fr", "es", "it", "ru", "uk", "pl", "zh", "ja", "ko", "tr", "fi"]}
PER_LANGUAGE, MAX_CHARS = 8000, 4000


def documents(langs: list[str]) -> list[tuple[str, str]]:
    """(text, language) pairs, interleaved across languages, then shuffled with seed 0."""
    by_lang = {}
    for lang in langs:
        texts = []
        with open(os.path.join(WIKI, f"{lang}.jsonl"), encoding="utf-8") as handle:
            for line in handle:
                if len(texts) == PER_LANGUAGE:
                    break
                text = json.loads(line)["text"]
                if len(text) > MAX_CHARS:
                    cut = text.rfind("\n", 0, MAX_CHARS)
                    text = text[:cut if cut > MAX_CHARS // 2 else MAX_CHARS]
                texts.append(text)
        by_lang[lang] = texts
    docs = [(by_lang[lang][i], lang) for i in range(PER_LANGUAGE) for lang in langs
            if i < len(by_lang[lang])]
    random.Random(0).shuffle(docs)
    return docs


def flores(lang: str) -> list[str]:
    with open(os.path.join(FLORES, CODES[lang] + ".devtest"), encoding="utf-8") as handle:
        return handle.read().splitlines()


def bpe_multilingual(texts: list[str], vocab_size: int):
    """Appendix C: BPE-Standard, which in the multilingual benchmark adds NFD."""
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE(unk_token="<unk>", end_of_word_suffix="</w>"))
    tok.normalizer = normalizers.NFD()
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator(texts, trainers.BpeTrainer(
        vocab_size=vocab_size, show_progress=False, special_tokens=["<unk>"],
        end_of_word_suffix="</w>"))
    return tok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=int, choices=sorted(TABLES), required=True)
    parser.add_argument("--vocab", type=int, default=100000)
    args = parser.parse_args()
    langs = TABLES[args.table]
    missing = [p for p in [os.path.join(WIKI, f"{lang}.jsonl") for lang in langs]
               + [os.path.join(FLORES, CODES[lang] + ".devtest") for lang in langs]
               if not os.path.exists(p)]
    if missing:
        sys.exit(f"missing dataset: {missing[0]} (see datasets/README.md)")

    from superbpe import train_superbpe

    from supratok import train_tokenizer

    docs = documents(langs)
    texts = [text for text, _ in docs]
    if args.table == 1:
        configs = [("without", lambda: train_tokenizer(docs, vocab_size=args.vocab,
                                                       script_aware=False)),
                   ("with", lambda: train_tokenizer(docs, vocab_size=args.vocab,
                                                    normalize_hamza=True))]
    else:
        configs = [("BPE-Standard", lambda: bpe_multilingual(texts, args.vocab)),
                   ("SuperBPE", lambda: train_superbpe(texts, args.vocab)),
                   ("SupraTok", lambda: train_tokenizer(docs, vocab_size=args.vocab))]
    print(f"\nTable {args.table}: {len(docs):,} documents, {len(langs)} languages -> "
          f"{args.vocab:,} vocabulary; FLORES-200 devtest characters per token\n")
    scores = {}
    for name, build in configs:
        start = time.time()
        tokenizer = build()
        scores[name] = {lang: chars_per_token(tokenizer, flores(lang)) for lang in langs}
        print(f"  {name:14} " + " ".join(f"{lang}:{c:.3f}" for lang, c in scores[name].items())
              + f"   ({time.time() - start:.0f} s)", flush=True)
    baseline = "without" if args.table == 1 else "BPE-Standard"
    for name in scores:
        if name == baseline:
            continue
        gains = {lang: 100 * (scores[name][lang] / scores[baseline][lang] - 1) for lang in langs}
        print(f"\n  {name} vs {baseline}: macro {sum(gains.values()) / len(gains):+.1f}%  "
              + " ".join(f"{lang}:{g:+.1f}" for lang, g in gains.items()))
    print()


if __name__ == "__main__":
    main()
