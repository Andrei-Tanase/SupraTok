"""Compression benchmark for the Figure 1 curriculum (Tables 4A, 5 and 7).

Trains every tokenizer on the first --docs Nemotron-CC documents and reports characters per
token on WikiText-103 validation as distributed (raw), on a detokenized copy (WikiText's
tokenization spaces removed), and on the next 5,000 Nemotron documents (held-out) when the
file has them. Baselines are BPE-Standard (Appendix C) and SuperBPE (see superbpe.py).
Without --quick it also prints Figure 2's segmentation, the word-bounded ceiling of each
evaluation set and the Eq. 1 entropy statistics of the training documents.

    python benchmarks/figure1_benchmark.py
    python benchmarks/figure1_benchmark.py --docs 100000 --vocab 100000           # Table 4A
    python benchmarks/figure1_benchmark.py --docs 100000 --vocab 300000 --quick   # Table 5
    python benchmarks/figure1_benchmark.py --docs 10000 --vocab 64000             # Table 7
"""

from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from evaluation import (
    EVAL,
    TRAIN,
    chars_per_token,
    evaluation_sets,
    nemotron_documents,
)

FIGURE2 = "The United Nations headquarters in New York announced new climate change initiatives."


def _peak_gb() -> float:
    """Peak RSS. ru_maxrss is bytes on macOS and kilobytes on Linux."""
    try:
        import resource
    except ImportError:
        return float("nan")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 ** 3) if sys.platform == "darwin" else peak / (1024 ** 2)


def _cross_share(tokenizer) -> float:
    """Share of the vocabulary that joins words across a space."""
    from supratok.trainer import spans_boundary
    vocab = tokenizer.get_vocab()
    # Byte-level vocabularies (SuperBPE) write the space as U+0120.
    return sum(spans_boundary(t.replace("Ġ", " ")) for t in vocab) / len(vocab)


def bpe_standard(docs, vocab_size):
    """Appendix C: HuggingFace BPE, Whitespace() pre-tokenization, '</w>' suffix."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE(unk_token="<unk>", end_of_word_suffix="</w>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator(docs, trainers.BpeTrainer(
        vocab_size=vocab_size, show_progress=False, special_tokens=["<unk>"],
        end_of_word_suffix="</w>"))
    return tok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=int, default=20000)
    parser.add_argument("--vocab", type=int, default=100000)
    parser.add_argument("--no-superbpe", action="store_true", help="skip the SuperBPE baseline")
    parser.add_argument("--screening-pmi", type=float, default=5.0,
                        help="SupraTok's later-pass screening threshold (paper: 5.0)")
    parser.add_argument("--quick", action="store_true",
                        help="only BPE-Standard, SuperBPE and the full SupraTok (the sweep)")
    args = parser.parse_args()
    for path in (TRAIN, EVAL):
        if not os.path.exists(path):
            sys.exit(f"missing dataset: {path} (see datasets/README.md)")

    from superbpe import train_superbpe

    from supratok import train_tokenizer

    docs = nemotron_documents(0, args.docs)
    sets = evaluation_sets(args.docs)
    print(f"\n{len(docs):,} Nemotron documents -> {args.vocab:,} vocabulary\n")
    print(f"  {'tokenizer':32} " + " ".join(f"{name:>10}" for name in sets)
          + "  cross-boundary   seconds  process peak GB")

    configs = [("BPE-Standard (Appendix C)", lambda: bpe_standard(docs, args.vocab))]
    if not args.no_superbpe:
        configs.append(("SuperBPE", lambda: train_superbpe(docs, args.vocab)))
    for name, options in [
            ("SupraTok TRADITIONAL", {"enable_seeds": False, "enable_multiword": False,
                                      "enable_expression": False}),
            ("SupraTok +seeds", {"enable_multiword": False, "enable_expression": False}),
            ("SupraTok +MULTIWORD", {"enable_expression": False}),
            ("SupraTok (full)", {}),
            ("SupraTok, shortest-path", {"segmentation": "shortest"}),
            ("SupraTok, entropy-filtered", {"use_entropy_filtering": True})]:
        if args.quick and name != "SupraTok (full)":
            continue
        configs.append((name, lambda o=options: train_tokenizer(
            docs, vocab_size=args.vocab, screening_pmi=args.screening_pmi, **o)))

    scores = {}
    for name, build in configs:
        start = time.time()
        tokenizer = build()
        elapsed = time.time() - start
        if name == "SupraTok (full)":
            figure2 = tokenizer.encode(FIGURE2).tokens
        scores[name] = {k: chars_per_token(tokenizer, v) for k, v in sets.items()}
        print(f"  {name:32} " + " ".join(f"{c:10.4f}" for c in scores[name].values())
              + f"  {100 * _cross_share(tokenizer):13.1f}%  {elapsed:8.0f}  {_peak_gb():15.2f}",
              flush=True)

    full = scores["SupraTok (full)"]
    for baseline in ("BPE-Standard (Appendix C)", "SuperBPE"):
        if baseline in scores:
            gains = "  ".join(f"{k} {100 * (full[k] / v - 1):+.1f}%"
                              for k, v in scores[baseline].items())
            print(f"\n  SupraTok vs {baseline}: {gains}")
    if not args.quick:
        from supratok.data_loader import ThreeBinSampler
        print(f"\n  Figure 2 (paper: 9 tokens): {len(figure2)} tokens: "
              + " | ".join(t.strip() or repr(t) for t in figure2))
        ceilings = "  ".join(f"{k} {sum(map(len, v)) / sum(len(t.split()) for t in v):.2f}"
                             for k, v in sets.items())
        print(f"  word-bounded ceiling (characters per whitespace-delimited word): {ceilings}")
        sampler = ThreeBinSampler()  # the same selection as use_entropy_filtering=True
        kept = sum(map(sampler.should_sample, docs))
        print(f"  Eq. 1: {sampler.bin_counts['high'] / len(docs):.1%} of documents above 4.5 bits;"
              f" the Section 3.2 retention keeps {kept:,} ({kept / len(docs):.1%})")
    print()


if __name__ == "__main__":
    main()
