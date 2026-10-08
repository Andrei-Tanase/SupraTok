"""Ablation benchmark for Table 3.

Trains on the first --docs Nemotron-CC documents (Table 3 uses 50,000) at --vocab tokens,
with the full configuration entropy-filtered, as in Table 3, and removes one component per
row. Reports characters per token on WikiText-103 validation as distributed (raw), on a
detokenized copy, and on the next 5,000 Nemotron documents (held-out), with each row's
change from the full configuration.

"- curriculum" is one BPE run whose merges may cross whitespace from the start: the
SuperBPE pipeline with no inherited merges, on the same filtered documents. That pipeline
differs from SupraTok in more than its curriculum, so SuperBPE with its own curriculum is
trained too, and the curriculum's effect inside that one pipeline is printed after the
table. "BPE-Standard" stands in for Table 3's "Minimal BPE".

    python benchmarks/ablation_benchmark.py --docs 50000 --vocab 100000
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
from figure1_benchmark import bpe_standard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=int, default=50000)
    parser.add_argument("--vocab", type=int, default=100000)
    args = parser.parse_args()
    for path in (TRAIN, EVAL):
        if not os.path.exists(path):
            sys.exit(f"missing dataset: {path} (see datasets/README.md)")

    from superbpe import train_superbpe

    from supratok import train_tokenizer
    from supratok.data_loader import ThreeBinSampler

    docs = nemotron_documents(0, args.docs)
    sampler = ThreeBinSampler()  # the same selection as use_entropy_filtering=True
    filtered = [d for d in docs if sampler.should_sample(d)]
    sets = evaluation_sets(args.docs)
    filt = {"use_entropy_filtering": True}
    rows = [
        ("SupraTok, entropy-filtered", lambda: train_tokenizer(
            docs, vocab_size=args.vocab, **filt)),
        ("- curriculum", lambda: train_superbpe(filtered, args.vocab, transition=0.0)),
        ("- cross-boundary processing", lambda: train_tokenizer(
            docs, vocab_size=args.vocab, enable_seeds=False, enable_multiword=False,
            enable_expression=False, **filt)),
        ("- entropy filtering", lambda: train_tokenizer(docs, vocab_size=args.vocab)),
        ("- script-aware processing", lambda: train_tokenizer(
            docs, vocab_size=args.vocab, script_aware=False, **filt)),
        ("- PMI seed phrases", lambda: train_tokenizer(
            docs, vocab_size=args.vocab, enable_seeds=False, **filt)),
        ("BPE-Standard", lambda: bpe_standard(docs, args.vocab)),
        ("SuperBPE, entropy-filtered", lambda: train_superbpe(filtered, args.vocab)),
    ]
    print(f"\n{len(docs):,} Nemotron documents ({len(filtered):,} after entropy filtering)"
          f" -> {args.vocab:,} vocabulary\n")
    print(f"  {'configuration':30} " + " ".join(f"{name:>10}" for name in sets)
          + "   change from full   seconds")
    results = {}
    for name, build in rows:
        start = time.time()
        tokenizer = build()
        elapsed = time.time() - start
        scores = results[name] = {k: chars_per_token(tokenizer, v) for k, v in sets.items()}
        full = next(iter(results.values()))
        change = " / ".join(f"{100 * (scores[k] / full[k] - 1):+.1f}%" for k in sets)
        print(f"  {name:30} " + " ".join(f"{c:10.4f}" for c in scores.values())
              + f"   {change if scores is not full else '':>20}  {elapsed:6.0f}", flush=True)
    within = " / ".join(
        f"{100 * (results['- curriculum'][k] / results['SuperBPE, entropy-filtered'][k] - 1):+.1f}%"
        for k in sets)
    print(f"\n  curriculum inside the SuperBPE pipeline (transition 0 vs 0.78125): {within}\n")


if __name__ == "__main__":
    main()
