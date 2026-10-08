# SupraTok

Source code for the tokenizer in
[SupraTok: Cross-Boundary Tokenization for Enhanced Language Model Performance](docs/tacl.a.774.pdf)
(TACL vol. 14, pp. 1787–1802, 2026). Version 1.0.0.

SupraTok learns tokens that cross whitespace, so recurring multi-word expressions such as
` New York` or ` climate change` become single vocabulary entries. The package implements the
Figure 1 training procedure, Appendix A span scoring, entropy-based data curation and
script-aware processing, with the paper's settings as defaults, and exports to HuggingFace
and SentencePiece formats.

This is a reference implementation based on the paper; details the paper leaves open are
described in [docs/implementation.md](docs/implementation.md).

## Architecture

| step                     | paper                         | what it does                                                                                                                                                                                                                                         |
| ------------------------ | ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| data curation (optional) | §3.2, Eq. 1                   | keeps 10% / 50% / 90% of documents with character-bigram entropy < 3.0 / 3.0–4.5 / > 4.5                                                                                                                                                             |
| initial vocabulary       | Fig. 1 line 3, §4.1           | Unicode characters and the 256 byte-fallback symbols                                                                                                                                                                                                 |
| seed phrases             | Fig. 1 lines 2–3, §3.3, Eq. 2 | word n-grams, n ∈ {2..6}, PMI ≥ 2.0, frequency ≥ 100 on D′, with an overlap control                                                                                                                                                                  |
| TRADITIONAL pass         | §3.3                          | BPE inside one pre-token, a word with the whitespace before it                                                                                                                                                                                       |
| MULTIWORD pass           | §3.3, §3.5                    | BPE resumed across junctions that pass the §3.5 screen (PMI ≥ 5.0, frequency ≥ 5), lie inside a seed, or join a word to what is attached to it; each merge joins the pieces on either side of a junction                                             |
| EXPRESSION pass          | §3.3, Appendix A              | 2–5 word spans scored by Eqs. 5–12 and accepted when Net(s) > 0 under leaf surgery (Eqs. 13–14), in chunks kept while they lower the token count of a verification shard                                                                             |
| script-aware processing  | §3.4, Appendix E              | supercategory pre-tokens (a punctuation run keeps the line breaks after it); character-bigram gate (min count 50) for Chinese, Japanese and Thai; suffix-aware weighting from 200 mined suffixes for Turkish and Finnish; Arabic hamza normalization |

Each pass continues from the vocabulary the previous one left, and the pass-1 merges that
form words rank before every merge that joins words.

## Install

```bash
pip install -e .                  # tokenizer training
pip install -e ".[sentencepiece]" # adds SentencePiece export
```

Requires Python 3.9+, pip 21.3+ for the editable install, and `tokenizers` 0.20 or later.

## Quick start

```python
from supratok import train_tokenizer

def documents():
    with open("corpus.txt", encoding="utf-8") as handle:
        for line in handle:
            yield line

tokenizer = train_tokenizer(documents, vocab_size=100000)
tokenizer.save("tokenizer.json")
```

`train_tokenizer` takes an iterable of documents, or a callable returning one; a document
is a `str` or a `(str, language)` pair. It returns a `tokenizers.Tokenizer`. Encoding is
lossless for text without special tokens, unless `normalize_hamza=True`. With
`segmentation="shortest"` the same vocabulary encodes text in the fewest tokens instead of
in merge order.

The language tag selects suffix-aware weighting for Turkish (`tr`) and Finnish (`fi`); the
bigram gate goes by script. The streaming loader reads JSON-L with a `language` field:

```python
from supratok import StreamingDataLoader, train_tokenizer, to_sentencepiece

loader = StreamingDataLoader(use_entropy_filtering=True)
corpus = lambda: loader.stream_documents("culturax.jsonl", with_language=True)
tokenizer = train_tokenizer(corpus, vocab_size=100000)
to_sentencepiece(tokenizer, "supratok.model")
```

Enable entropy curation in the loader or in the trainer, not both. The pieces separately:

```python
from supratok import SupraTokTrainer, build_bpe_tokenizer, script_aware

trainer = SupraTokTrainer(vocab_size=100000, special_tokens=["<pad>"])
vocab, merges = trainer.train(documents)
normalizer = script_aware.hamza_normalizer() if trainer.normalize_hamza else None
tokenizer = build_bpe_tokenizer(vocab, merges, trainer.special_tokens, normalizer=normalizer)
```

The package also exports `build_shortest_tokenizer`, `EntropyFilter`, `ThreeBinSampler`,
the `appendix_a` scoring functions and the `script_aware` helpers.

## Parameters

| argument                                                | default                   | paper                                                                                                                                                                                                                                |
| ------------------------------------------------------- | ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `vocab_size`                                            | 100000                    | Table 4 uses 100k; Appendix B searched 160k–320k. Must leave room beyond the special tokens, the 256 byte tokens and 20 characters                                                                                                   |
| `special_tokens`                                        | none                      | counted in `vocab_size`; the corpus is split at them, as the input is, so no token or phrase spans or spells one; a special token cannot be the space, one of the 19 characters that spell byte tokens, or spelled like a byte token |
| `min_frequency`                                         | 5                         | minimum count for a merge or an alphabet character; a character must also be at least one in 5 million training characters. This and the other frequency thresholds are non-negative ints                                            |
| `seed_pmi_threshold`                                    | 2.0                       | §3.3 seed mining                                                                                                                                                                                                                     |
| `seed_min_frequency`                                    | 100                       | §3.3 seed mining                                                                                                                                                                                                                     |
| `max_seed_ngram`                                        | 6                         | §3.3, n ∈ {2..6}; 2 to 64                                                                                                                                                                                                            |
| `seed_overlap`                                          | 0.5                       | §3.3 overlap control                                                                                                                                                                                                                |
| `screening_pmi`                                         | 5.0                       | §3.5 later-pass screening threshold                                                                                                                                                                                                  |
| `screening_frequency`                                   | 5                         | §3.5 later-pass frequency threshold                                                                                                                                                                                                  |
| `max_span`                                              | 5                         | Appendix A, spans of 2–5 words; 2 to 64                                                                                                                                                                                              |
| `holdout_fraction`                                      | 0.1                       | Appendix A held-out shard, and an equal verification shard, drawn by a hash of each document's text; EXPRESSION needs them                                                                                                           |
| `use_entropy_filtering`                                 | False                     | §3.2, a separate pipeline stage                                                                                                                                                                                                      |
| `space_separated_stream`                                | False                     | rewrite the training text with one space between pre-tokens and no line breaks (`York, don't` becomes `York , don 't`, where `'t` is then two pre-tokens), for text in that pre-tokenized format, such as WikiText-103               |
| `script_aware`                                          | True                      | §3.4 bigram gate and suffix-aware weighting                                                                                                                                                                                          |
| `suffix_weight`                                         | 1                         | §3.4 suffix-aware weighting; a non-negative int                                                                                                                                                                                      |
| `normalize_hamza`                                       | False                     | §3.4.1 Arabic hamza normalization (lossy)                                                                                                                                                                                            |
| `stopwords`                                             | 57 English function words | Appendix A stopword ratio                                                                                                                                                                                                            |
| `enable_seeds`, `enable_multiword`, `enable_expression` | True                      | ablation switches                                                                                                                                                                                                                    |
| `segmentation` (`train_tokenizer`)                      | `"merges"`                | `"shortest"` encodes in the fewest tokens of the same vocabulary; not used for the paper's numbers                                                                                                                                   |

Paper-stated values are the defaults; hamza normalization, which is lossy, is opt-in. The
Appendix A constants are in [supratok/appendix_a.py](supratok/appendix_a.py).

## Benchmarks

The scripts in `benchmarks/` train SupraTok, BPE-Standard and SuperBPE and report characters
per token. Data recipes are in [datasets/README.md](datasets/README.md); the multilingual
tables use Wikipedia in place of the gated CulturaX.

```bash
python benchmarks/figure1_benchmark.py --docs 100000 --vocab 100000         # Tables 4A and 4B
python benchmarks/figure1_benchmark.py --docs 100000 --vocab 50000 --quick  # Table 5
python benchmarks/figure1_benchmark.py --docs 10000 --vocab 64000 --quick   # Table 7
python benchmarks/ablation_benchmark.py --docs 50000 --vocab 100000         # Table 3
python benchmarks/multilingual_benchmark.py --table 1                       # Table 1
python benchmarks/multilingual_benchmark.py --table 6                       # Table 6
```

Training on 100,000 documents needs about 13 GB of RAM.

## Tests

```bash
pip install -e ".[dev]" && pytest
```

The tests run in seconds on synthetic corpora; no datasets are needed.

## Layout

```
supratok/
  trainer.py        Figure 1 training procedure
  chains.py         word chains for the MULTIWORD pass
  appendix_a.py     Appendix A span scoring, Eqs. 5-14
  data_loader.py    Section 3.2 entropy-based curation and the streaming loader
  script_aware.py   Section 3.4 mechanisms
  arabic.py         hamza normalization
  hf_build.py       validated HuggingFace model construction, merge-order and shortest-path
  export.py         SentencePiece export
benchmarks/
  figure1_benchmark.py       Tables 4A, 4B, 5 and 7; data in datasets/README.md
  ablation_benchmark.py      Table 3
  multilingual_benchmark.py  Tables 1 and 6
  evaluation.py              evaluation sets (raw and detokenized WikiText-103, held-out Nemotron)
  superbpe.py                the SuperBPE baseline
tests/
  test_architecture.py   tokenizer tests, no datasets needed
  test_chains.py         word-chain extraction
  test_evaluation.py     the WikiText detokenizer
  test_benchmarks.py     the ablation script's rows, with the trainers stubbed
docs/
  implementation.md      implementation details
  tacl.a.774.pdf
```

## Citation

See [CITATION.cff](CITATION.cff). Licensed under MIT — see [LICENSE](LICENSE).