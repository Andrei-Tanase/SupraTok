"""SupraTok: cross-boundary tokenization.

Reference code for the tokenizer in "SupraTok: Cross-Boundary Tokenization for
Enhanced Language Model Performance" (TACL vol. 14, pp. 1787-1802, 2026).

    from supratok import train_tokenizer

    def documents():
        with open("corpus.txt", encoding="utf-8") as handle:
            yield from handle

    tokenizer = train_tokenizer(documents, vocab_size=100000)
    tokenizer.save("tokenizer.json")
"""

__version__ = "1.0.0"

from . import appendix_a, script_aware
from .data_loader import EntropyFilter, StreamingDataLoader, ThreeBinSampler
from .export import to_sentencepiece
from .hf_build import InvalidBpeSpec, build_bpe_tokenizer, build_shortest_tokenizer
from .trainer import SupraTokTrainer, train_tokenizer

__all__ = [
    "EntropyFilter",
    "InvalidBpeSpec",
    "StreamingDataLoader",
    "SupraTokTrainer",
    "ThreeBinSampler",
    "appendix_a",
    "build_bpe_tokenizer",
    "build_shortest_tokenizer",
    "script_aware",
    "to_sentencepiece",
    "train_tokenizer",
]
