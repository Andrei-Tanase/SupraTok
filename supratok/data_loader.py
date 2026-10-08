"""Section 3.2 entropy-based data curation and the streaming JSON-L loader of Section 3.6."""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterator

import numpy as np

logger = logging.getLogger(__name__)

_SURROGATE = re.compile("[\ud800-\udfff]")


class EntropyFilter:
    """Character-bigram document entropy (paper Eq. 1).

    window_size=None scores the whole document, as Eq. 1 specifies; an integer scores
    a window of that many characters from the middle of the document instead.
    """

    def __init__(self,
                 window_size: int | None = None,
                 low_threshold: float = 3.0,
                 high_threshold: float = 4.5):
        self.window_size = window_size
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def calculate_entropy(self, text: str) -> float:
        if len(text) < 2:
            return 0.0

        if self.window_size and len(text) > self.window_size:
            start_idx = len(text) // 2 - self.window_size // 2
            text = text[start_idx:start_idx + self.window_size]

        bigram_counts: Counter[str] = Counter()
        for i in range(len(text) - 1):
            bigram_counts[text[i:i + 2]] += 1

        total_bigrams = sum(bigram_counts.values())
        if total_bigrams == 0:
            return 0.0

        entropy = 0.0
        for count in bigram_counts.values():
            probability = count / total_bigrams
            entropy -= probability * math.log2(probability)

        return entropy

    def _bin_for_entropy(self, entropy: float) -> str:
        if entropy < self.low_threshold:
            return "low"
        if entropy <= self.high_threshold:
            return "medium"
        return "high"

    def bin_of(self, text: str) -> str:
        return self._bin_for_entropy(self.calculate_entropy(text))


class ThreeBinSampler:
    """Section 3.2 retention: keep 10% of low-, 50% of medium-, 90% of high-entropy
    documents, sampled with a seeded generator so a run is reproducible."""

    def __init__(self,
                 low_rate: float = 0.1,
                 medium_rate: float = 0.5,
                 high_rate: float = 0.9,
                 low_threshold: float = 3.0,
                 high_threshold: float = 4.5,
                 seed: int = 0):
        self.rates = {"low": low_rate, "medium": medium_rate, "high": high_rate}
        self.entropy_filter = EntropyFilter(window_size=None,
                                            low_threshold=low_threshold,
                                            high_threshold=high_threshold)
        self._rng = np.random.default_rng(seed)

        self.bin_counts: defaultdict[str, int] = defaultdict(int)
        self.total_processed = 0
        self.total_sampled = 0

    def should_sample(self, text: str) -> bool:
        bucket = self.entropy_filter.bin_of(text)
        self.bin_counts[bucket] += 1
        self.total_processed += 1

        keep = bool(self._rng.random() < self.rates[bucket])
        if keep:
            self.total_sampled += 1
        return keep

    def get_statistics(self) -> dict[str, object]:
        return {
            "total_processed": self.total_processed,
            "total_sampled": self.total_sampled,
            "sample_rate": self.total_sampled / max(self.total_processed, 1),
            "bin_distribution": dict(self.bin_counts),
        }


class StreamingDataLoader:
    """Streaming JSON-L reader with language and entropy filtering.

    A document is kept when its stripped text is non-empty and at least
    `min_text_length` characters long, the loader rule of Table E1. Lines that are not
    UTF-8 JSON objects with a `text` string are skipped and counted as invalid.
    """

    def __init__(self,
                 language_filters: list[str] | None = None,
                 use_entropy_filtering: bool = False,
                 min_text_length: int = 50,
                 entropy_seed: int = 0):
        self.language_filters = ({lang.lower() for lang in language_filters}
                                 if language_filters else None)
        self.use_entropy_filtering = use_entropy_filtering
        self.min_text_length = min_text_length
        self.entropy_seed = entropy_seed
        self.sampler = ThreeBinSampler(seed=entropy_seed) if use_entropy_filtering else None
        self.stats: dict[str, int] = {
            "total_lines": 0,
            "invalid_lines": 0,
            "valid_documents": 0,
            "language_filtered": 0,
            "length_filtered": 0,
            "entropy_filtered": 0,
            "final_sampled": 0,
        }

    def stream_documents(self,
                         file_path: str,
                         max_documents: int | None = None,
                         with_language: bool = False) -> Iterator[str | tuple[str, str]]:
        """Yield the documents of a JSON-L file with `text` and optional `language` fields.

        `max_documents` limits the input lines read. With `with_language`, documents are
        `(text, language)` pairs, the form the trainer uses for language-specific
        processing. Every call samples and counts afresh, so consecutive passes are
        identical.
        """
        logger.info("Streaming documents from %s", file_path)
        self.stats = dict.fromkeys(self.stats, 0)
        sampler = self.sampler = (ThreeBinSampler(seed=self.entropy_seed)
                                  if self.use_entropy_filtering else None)
        with open(file_path, "rb") as f:
            for line_num, line in enumerate(f, 1):
                if max_documents is not None and line_num > max_documents:
                    break
                if not line.strip():
                    continue
                self.stats["total_lines"] += 1
                try:
                    data = json.loads(line.decode("utf-8-sig"))
                except (ValueError, RecursionError) as e:  # not UTF-8, not JSON, or nested too deep
                    logger.warning("Skipping line %d: %s", line_num, e)
                    self.stats["invalid_lines"] += 1
                    continue
                text = data.get("text") if isinstance(data, dict) else None
                if not isinstance(text, str) or _SURROGATE.search(text):
                    self.stats["invalid_lines"] += 1
                    continue
                text = text.strip()
                language = data.get("language")
                if not isinstance(language, str) or not language:
                    language = "unknown"
                if self.language_filters and language.lower() not in self.language_filters:
                    self.stats["language_filtered"] += 1
                    continue
                if not text or len(text) < self.min_text_length:
                    self.stats["length_filtered"] += 1
                    continue
                self.stats["valid_documents"] += 1
                if sampler is not None and not sampler.should_sample(text):
                    self.stats["entropy_filtered"] += 1
                    continue
                self.stats["final_sampled"] += 1
                yield (text, language) if with_language else text
        self._log_final_statistics()

    def _log_final_statistics(self):
        logger.info("Data loading completed:")
        for key, value in self.stats.items():
            logger.info("  %s: %s", key, format(value, ","))
        if self.sampler:
            sampling = self.sampler.get_statistics()
            logger.info("  Entropy sampling rate: %.2f%%", 100 * sampling["sample_rate"])
            logger.info("  Entropy bin distribution: %s", sampling["bin_distribution"])

    def get_statistics(self) -> dict[str, object]:
        stats: dict[str, object] = dict(self.stats)
        if self.sampler:
            stats["entropy_sampling"] = self.sampler.get_statistics()
        return stats
