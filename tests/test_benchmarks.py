"""The ablation benchmark's control flow, with every tokenizer trainer stubbed."""

import os
import sys
from collections import Counter
from types import SimpleNamespace

BENCHMARKS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "benchmarks")
sys.path.insert(0, BENCHMARKS)

import ablation_benchmark
import superbpe

import supratok


class _Tokenizer:
    def encode_batch(self, texts, add_special_tokens=False):
        return [SimpleNamespace(ids=list(text)) for text in texts]


def test_ablation_trains_each_row_once(monkeypatch, capsys):
    calls = Counter()

    def stub(name):
        def train(*args, **kwargs):
            calls[name, kwargs.get("transition")] += 1
            return _Tokenizer()
        return train

    monkeypatch.setattr(ablation_benchmark, "TRAIN", __file__)
    monkeypatch.setattr(ablation_benchmark, "EVAL", __file__)
    monkeypatch.setattr(ablation_benchmark, "nemotron_documents",
                        lambda start, stop: ["some text here"] * 10)
    monkeypatch.setattr(ablation_benchmark, "evaluation_sets",
                        lambda n: {"wiki raw": ["a b"], "wiki detok": ["a b"], "held-out": ["c"]})
    monkeypatch.setattr(ablation_benchmark, "bpe_standard", stub("bpe"))
    monkeypatch.setattr(supratok, "train_tokenizer", stub("supratok"))
    monkeypatch.setattr(superbpe, "train_superbpe", stub("superbpe"))
    monkeypatch.setattr(sys, "argv", ["ablation_benchmark.py", "--docs", "10", "--vocab", "1000"])
    ablation_benchmark.main()
    # Five SupraTok configurations, BPE-Standard, and SuperBPE with and without its
    # curriculum, so the curriculum is also measured inside one pipeline.
    assert calls == {("supratok", None): 5, ("bpe", None): 1, ("superbpe", 0.0): 1,
                     ("superbpe", None): 1}
    assert "SuperBPE" in capsys.readouterr().out
