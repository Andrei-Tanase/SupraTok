"""The benchmark's WikiText detokenizer."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "benchmarks"))

from evaluation import detokenize_wikitext


def test_detokenizer_rejoins_wikitext_spacing():
    raw = " = Valkyria = \n York , don 't go ( 1 @,@ 000 ) ; it 's 2 @.@ 5 @-@ fold .\n"
    assert detokenize_wikitext(raw) == " = Valkyria = \n York, don't go (1,000); it's 2.5-fold.\n"


def test_detokenizer_leaves_quotes_spaced():
    assert detokenize_wikitext(' he said " yes " .') == ' he said " yes ".'


def test_detokenizer_rejoins_negations():
    assert detokenize_wikitext(" I do n't know , he ca n't .") == " I don't know, he can't."
