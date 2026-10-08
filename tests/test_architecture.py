"""Architecture tests: SupraTokTrainer against the paper it implements.

Every corpus is built in-process; no dataset file is read and no vocabulary above 3,000
is trained.
"""

import json
import math
import random
import unicodedata
from collections import Counter

import numpy as np
import pytest
import regex
from tokenizers import AddedToken, Tokenizer, models, trainers

from supratok import appendix_a as A
from supratok import script_aware as S
from supratok import to_sentencepiece
from supratok import trainer as trainer_module
from supratok.data_loader import EntropyFilter, StreamingDataLoader, ThreeBinSampler
from supratok.hf_build import (
    BYTE_TOKENS,
    InvalidBpeSpec,
    build_bpe_tokenizer,
    build_shortest_tokenizer,
)
from supratok.trainer import (
    _CHUNK,
    _CROSSES,
    _WORD,
    SupraTokTrainer,
    _bounds,
    _bpe,
    _derive,
    _extract,
    _plan,
    _sort,
    _Span,
    _take,
    _train_bpe,
    shard_of,
    train_tokenizer,
)

VOCAB_SIZE = 1000

ROUNDTRIP_PROBES = [
    "the government announced new findings",
    "Café résumé naïve Zürich",
    "北京大学 ภาษาไทย 한국어",
    "\U0001F30D\U0001F600 emoji tail",
    "tab\there\nand a newline",
    "two  spaces   and    four",
    "",
]

_CONSONANTS = "bdfgklmnprstvz"
_VOWELS = "aeiou"


def _filler_words(n, seed):
    """Distinct pseudo-words, so BPE has enough merge material to fill the vocab."""
    rng = random.Random(seed)
    out = []
    seen = set()
    while len(out) < n:
        w = "".join(rng.choice(_CONSONANTS) + rng.choice(_VOWELS)
                    for _ in range(rng.randint(2, 4)))
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def n_words(token):
    """Words a token spans: an internal space is a crossed word boundary."""
    return token[1:].count(" ") + 1


def _phrases_and_punctuated_words():
    """6,000 documents of planted four-word phrases, then filler words each followed by a
    comma; two spaces after each comma, so no seed or screened bigram starts at it."""
    rng = random.Random(15)
    phrases = [" ".join(_filler_words(4, 100 + i)) for i in range(120)]
    filler = _filler_words(100, 16)
    return [" ".join(rng.choice(phrases) for _ in range(4)) + " "
            + "  ".join(rng.choice(filler) + "," for _ in range(16)) for _ in range(6000)]


@pytest.fixture(scope="module")
def source():
    """Planted collocations ("New York") and 3-5 word expressions at high frequency,
    padded with pseudo-word filler. Single spaces only, so an internal space can come
    from nothing but a genuine cross-boundary merge."""
    rng = random.Random(0)
    filler = _filler_words(240, 1)
    subjects = ["the government", "the company", "New York", "the United Nations",
                "our research team", "the central bank"]
    verbs = ["announced", "reported", "confirmed", "published", "reviewed"]
    objects = ["climate change initiatives", "the annual report", "new findings",
               "quantum entanglement research", "several policy documents"]
    tails = ["in order to improve outcomes.", "as well as related measures.",
             "due to rising demand.", "after a lengthy consultation.",
             "with support from local partners."]
    docs = []
    for _ in range(300):
        sentence = (f"{rng.choice(subjects)} {rng.choice(verbs)} "
                    f"{rng.choice(objects)} {rng.choice(tails)}")
        pad = " ".join(rng.choice(filler) for _ in range(12))
        docs.append(" ".join([sentence] * rng.randint(2, 4)) + " " + pad)
    return lambda: iter(docs)


def _train(source, **kwargs):
    trainer = SupraTokTrainer(vocab_size=VOCAB_SIZE, min_frequency=2, **kwargs)
    vocab, merges = trainer.train(source)
    return trainer, vocab, merges


@pytest.fixture(scope="module")
def traditional(source):
    return _train(source, enable_multiword=False, enable_expression=False, enable_seeds=False)


@pytest.fixture(scope="module")
def multiword(source):
    return _train(source, enable_multiword=True, enable_expression=False, enable_seeds=False)


@pytest.fixture(scope="module")
def seeded(source):
    return _train(source, enable_multiword=True, enable_expression=False)


@pytest.fixture(scope="module")
def expression(source):
    return _train(source, enable_multiword=True, enable_expression=True)


# ------------------------------------------------------- 1. well-formedness

def test_training_output_builds_a_tokenizer(expression):
    _, vocab, merges = expression
    tokenizer = build_bpe_tokenizer(vocab, merges, ["<pad>"])
    assert tokenizer.get_vocab_size() == VOCAB_SIZE + 1


@pytest.mark.parametrize("probe", ROUNDTRIP_PROBES)
def test_roundtrip_is_exact(expression, probe):
    """Byte-level coverage: unseen scripts, emoji and whitespace runs must survive."""
    _, vocab, merges = expression
    tokenizer = build_bpe_tokenizer(vocab, merges, [])
    assert tokenizer.decode(tokenizer.encode(probe).ids) == probe


@pytest.mark.parametrize("fixture", ["traditional", "multiword", "expression"])
def test_vocab_hits_requested_size_exactly(fixture, request):
    _, vocab, _ = request.getfixturevalue(fixture)
    assert len(vocab) == VOCAB_SIZE


def test_base_vocabulary_is_characters_plus_byte_fallback(expression):
    """Section 4.1: Unicode characters plus the 256 byte-fallback symbols."""
    trainer, vocab, _ = expression
    assert all(t in vocab for t in BYTE_TOKENS)
    assert trainer.alphabet
    assert {t for t in vocab if len(t) == 1} == set(trainer.alphabet)


def test_rare_and_unseen_characters_use_byte_fallback():
    docs = ["alpha beta gamma"] * 50 + ["zeta Ω"] * 2
    tokenizer = train_tokenizer(lambda: iter(docs), vocab_size=300, min_frequency=5,
                                enable_seeds=False, enable_multiword=False,
                                enable_expression=False)
    assert "Ω" not in tokenizer.get_vocab()
    assert tokenizer.encode("Ω").tokens == ["<0xCE>", "<0xA9>"]
    assert tokenizer.decode(tokenizer.encode("alpha Ω☃").ids) == "alpha Ω☃"


def test_rare_characters_cut_training_items():
    """A character below the floor goes through byte fallback, so the characters on
    either side of it never become a merge."""
    docs = ["alpha beta gamma"] * 50 + [f"q{chr(0xAC00 + i)}z" for i in range(200)]
    tokenizer = train_tokenizer(docs, vocab_size=320, min_frequency=5, enable_seeds=False,
                                enable_multiword=False, enable_expression=False)
    assert "q" in tokenizer.get_vocab() and "qz" not in tokenizer.get_vocab()


@pytest.mark.parametrize("text,chunks", [
    ("New York, 2024!", ["New", " York", ",", " 2024", "!"]),
    ("a  b", ["a", " ", " b"]),
    ("end.\n\nNext", ["end", ".\n\n", "Next"]),
    ("end \n\nNext", ["end", " \n\n", "Next"]),
    ("don't", ["don", "'t"]),
    ("Ankara'da", ["Ankara'da"]),
    ("l'homme d\u2019un", ["l'homme", " d\u2019un"]),
    ("O'Brien's dell'anno", ["O'Brien", "'s", " dell'anno"]),
    ("rock'n'roll 'quoted'", ["rock'n'roll", " '", "quoted", "'"]),
    ("\u0441\u043b\u043e\u0432\u043e\u00a0\u2014 \u0442.\u00a0\u0435.",
     ["\u0441\u043b\u043e\u0432\u043e", "\u00a0\u2014", " \u0442", ".", "\u00a0\u0435", "."]),
    ("x\u200by", ["x", "\u200b", "y"]),
])
def test_pretokens_follow_the_supercategories(text, chunks):
    """Section 3.4: LM, N, PS and C runs with their leading space, and whitespace runs; a
    PS run keeps the line breaks that follow it."""
    assert _CHUNK.findall(text) == chunks


def test_punctuation_before_a_line_break_does_not_cross_a_boundary():
    """A pre-token such as ".\n" holds no space after text, so it joins no words."""
    assert not _CROSSES.search(".\n\n")
    assert _CROSSES.search("New York") and _CROSSES.search("end. Next")


def test_special_tokens_fill_the_vocabulary_exactly(source):
    """Special tokens are cut out of the corpus as they are cut out of the input, so one
    that also occurs in the text is not learned, and a repeated one counts once."""
    docs = [doc + " ###" for doc in source()]
    trainer = SupraTokTrainer(vocab_size=VOCAB_SIZE, min_frequency=2,
                              special_tokens=["<pad>", "###", "<pad>"])
    vocab, merges = trainer.train(docs)
    assert trainer.special_tokens == ["<pad>", "###"]
    assert "##" not in vocab and "###" not in vocab
    tokenizer = build_bpe_tokenizer(vocab, merges, trainer.special_tokens)
    assert tokenizer.get_vocab_size() == VOCAB_SIZE
    assert tokenizer.token_to_id("###") in tokenizer.encode("a ### b").ids


@pytest.mark.parametrize("spaced", [False, True])
def test_special_tokens_split_the_training_text(source, spaced):
    """No token or phrase spans a special token, and the text around one gains nothing."""
    docs = [doc + " zorbix<sep> quandle\n<sep>\n" for doc in source()]
    trainer = SupraTokTrainer(vocab_size=VOCAB_SIZE, min_frequency=2, special_tokens=["<sep>"],
                              space_separated_stream=spaced)
    vocab, merges = trainer.train(docs)
    assert not [t for t in vocab if "zorbix" in t and "quandle" in t]
    assert "\n\n" not in vocab
    assert build_bpe_tokenizer(vocab, merges, ["<sep>"]).get_vocab_size() == VOCAB_SIZE


def test_respacing_cannot_spell_a_special_token():
    """Respacing turns "York, and" into "York , and"; the special token it spells is cut."""
    filler = _filler_words(2000, 3)
    docs = ["York, and " + " ".join(filler[i * 40:(i + 1) * 40]) for i in range(50)]
    trainer = SupraTokTrainer(vocab_size=600, min_frequency=2, special_tokens=["York ,"],
                              space_separated_stream=True, enable_expression=False)
    vocab, merges = trainer.train(docs)
    assert "York ," not in vocab
    assert build_bpe_tokenizer(vocab, merges, ["York ,"]).get_vocab_size() == len(vocab) + 1


def test_documents_may_be_a_list_an_iterator_or_a_callable(source):
    docs = list(source())[:60]
    options = {"vocab_size": 400, "min_frequency": 2, "enable_expression": False}
    expected = SupraTokTrainer(**options).train(docs)
    assert SupraTokTrainer(**options).train(iter(docs)) == expected
    assert SupraTokTrainer(**options).train(lambda: iter(docs)) == expected
    with pytest.raises(TypeError):
        SupraTokTrainer(**options).train("corpus.txt")


@pytest.mark.parametrize("options,error", [
    ({"vocab_size": 4e3}, TypeError),
    ({"special_tokens": [""]}, ValueError),
    ({"seed_overlap": 0.0}, ValueError),
    ({"seed_overlap": 1.5}, ValueError),
    ({"special_tokens": "<pad>"}, TypeError),
    ({"stopwords": "the"}, TypeError),
    ({"special_tokens": [" "]}, ValueError),
    ({"special_tokens": ["<0x+A>"]}, ValueError),
    ({"special_tokens": ["x"]}, ValueError),
    ({"special_tokens": ["0"]}, ValueError),
    ({"vocab_size": 256}, ValueError),
])
def test_invalid_arguments_fail_before_training(options, error):
    with pytest.raises(error):
        SupraTokTrainer(**options)


def test_vocab_size_must_leave_room_for_merges():
    trainer = SupraTokTrainer(vocab_size=300, min_frequency=1)
    with pytest.raises(ValueError):
        trainer.train(["".join(map(chr, range(0x4E00, 0x4E80)))] * 3)
    # A failed run releases the corpus it read, as a finished one does.
    assert not len(trainer.ids) and not trainer.types and not trainer.heldout


def test_entropy_filtering_curates_the_training_corpus(source):
    docs = list(source()) + ["ab " * 40] * 300
    trainer = SupraTokTrainer(vocab_size=400, min_frequency=2, use_entropy_filtering=True,
                              enable_expression=False)
    trainer.train(docs)
    assert 200 < trainer.stats["documents"] < 400


def test_joined_feeding_trains_like_feeding_every_copy():
    """BPE runs feed each text's copies joined into a few strings; the trainer must see
    exactly the words it would if every copy were fed on its own."""
    items = [("abab", 7), ("ab", 3), ("bca", 5), (" ab", 2), ("xy", 9000)]

    def trainer():
        return trainers.BpeTrainer(vocab_size=40, min_frequency=1, show_progress=False)

    one_by_one = Tokenizer(models.BPE())
    one_by_one.train_from_iterator((t for t, c in items for _ in range(c)), trainer=trainer())
    assert _train_bpe(items, trainer(), "\x00") == _extract(one_by_one)[1]
    # Counts are exact across batches: "xy" merges only if all 4,097 copies arrive.
    strict = trainers.BpeTrainer(vocab_size=40, min_frequency=4097, show_progress=False)
    assert _train_bpe([("xy", 4097)], strict, "\x00") == [("x", "y")]


@pytest.mark.parametrize("high,size", [(10, 5000), (2 ** 40, 5000), (2 ** 62, 140000)])
def test_key_sort_matches_numpy(high, size):
    """The radix path and, for keys too wide for it, the plain sort."""
    rng = np.random.default_rng(0)
    for n in (0, 1, 7, size):
        keys = rng.integers(0, high, n, dtype=np.int64)
        keys = np.concatenate([keys, keys[:n // 2]])
        out, order = _sort(keys)
        assert np.array_equal(out, np.sort(keys)) and np.array_equal(keys[order], out)
        assert np.array_equal(np.sort(order), np.arange(len(keys)))


def test_unit_selection_is_the_left_to_right_greedy():
    rng = random.Random(0)
    for _ in range(500):
        lengths = [rng.choice([0, 0, 0, 2, 3, 4, 5, 6]) for _ in range(rng.randint(0, 80))]
        span_n = np.array(lengths + [0] * 6, np.int8)
        taken, end = [], 0
        for p, n in enumerate(lengths):
            if n and p >= end:
                taken.append(p)
                end = p + n
        assert _take(span_n).tolist() == taken


# ------------------------------------------------- 2. three ordered passes

def test_traditional_pass_produces_no_cross_boundary_token(traditional):
    """Pass 1 merges are bounded inside a word; nothing may cross a space."""
    trainer, vocab, _ = traditional
    offenders = [t for t in vocab if n_words(t) > 1]
    assert offenders == []
    assert trainer.stats["cross_boundary"] == 0


def test_multiword_pass_creates_cross_boundary_tokens(multiword):
    """Section 3.5: a PMI-screened word bigram becomes one mergeable unit."""
    trainer, vocab, _ = multiword
    assert trainer.stats["pass2_candidates"] > 0
    assert [t for t in vocab if n_words(t) == 2]


def test_screened_junctions_chain_beyond_two_words():
    """Section 3.5 screens later-pass merges, so screened junctions chain into one token."""
    rng = random.Random(8)
    filler = _filler_words(400, 9)
    docs = [" ".join(rng.choice(filler) for _ in range(20)) + " zylphon quarnex vexilor "
            + " ".join(rng.choice(filler) for _ in range(20)) for _ in range(200)]
    tokenizer = train_tokenizer(docs, vocab_size=900, min_frequency=2, enable_seeds=False,
                                enable_expression=False)
    assert " zylphon quarnex vexilor" in tokenizer.get_vocab()


def test_loose_frequent_pairs_are_not_joined_at_the_paper_screen():
    """' of the' joins two words seen everywhere: frequent, but its PMI is low."""
    rng = random.Random(4)
    filler = _filler_words(300, 7)
    docs = []
    for _ in range(300):
        parts = [rng.choice(filler) for _ in range(12)]
        parts[3:3] = ["of", "the"]
        parts[8:8] = ["the"]
        parts[10:10] = ["of"]
        docs.append(" ".join(parts))
    tokenizer = train_tokenizer(docs, vocab_size=700, min_frequency=2, enable_seeds=False,
                                enable_expression=False)
    assert " of the" not in tokenizer.get_vocab()


def test_seed_occurrences_pass_the_junction_screen():
    """Section 3.3's PMI-seeded MULTIWORD candidates: junctions inside a seed pass even when
    their bigram falls below the later-pass screen."""
    rng = random.Random(5)
    filler = _filler_words(40, 6)
    docs = [f"{rng.choice(filler)} in order to {rng.choice(filler)}" for _ in range(60)]
    trainer = SupraTokTrainer(seed_min_frequency=10, max_seed_ngram=3, screening_pmi=50.0,
                              enable_expression=False)
    trainer._read(lambda: iter(docs))
    mined = trainer._mine()
    assert (" in", " order", " to") in mined.seeds
    words = [trainer.types[t] for t in trainer.ids.tolist()]
    inside = [i for i in range(1, len(words))
              if (words[i - 1], words[i]) in ((" in", " order"), (" order", " to"))]
    assert inside and all(mined.junctions[i] for i in inside)
    assert not mined.junctions[[i for i, w in enumerate(words) if w == " in"]].any()


def test_attached_junctions_pass_without_the_screen():
    """A word and what follows it without a space form one whitespace-delimited word
    (Appendix A), so that junction passes even when the Section 3.5 screen is closed."""
    trainer = SupraTokTrainer(screening_pmi=50.0, enable_seeds=False, enable_expression=False)
    trainer._read(lambda: iter(["alpha beta, gamma's delta"] * 20))
    mined = trainer._mine()
    words = [trainer.types[t] for t in trainer.ids.tolist()]
    attached = [i for i, w in enumerate(words) if w in (",", "'s")]
    spaced = [i for i, w in enumerate(words) if w in (" beta", " gamma", " delta")]
    assert attached and all(mined.junctions[i] for i in attached)
    assert not any(mined.junctions[i] for i in spaced)


def test_joint_run_sees_whitespace_delimited_words_whole():
    """The competition that sizes the passes sees "beta," and "gamma's" as the words they
    are, so the merges joining their pieces compete there too."""
    trainer = SupraTokTrainer(enable_expression=False)
    trainer._read(lambda: iter(["alpha beta, gamma's delta"] * 20))
    items = dict(trainer._mine().plain)
    assert items[" beta,"] == 20 and items[" gamma's"] == 20
    assert " beta" not in items and "," not in items


def test_joint_run_items_stop_at_the_multiword_cap():
    """A long space-free run ("ab,cd,ef,...") enters the sizing run in windows of at most
    six words, the most MULTIWORD can join, so its cost does not grow with length."""
    trainer = SupraTokTrainer(enable_expression=False)
    trainer._read(lambda: iter([",".join(["ab", "cd"] * 400) + " end"] * 3))
    longest = max(len(_CHUNK.findall(text)) for text, _ in trainer._mine().plain)
    assert longest == 6


def _phrases(n=300, seed=0):
    """Documents repeating whole clauses, so seeds are mined and MULTIWORD extends them."""
    rng = random.Random(seed)
    filler = _filler_words(240, 1)
    parts = (["the government", "the company", "New York", "the United Nations"],
             ["announced", "reported", "confirmed", "published"],
             ["climate change initiatives", "the annual report", "new findings"],
             ["in order to improve outcomes.", "due to rising demand.",
              "with support from local partners."])
    docs = []
    for _ in range(n):
        clause = " ".join(rng.choice(p) for p in parts)
        docs.append(" ".join([clause] * rng.randint(2, 4)) + " "
                    + " ".join(rng.choice(filler) for _ in range(12)))
    return docs


def test_multiword_never_builds_a_token_twice():
    """MULTIWORD learns the seeds again over its chains and may reach a seed's text by
    another route; a merge forming a token T already holds is skipped, so no slot of T
    is spent twice."""
    trainer = SupraTokTrainer(vocab_size=600, min_frequency=2, enable_expression=False)
    _, merges = trainer.train(_phrases())
    made = Counter(a + b for a, b in merges)
    assert trainer.seeds and trainer.stats["pass2_merges"] > 0
    assert [t for t, n in made.items() if n > 1] == []


def test_multiword_trains_long_chains_in_windows(monkeypatch):
    """A space-free run of words ("ab,cd,ef,...") is one chain; MULTIWORD trains it in
    windows of MAX_CHAIN words, so its cost does not grow with the chain's length."""
    longest = []
    real = trainer_module._train_bpe

    def spied(items, trainer, sep):
        items = list(items)
        if sep == "\x00":
            longest.append(max(len(seq) for seq, _ in items))
        return real(items, trainer, sep)

    monkeypatch.setattr(trainer_module, "_train_bpe", spied)
    blob = ",".join(["ab", "cd", "e"] * 300)
    train_tokenizer([blob + " x"] * 5 + ["the cat sat on the mat"] * 50, vocab_size=600,
                    enable_expression=False)
    assert longest and max(longest) <= trainer_module.MAX_CHAIN


def test_seeds_leave_the_attached_share_to_multiword():
    """Seeds take only the cross-word share of T; the share that joins words to what is
    attached to them stays with MULTIWORD even when the seeds would fill the rest."""
    docs = _phrases_and_punctuated_words()
    filler = _filler_words(100, 16)
    trainer = SupraTokTrainer(vocab_size=1500, min_frequency=2, seed_min_frequency=60,
                              enable_expression=False)
    vocab, _ = trainer.train(docs)
    assert trainer.stats["attached_share"] > 0
    assert trainer.stats["pass2_merges"] > 0
    assert any(t.endswith(",") and t[:-1].strip() in filler for t in vocab)


def test_attached_share_counts_merges_across_pre_tokens():
    """The joint run's merges that join pre-tokens without a space form the attached share,
    fragments ("s," inside " cats,") included, since MULTIWORD joins pieces; no word here
    recurs whole with its punctuation, yet MULTIWORD has room and uses it."""
    rng = random.Random(0)
    phrases = ["the lazy dog", "new york city", "machine learning model", "united states"]
    docs = []
    for _ in range(3000):
        words = []
        for _ in range(30):
            if rng.random() < 0.5:
                words.append(rng.choice(phrases))
            else:
                word = "".join(rng.choice("bcdfghjklmnpqrstvwxz") + rng.choice("aeiou")
                               for _ in range(3))
                words.append(word + rng.choice(["s,", "s.", "e,", "ed.", "s;"]))
        docs.append(" ".join(words))
    trainer = SupraTokTrainer(vocab_size=3000, enable_expression=False, seed_min_frequency=50)
    vocab, _ = trainer.train(docs)
    assert trainer.stats["attached_share"] > 0 and trainer.stats["pass2_merges"] > 0
    assert any(t in vocab for t in ("s,", "s.", "ed."))


def test_attached_runs_stop_at_the_multiword_cap():
    """Unspaced text joined by punctuation ("あい、うえ、...") is one long attached run; it
    enters the sizing run in windows of six words, MULTIWORD's limit."""
    run = "、".join(["あい", "うえ"] * 150)
    trainer = SupraTokTrainer(script_aware=False, enable_expression=False)
    trainer._read(lambda: iter([run + " end"] * 3))
    plain = trainer._mine().plain
    assert max(len(_CHUNK.findall(text)) for text, _ in plain) == 6


def test_refill_rounds_keep_the_seeds_that_fit(monkeypatch):
    """Slots MULTIWORD leaves unused go back to TRADITIONAL, and the attached share shrinks
    by as many, so every seed that fitted before still fits."""
    monkeypatch.setattr(SupraTokTrainer, "_multiword",
                        lambda self, merges, budget, junctions: (merges, 0, 0))
    placed = []
    real = SupraTokTrainer._pass1

    def spied(self, *args, **kwargs):
        out = real(self, *args, **kwargs)
        placed.append(len(out[3]))
        return out

    monkeypatch.setattr(SupraTokTrainer, "_pass1", spied)
    docs = _phrases_and_punctuated_words()
    trainer = SupraTokTrainer(vocab_size=1500, min_frequency=2, seed_min_frequency=60,
                              enable_expression=False)
    trainer.train(docs)
    assert trainer.stats["attached_share"] > 0 and len(placed) > 1
    assert placed[0] > 0 and all(n >= placed[0] for n in placed)


def test_a_non_breaking_space_leads_a_word_like_a_space():
    """Russian and French text put U+00A0 before dashes and some punctuation; it leads the
    next pre-token as a space does, so `\u00a0\u2014` is learned as ` \u2014` would be."""
    rng = random.Random(6)
    filler = _filler_words(200, 6)
    docs = [" ".join(rng.choice(filler) for _ in range(8)) + "\u00a0\u2014 "
            + " ".join(rng.choice(filler) for _ in range(8)) for _ in range(300)]
    tokenizer = train_tokenizer(docs, vocab_size=500, min_frequency=2, enable_expression=False)
    assert "\u00a0\u2014" in tokenizer.get_vocab()
    assert tokenizer.decode(tokenizer.encode(docs[0]).ids) == docs[0]
    # The junction before it is one between words, not one a word shares with what is
    # attached to it.
    trainer = SupraTokTrainer(enable_expression=False, enable_seeds=False, enable_multiword=False)
    trainer._read(lambda: iter(["alpha\u00a0\u2014 beta"] * 3))
    junctions = trainer._mine().junctions
    at = [i for i in range(len(trainer.ids)) if trainer.types[trainer.ids[i]] == "\u00a0\u2014"]
    assert at and not junctions[at].any()


def test_a_word_spelled_with_a_byte_token_ends_a_chain():
    """A word holding a character T lacks, spelled through byte fallback, joins no chain,
    so nothing attached to it ("(" before " ca\u0101fe") merges with its pieces."""
    docs = [f"zeta (ca{chr(0x0100 + i % 300)}fe) eta" for i in range(900)]
    tokenizer = train_tokenizer(docs, vocab_size=400, min_frequency=5, enable_seeds=False,
                                enable_expression=False)
    vocab = tokenizer.get_vocab()
    assert chr(0x0100) not in vocab and "ca" in vocab
    assert "(ca" not in vocab and "fe)" not in vocab


def test_attached_punctuation_joins_its_word():
    rng = random.Random(13)
    filler = _filler_words(300, 14)
    docs = [" ".join(rng.choice(filler) for _ in range(10)) + " zylphon quarnex, "
            + " ".join(rng.choice(filler) for _ in range(10)) for _ in range(200)]
    tokenizer = train_tokenizer(docs, vocab_size=800, min_frequency=2, enable_expression=False)
    assert any(t.endswith("quarnex,") for t in tokenizer.get_vocab())


def test_multiword_fills_its_room_when_it_relearns_seeds(monkeypatch):
    """Seed merges are learned again over the chains and skipped; the chain trainer is
    asked for them on top of MULTIWORD's room, so it stops when its candidates run out,
    never at its vocabulary cap, and T is filled."""
    spare, calls = [], []
    real_train, real_multiword = trainer_module._train_bpe, SupraTokTrainer._multiword

    def spied_train(items, trainer, sep):
        items = list(items)
        learned = real_train(items, trainer, sep)
        symbols = len(set("".join(seq for seq, _ in items)))
        spare.append(trainer.vocab_size - symbols - len(learned))
        return learned

    def spied_multiword(self, merges, budget, junctions):
        room = budget - len(self._vocab(merges))
        out = real_multiword(self, merges, budget, junctions)
        filled = len(self._vocab(out[0])) - len(self._vocab(merges))
        calls.append((room, filled, spare[-1]))
        return out

    monkeypatch.setattr(trainer_module, "_train_bpe", spied_train)
    monkeypatch.setattr(SupraTokTrainer, "_multiword", spied_multiword)
    rng = random.Random(15)
    phrases = [" ".join(_filler_words(4, 100 + i)) for i in range(120)]
    filler = _filler_words(1500, 16)
    docs = [", ".join(rng.choice(phrases) for _ in range(4)) + ". "
            + " ".join(rng.choice(filler) + rng.choice(",.;:") for _ in range(16))
            for _ in range(6000)]
    trainer = SupraTokTrainer(vocab_size=3000, min_frequency=2, seed_min_frequency=60,
                              enable_expression=False)
    vocab, _ = trainer.train(docs)
    assert trainer.stats["seed_units"] > 50
    assert calls and all(filled == room or left > 0 for room, filled, left in calls)
    assert len(vocab) == 3000


def test_multiword_never_learns_a_merge_beyond_the_word_cap(monkeypatch):
    """The six-piece cap binds inside the Rust trainer, so every learned merge is a valid
    candidate and later counts come from states the tokenizer can reach."""
    learned = []
    real = trainer_module._train_bpe

    def spy(items, trainer, sep):
        merges = real(items, trainer, sep)
        if sep == "\x00":
            learned.extend(merges)
        return merges

    monkeypatch.setattr(trainer_module, "_train_bpe", spy)
    rng = random.Random(11)
    filler = _filler_words(400, 12)
    phrase = "zylphon quarnex vexilor torbund marlix quopel snedrik flavorn"
    docs = [" ".join(rng.choice(filler) for _ in range(15)) + f" {phrase} "
            + " ".join(rng.choice(filler) for _ in range(15)) for _ in range(200)]
    train_tokenizer(docs, vocab_size=900, min_frequency=2, enable_seeds=False,
                    enable_expression=False)
    assert learned and max(len(a) + len(b) for a, b in learned) <= 6


def test_multiword_tokens_stay_within_the_ngram_range(multiword):
    trainer, vocab, _ = multiword
    assert trainer.stats["pass2_chains"] > 0
    assert max(n_words(t) for t in vocab) <= max(trainer.max_seed_ngram, trainer.max_span)


def test_chains_stop_at_line_breaks_and_double_spaces():
    docs = ["zylphon quarnex\nvexilor torbund  marlix quopel"] * 80
    tokenizer = train_tokenizer(docs, vocab_size=330, min_frequency=2, enable_expression=False)
    assert not [t for t in tokenizer.get_vocab() if ("\n" in t or "  " in t) and t.strip()]


def test_corpus_without_chains_fills_through_traditional():
    docs = ["".join(_filler_words(40, seed)) for seed in range(60)]
    trainer = SupraTokTrainer(vocab_size=600, min_frequency=1, enable_expression=False)
    vocab, _ = trainer.train(docs)
    assert trainer.stats["pass2_merges"] == 0 and len(vocab) == 600


def test_multiword_joins_the_pieces_on_either_side_of_a_junction():
    """MULTIWORD merges at the level of pieces: a word T still splits lends its last piece
    to what follows it (`aing,` from ` novevuing,`), and every MULTIWORD merge crosses a
    junction."""
    rng = random.Random(4)
    syllables = [c + v for c in _CONSONANTS for v in _VOWELS]
    stems = sorted({"".join(rng.choice(syllables) for _ in range(3)) for _ in range(400)})
    filler = _filler_words(300, 5)
    docs = [" ".join(rng.choice(filler) for _ in range(6)) + "  "
            + "  ".join(stem + "ing," for stem in rng.sample(stems, 8)) for _ in range(800)]
    trainer = SupraTokTrainer(vocab_size=700, min_frequency=2, enable_seeds=False,
                              enable_expression=False)
    vocab, merges = trainer.train(docs)
    assert [t for t in vocab if t.endswith("ing,") and not t.startswith(" ")]
    later = merges[trainer.stats["pass1_merges"]:]
    assert later and all(len(_CHUNK.findall(a + b)) > 1 for a, b in later)


def test_built_tokenizer_emits_cross_boundary_tokens(expression):
    """The built tokenizer fires the cross-boundary merges, since it encodes the whole
    text without pre-tokenization."""
    _, vocab, merges = expression
    tokenizer = build_bpe_tokenizer(vocab, merges, [])
    emitted = tokenizer.encode(
        "the United Nations reported new findings "
        "with support from local partners.").tokens
    assert max(n_words(t) for t in emitted) > 2


def test_expression_pass_spans_more_than_two_words(expression, seeded):
    """Pass 3 accepts Appendix A scored 2-5 word spans; seeds reach six words."""
    trainer, vocab, _ = expression
    assert trainer.stats["pass3_accepted"] > 0
    assert [t for t in vocab if n_words(t) > 2 and t not in seeded[1]]
    assert max(n_words(t) for t in vocab) <= max(trainer.max_span, trainer.max_seed_ngram)


def test_passes_continue_from_the_traditional_vocabulary(traditional, expression):
    """Section 3.3: pass 1 is word-bounded BPE, and later passes extend it rather than
    restarting, so every cross-boundary merge ranks after the pass-1 merges that leaf
    surgery left in place."""
    trainer, _, merges = expression
    k = trainer.stats["pass1_merges"]
    kept = set(merges)
    prefix = [m for m in traditional[2][:k] if m in kept]
    assert 0 < len(prefix) < len(merges)
    assert len(prefix) >= k - trainer.stats["surgery_retired"]
    assert merges[:len(prefix)] == prefix
    assert not any(_CROSSES.search(a + b) for a, b in prefix)


def test_leaf_pool_holds_every_leaf_with_its_held_out_utility(traditional, source):
    """Eq. 13: U(m) = f_eval(m) + 0.35 df_eval(m) over the held-out shard, for every
    merge whose output no other merge consumes, cheapest first."""
    _, vocab, merges = traditional
    heldout = [doc for doc in source() if shard_of(doc, 0.1) == "held-out"]
    tokenizer = build_bpe_tokenizer(vocab, merges, [])
    pool = SupraTokTrainer._leaves(merges, heldout, tokenizer)
    per_doc = [Counter(tokenizer.encode(doc).tokens) for doc in heldout]
    inputs = {x for m in merges for x in m}
    leaves = [m for m in merges if m[0] + m[1] not in inputs]
    assert sorted(m for _, _, m in pool) == sorted(leaves)
    for u, _, (a, b) in pool:
        f = sum(c[a + b] for c in per_doc)
        df = sum(1 for c in per_doc if c[a + b])
        assert u == pytest.approx(f + 0.35 * df)
    assert [u for u, _, _ in pool] == sorted(u for u, _, _ in pool)
    assert any(u > 0 for u, _, _ in pool) and any(u == 0 for u, _, _ in pool)


def test_expression_pass_respects_leaf_surgery_net_utility(source, monkeypatch):
    """Eqs. 13-14: a span is accepted only if Net(s) > 0, so with every leaf priced out of
    reach none is."""
    monkeypatch.setattr(A, "leaf_utility", lambda count, df: 1e9)
    trainer, _, _ = _train(source)
    assert trainer.stats["pass3_candidates"] > 0
    assert trainer.stats["pass3_accepted"] == 0


def test_expression_reverts_a_chunk_that_raises_verification_tokens(source, monkeypatch):
    """Net(s) > 0 is necessary, not sufficient: a chunk that costs the verification shard
    more tokens is reverted, and the pass stops."""
    counts = iter([100, 120])
    monkeypatch.setattr(SupraTokTrainer, "_verification_tokens", lambda self, m: next(counts))
    trainer, _, _ = _train(source, enable_seeds=True, enable_expression=True)
    assert trainer.stats["pass3_accepted"] == 0 and trainer.stats["surgery_retired"] == 0
    assert trainer.stats["pass3_chunks_kept"] == 0


def test_expression_keeps_chunks_that_lower_verification_tokens(source, monkeypatch):
    counts = iter(range(10 ** 6, 0, -1))
    monkeypatch.setattr(SupraTokTrainer, "_verification_tokens", lambda self, m: next(counts))
    trainer, _, _ = _train(source, enable_seeds=True, enable_expression=True)
    assert trainer.stats["pass3_accepted"] > 0 and trainer.stats["pass3_chunks_kept"] >= 1


def test_expression_without_a_verification_document_is_reported(caplog):
    texts = [f"tiny corpus {i}" for i in range(400)]
    keep = [t for t in texts if shard_of(t, 0.1) != "verification"][:60]
    trainer = SupraTokTrainer(vocab_size=300, min_frequency=1)
    trainer.train(keep)
    assert trainer.stats["heldout_documents"] and not trainer.stats["verification_documents"]
    assert trainer.stats["pass3_accepted"] == 0
    assert "verification shard" in caplog.text


def test_leaf_surgery_scores_against_the_current_tokenizer():
    """Eqs. 13-14 use l_tok of the tokenizer as it stands: once ' a b' is accepted,
    ' a b c' spans two units, so its A(s) no longer covers the leaf it would displace."""
    trainer = SupraTokTrainer(vocab_size=400, min_frequency=1)
    trainer._set_alphabet([("abcxyz ", 1)])
    trainer.heldout = [" a b c" * 10, "zx" * 5]
    trainer.verification = trainer.heldout
    merges = [(" ", "a"), (" ", "b"), (" ", "c"), ("x", "y"), ("z", "x")]
    common = {"npmi": 0.9, "boundary_entropy": 3.0, "context_peak": 0.1, "avg_idf": 1.0,
              "internal_npmi": 0.9, "stopword_ratio": 0.0, "avg_word_len": 1.0}
    spans = [_Span((" a", " b"), 50, 10, 5, char_len=4, **common),
             _Span((" a", " b", " c"), 50, 3, 2, char_len=6, **common)]
    budget = len(trainer._vocab(merges))
    out, scored, accepted, retired, _ = trainer._expression(merges, spans, budget)
    assert (scored, accepted, retired) == (2, 1, 1)
    assert (" a", " b") in out and ("x", "y") not in out
    assert all(a + b != " a b c" for a, b in out)


def test_leaf_surgery_keeps_vocabulary_size_when_the_span_already_exists():
    """|T| stays at V when a span's token already exists on another merge path: that
    token is part of the derivation, so it is never the leaf displaced for it."""
    trainer = SupraTokTrainer(vocab_size=400, min_frequency=1)
    trainer._set_alphabet([("abcxyz ", 1)])
    trainer.heldout = [" a b c" * 10, "zx" * 5, "xy" * 5]
    trainer.verification = trainer.heldout
    merges = [(" ", "a"), (" ", "b"), (" ", "c"), (" a", " "), ("b", " c"), (" a ", "b c"),
              ("x", "y"), ("z", "x")]
    common = {"npmi": 0.9, "boundary_entropy": 3.0, "context_peak": 0.1, "avg_idf": 1.0,
              "internal_npmi": 0.9, "stopword_ratio": 0.0, "avg_word_len": 1.0}
    spans = [_Span((" a", " b", " c"), 50, 10, 5, char_len=6, **common)]
    budget = len(trainer._vocab(merges))
    out = trainer._expression(merges, spans, budget)[0]
    assert len(trainer._vocab(out)) == budget
    assert (" a ", "b c") in out
    tokenizer = build_bpe_tokenizer(trainer._vocab(out), out, [])
    assert tokenizer.encode(" a b c").tokens == [" a b c"]


def test_derivation_prefers_existing_tokens_and_admitted_intermediates():
    pieces = [" to", " the", " city"]
    plan, new, bad = _derive(pieces, {" the city"})
    assert plan == [(" the", " city"), (" to", " the city")]
    assert (new, bad) == (1, 0)
    assert _derive(pieces, set(), admitted=set())[1:] == (2, 1)
    assert _derive(pieces, set(), admitted={" the city"})[2] == 0


def test_traditional_keeps_the_word_internal_share_less_the_attached_share(monkeypatch):
    """Of the joint run's merges that cross no space, the attached share goes to MULTIWORD
    and TRADITIONAL keeps the rest."""
    runs, sizes = [], []
    real_run, real_pass1 = SupraTokTrainer._bpe_run, SupraTokTrainer._pass1

    def run(self, items, budget):
        out = real_run(self, items, budget)
        runs.append((items, out))
        return out

    def pass1(self, trad, seeds, size, budget, fill, reserve=0):
        sizes.append((size, reserve))
        return real_pass1(self, trad, seeds, size, budget, fill, reserve=reserve)

    monkeypatch.setattr(SupraTokTrainer, "_bpe_run", run)
    monkeypatch.setattr(SupraTokTrainer, "_pass1", pass1)
    docs = _phrases_and_punctuated_words()
    trainer = SupraTokTrainer(vocab_size=1500, min_frequency=2, seed_min_frequency=60,
                              enable_expression=False)
    trainer.train(docs)
    joint = [out for items, out in runs if any(_CROSSES.search(text) for text, _ in items)]
    within = {a + b for a, b in joint[0] if not _CROSSES.search(a + b)}
    attached = trainer.stats["attached_share"]
    assert len(joint) == 1 and attached > 0
    assert attached == sum(1 for t in within if len(_CHUNK.findall(t)) > 1)
    assert sizes[0] == (len(within) - attached, attached)


def _tied_attached_corpus():
    """Words that each occur with "," and "." equally often, more of them than MULTIWORD
    has room for, so its last merges are picked among equal counts."""
    rng = random.Random(3)
    filler = list(dict.fromkeys("".join(rng.choice(_CONSONANTS) + rng.choice(_VOWELS)
                                        for _ in range(2)) for _ in range(400)))[:150]
    docs = []
    for i, word in enumerate(filler):
        for k in range(8):
            docs.append(" ".join(filler[(i * 7 + k * 13 + j) % 150] for j in range(8))
                        + " " + word + ", " + word + ".")
    return docs


def test_multiword_joins_attached_words_without_any_unit():
    """A word joins what is attached to it even when no seed or screened bigram
    exists: the attached share and MULTIWORD do not depend on the units."""
    rng = random.Random(0)
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]
    docs = [" ".join(rng.choice(words) + rng.choice([",", ".", ";", ""]) for _ in range(20))
            for _ in range(2000)]
    trainer = SupraTokTrainer(vocab_size=400, min_frequency=2, enable_seeds=False,
                              enable_expression=False)
    vocab, _ = trainer.train(docs)
    assert trainer.stats["pass2_candidates"] == 0
    assert trainer.stats["pass2_merges"] > 0 and trainer.stats["attached_share"] > 0
    assert any(t.strip(" ,.;") in words and t[-1] in ",.;" for t in vocab)


def test_multiword_asks_again_with_a_larger_margin(monkeypatch):
    """When the chain trainer stops at its cap with T not full (merges skipped), it is
    asked again for more."""
    asks = []
    real = trainer_module._train_bpe

    def spied(items, trainer, sep):
        out = real(items, trainer, sep)
        if sep != "\x00":
            return out
        asks.append(trainer.vocab_size)
        # The first answer repeats one merge, so all but one are skipped.
        return out[:1] * trainer.vocab_size if len(asks) == 1 else out

    monkeypatch.setattr(trainer_module, "_train_bpe", spied)
    trainer = SupraTokTrainer(vocab_size=600, min_frequency=2, enable_seeds=False,
                              enable_expression=False)
    vocab, _ = trainer.train(_tied_attached_corpus())
    assert len(asks) >= 2 and asks[1] > asks[0]
    assert len(vocab) == 600


def test_attached_runs_never_cross_a_space():
    """Where two seeds overlap (in "a b c d e f" the unit "a b c d" is taken, "c d e f" is
    not), the junctions of the seed left out still pass for MULTIWORD, but its words are not
    attached to one another."""
    a, b, c, d, e, f = (" " + w for w in _filler_words(6, 77))
    docs = ([a + b + c + d + "  x"] * 150 + [c + d + e + f + "  y"] * 150
            + [a + b + c + d + e + f + "  z"] * 60)
    trainer = SupraTokTrainer(enable_expression=False)
    trainer._read(lambda: iter(docs))
    mined = trainer._mine()
    assert set(mined.seeds) == {(a, b, c, d), (c, d, e, f)}
    assert not [text for text, _ in mined.plain if _CROSSES.search(text)]


# ------------------------------------------------------------------ 3. seeds

def test_seeds_are_word_ngrams(expression):
    """Figure 1 line 2: n in {2..6} n-grams of words joined by single spaces."""
    trainer = expression[0]
    assert trainer.seeds
    for seed in trainer.seeds:
        assert 2 <= len(seed) <= trainer.max_seed_ngram
        assert all(_WORD.fullmatch(w) for w in seed)
        assert all(w.startswith(" ") for w in seed[1:])


def test_every_seed_is_a_single_token(expression):
    """Figure 1 line 3: the selected seeds are entries of T."""
    trainer, vocab, merges = expression
    tokenizer = build_bpe_tokenizer(vocab, merges, [])
    assert all(len(tokenizer.encode("".join(seed)).tokens) == 1 for seed in trainer.seeds)


def test_seeds_that_do_not_fit_are_dropped_not_pass_one():
    """When the seeds would overflow T, those that no longer fit are dropped: pass 1 keeps its
    share, T is filled exactly, and every seed that entered is one token."""
    rng = random.Random(0)
    phrases = [" ".join(_filler_words(6, seed)) for seed in range(40)]
    docs = [" ".join(rng.choice(phrases) for _ in range(10)) for _ in range(600)]
    trainer = SupraTokTrainer(vocab_size=800)
    vocab, merges = trainer.train(docs)
    assert len(vocab) == 800
    assert trainer.stats["pass1_merges"] > 0
    assert 0 < trainer.stats["seed_units"] < trainer.stats["seeds_mined"]
    tokenizer = build_bpe_tokenizer(vocab, merges, [])
    assert all(len(tokenizer.encode("".join(seed)).tokens) == 1 for seed in trainer.seeds)


def test_seeds_that_cover_more_text_are_placed_first():
    """When T cannot hold every seed, the overlap control's order applies: the text a seed
    covers, count * (n - 1), not its count alone."""
    trainer = SupraTokTrainer(vocab_size=400, min_frequency=1)
    trainer._set_alphabet([(" abcdef", 1)])
    seeds = {(" a", " b"): 100, (" c", " d", " e", " f"): 60}
    budget = len(trainer._base) + 7  # the four-word seed needs 7 new tokens, the pair 3
    _, _, _, kept, _ = trainer._pass1([], seeds, 0, budget, fill=False)
    assert kept == {(" c", " d", " e", " f"): 60}


def test_derivations_complete_words_before_joining_them():
    """A seed or span whose word T still splits is built word by word: ' or' + 'der'
    becomes ' order' before it is joined, so no intermediate token is a fragment."""
    plan, _, _ = _derive([" in", " or", "der", " to"], set())
    outputs = [a + b for a, b in plan]
    assert outputs[0] == " order" and outputs[-1] == " in order to"
    bounds = _bounds(" in order to")
    for out in outputs:
        start = " in order to".index(out)
        inside_one_word = not any(start < x < start + len(out) for x in bounds)
        assert inside_one_word or (start in bounds and start + len(out) in bounds), out


def test_merge_plans_are_followed_by_bpe_on_repeated_pieces():
    """With repeated pieces BPE applies a merge at its leftmost occurrence, so the cheapest
    derivation of ' ha ha ha' (' ha' + ' ha ha') would never form; the plan is built in
    BPE's order instead. Balanced plans that BPE does follow are kept."""
    for pieces in ([" ha", " ha", " ha"], ["a", "b", "a", "b"], [" in", " or", "der", " to"]):
        plan, formed = _plan(pieces, {}, 0, set())
        after = {pair: i for i, pair in enumerate(plan)}
        assert _bpe(pieces, {}, after) == ["".join(pieces)]
        assert "".join(pieces) in formed
    assert _plan(["a", "b", "a", "b"], {}, 0, set())[0] == [("a", "b"), ("ab", "ab")]


def test_repeated_word_seed_is_one_token():
    rng = random.Random(0)
    filler = _filler_words(300, 5)
    docs = [" ".join(rng.choice(filler) for _ in range(8)) + " ha ha ha "
            + " ".join(rng.choice(filler) for _ in range(8)) for _ in range(400)]
    trainer = SupraTokTrainer(vocab_size=1200, min_frequency=2, enable_multiword=False,
                              enable_expression=False)
    vocab, merges = trainer.train(docs)
    assert (" ha", " ha", " ha") in trainer.seeds
    tokenizer = build_bpe_tokenizer(vocab, merges, [])
    assert tokenizer.encode(" ha ha ha").tokens == [" ha ha ha"]


@pytest.mark.parametrize("fraction", [0.0, 0.5, -0.1])
def test_expression_requires_two_held_out_shards(fraction):
    with pytest.raises(ValueError):
        SupraTokTrainer(holdout_fraction=fraction)
    if fraction == 0.0:
        SupraTokTrainer(holdout_fraction=0.0, enable_expression=False)


def test_expression_without_held_out_documents_is_reported(caplog):
    text = next(t for t in (f"only a few documents {i}" for i in range(100))
                if shard_of(t, 0.1) == "train")
    trainer = SupraTokTrainer(vocab_size=300, min_frequency=1)
    trainer.train([text] * 5)
    assert trainer.stats["pass3_accepted"] == 0
    assert "EXPRESSION skipped" in caplog.text


def _languages(n_per_language=120):
    languages = ["en", "de", "nl", "fr", "es", "it", "ru", "uk", "pl", "zh", "ja", "ko",
                 "tr", "fi"]
    return [(f"{lang} document {i} with some text", lang)
            for i in range(n_per_language) for lang in languages]


def test_shards_ignore_document_order():
    docs = _languages(30)
    first, second = SupraTokTrainer(), SupraTokTrainer()
    first._read(lambda: iter(docs))
    second._read(lambda: iter(docs[::-1]))
    assert first.heldout and sorted(first.heldout) == sorted(second.heldout)
    assert first.verification and sorted(first.verification) == sorted(second.verification)
    assert not set(first.heldout) & set(first.verification)


def test_held_out_shard_covers_every_language_of_an_interleaved_corpus():
    """Every tenth position would hold out only half of 14 interleaved languages."""
    docs = _languages()
    trainer = SupraTokTrainer()
    trainer._read(lambda: iter(docs))
    assert {t.split()[0] for t in trainer.heldout} == {lang for _, lang in docs}


def test_duplicate_documents_share_a_shard():
    text = next(t for t, _ in _languages() if shard_of(t, 0.1) == "held-out")
    trainer = SupraTokTrainer()
    trainer._read(lambda: iter([text] * 5))
    assert trainer.heldout == [text] * 5 and trainer.verification == []


def test_shards_are_collected_only_for_expression():
    trainer = SupraTokTrainer(enable_expression=False)
    trainer._read(lambda: iter(t for t, _ in _languages(10)))
    assert trainer.heldout == [] and trainer.verification == []
    assert not trainer._shard.any()


def test_punctuation_pretokens_are_words():
    """Any pre-token other than whitespace is a word, so punctuation joins phrases."""
    rng = random.Random(2)
    filler = _filler_words(60, 3)
    docs = [f"{rng.choice(filler)}, and {rng.choice(filler)}" for _ in range(60)]
    trainer = SupraTokTrainer(seed_min_frequency=30, max_seed_ngram=2,
                              enable_multiword=False, enable_expression=False)
    trainer._read(lambda: iter(docs))
    assert (",", " and") in trainer._mine().seeds


def test_seed_overlap_control_keeps_whole_phrases():
    """Section 3.3: coverage favors a phrase over its near-duplicate fragments, even
    when a fragment is also seen on its own."""
    rng = random.Random(5)
    filler = _filler_words(40, 6)
    docs = ([f"{rng.choice(filler)} in order to {rng.choice(filler)}" for _ in range(40)]
            + [f"{rng.choice(filler)} in order" for _ in range(10)])
    trainer = SupraTokTrainer(seed_min_frequency=10, max_seed_ngram=3,
                              enable_multiword=False, enable_expression=False)
    trainer._read(lambda: iter(docs))
    assert list(trainer._mine().seeds) == [(" in", " order", " to")]


def test_seed_shortfall_is_reported_once(monkeypatch, caplog):
    """Seeds left out are reported for the final pass sizes, not once per refill round."""
    monkeypatch.setattr(SupraTokTrainer, "_multiword",
                        lambda self, merges, budget, junctions: (merges, 0, 0))
    docs = _phrases_and_punctuated_words()
    trainer = SupraTokTrainer(vocab_size=1000, min_frequency=2, seed_min_frequency=60,
                              enable_expression=False)
    with caplog.at_level("WARNING"):
        trainer.train(docs)
    reports = [r.getMessage() for r in caplog.records if "seed phrases left out" in r.getMessage()]
    assert reports and len(reports) == len({m.split(": ")[-1] for m in reports})


# ---------------------------------------------------------- 4. PMI screening

@pytest.fixture(scope="module")
def screened():
    """A tight collocation of two rare words against a frequent but loose pair."""
    rng = random.Random(3)
    fillers = ["the", "of", "alpha", "beta", "gamma", "delta"]
    docs = []
    for i in range(200):
        parts = [rng.choice(fillers) for _ in range(10)]
        if i % 20 == 0:
            parts.insert(5, "zylphon quarnex")
        if i % 70 == 0:
            parts.insert(3, "vexil torbund")
        docs.append(" ".join(parts))
    trainer = SupraTokTrainer(enable_seeds=False, enable_expression=False)
    trainer._read(lambda: iter(docs))
    return trainer, docs, trainer._mine().bigrams


def _bigram_pmi(docs, pair):
    """Eq. 2 for a word bigram, counted by hand."""
    seqs = [_CHUNK.findall(d) for d in docs]
    words = Counter(w for s in seqs for w in s)
    pairs = Counter(p for s in seqs for p in zip(s, s[1:]))
    n1, n2 = sum(words.values()), sum(len(s) - 1 for s in seqs)
    joint = pairs[pair] / n2
    return pairs[pair], math.log2(joint / ((words[pair[0]] / n1) * (words[pair[1]] / n1)))


def test_screen_admits_tight_collocation(screened):
    trainer, docs, keep = screened
    count, pmi = _bigram_pmi(docs, (" zylphon", " quarnex"))
    assert count >= trainer.screening_frequency
    assert pmi >= trainer.screening_pmi
    assert keep[(" zylphon", " quarnex")] == count


def test_screen_rejects_frequent_but_loose_pair(screened):
    """Frequency alone does not pass the screen; PMI must pass too."""
    trainer, docs, keep = screened
    count, pmi = _bigram_pmi(docs, (" the", " of"))
    assert count >= trainer.screening_frequency
    assert pmi < trainer.screening_pmi
    assert (" the", " of") not in keep


def test_screen_rejects_a_tight_pair_below_the_frequency_floor(screened):
    """Section 3.5 screens on both PMI and frequency; high PMI alone is not enough."""
    trainer, docs, keep = screened
    count, pmi = _bigram_pmi(docs, (" vexil", " torbund"))
    assert pmi >= trainer.screening_pmi
    assert count < trainer.screening_frequency
    assert (" vexil", " torbund") not in keep


# ------------------------------------------------------------ 5. Appendix A

def test_appendix_a_constants():
    assert A.TAU_BE == 1.15
    assert A.PMI_FLOOR == 4.4
    assert A.MIN_COUNT == 3
    assert A.BOUNDARY_GATE_FLOOR == 0.18
    assert A.STOPWORD_CEILING == 0.75
    assert A.SCORE_FLOOR == 0.40
    assert A.NO_HOLDOUT_PENALTY == 0.45
    assert A.LEAF_DF_WEIGHT == 0.35
    assert A.PRIORITY_SCORE_WEIGHT == 0.03


BASE = {"gain": 3, "count": 100, "holdout_count": 10, "holdout_df": 5, "npmi": 0.8,
        "boundary_entropy": 2.0, "avg_idf": 1.0, "internal_npmi": 0.0,
        "context_peak": 0.0, "stopword_ratio": 0.0, "n_words": 2,
        "avg_word_len": 5.0, "char_len": 10}


def _score(**overrides):
    return A.score(**{**BASE, **overrides})


def test_boundary_gate_is_a_sigmoid_centered_on_tau_be():
    """Eq. 8: g_B = sigmoid(1.75, H - 1.15)."""
    assert A.sigmoid(1.75, 1.15 - A.TAU_BE) == pytest.approx(0.5)
    values = [A.sigmoid(1.75, h - A.TAU_BE) for h in (0.5, 1.0, 1.15, 1.5, 3.0)]
    assert values == sorted(values)
    ratio = _score(boundary_entropy=A.TAU_BE) / _score()
    assert ratio == pytest.approx(0.5 / A.sigmoid(1.75, 2.0 - A.TAU_BE))


def test_boundary_gate_floor_zeroes_the_score():
    assert A.sigmoid(1.75, 0.0 - A.TAU_BE) < A.BOUNDARY_GATE_FLOOR
    assert _score(boundary_entropy=0.0) == 0.0


def test_npmi_gate_uses_slope_six_and_offset_022():
    """Eq. 8: g_N = sigmoid(6.0, NPMI - 0.22)."""
    ratio = _score(npmi=0.4) / _score()
    assert ratio == pytest.approx(A.sigmoid(6.0, 0.4 - 0.22)
                                  / A.sigmoid(6.0, 0.8 - 0.22))
    assert _score(npmi=0.22) / _score() == pytest.approx(
        0.5 / A.sigmoid(6.0, 0.8 - 0.22))


def test_stopword_penalty_shape():
    """Eq. 9: p_stop = max(0.30, 1 - 0.78 * r ** 1.35)."""
    assert _score(stopword_ratio=0.5) / _score() == pytest.approx(
        1.0 - 0.78 * 0.5 ** 1.35)
    assert _score(stopword_ratio=1.0) / _score() == pytest.approx(0.30)


def test_surface_penalty_applies_above_42_chars():
    assert _score(char_len=42) / _score() == pytest.approx(1.0)
    assert _score(char_len=43) / _score() == pytest.approx(0.88)
    assert _score(char_len=400) / _score() == pytest.approx(0.88)


def test_missing_holdout_multiplies_by_the_penalty():
    unpenalised = (BASE["gain"] * (0 + 0.60 * BASE["holdout_df"]
                                   + 0.18 * math.sqrt(BASE["count"]))
                   * A.sigmoid(6.0, BASE["npmi"] - 0.22)
                   * A.sigmoid(1.75, BASE["boundary_entropy"] - A.TAU_BE))
    assert _score(holdout_count=0) == pytest.approx(
        unpenalised * A.NO_HOLDOUT_PENALTY)


def test_score_floor_zeroes_a_marginal_span():
    small = {"gain": 1, "count": 1, "holdout_count": 1, "npmi": 0.22,
             "boundary_entropy": A.TAU_BE + 0.4}
    assert _score(holdout_df=0, **small) == 0.0
    kept = _score(holdout_df=1, **small)
    assert kept > A.SCORE_FLOOR


def test_pmi_and_npmi_are_base_two():
    """Eq. 5 by hand: joint 0.5 over 0.5*0.5 is exactly one bit."""
    value, npmi = A.pmi(2, 4, [0.5, 0.5])
    assert value == pytest.approx(1.0)
    assert npmi == pytest.approx(1.0)
    assert A.pmi(1, 8, [0.5, 0.5])[0] == pytest.approx(-1.0)
    assert A.pmi(1, 4, [0.0, 0.5]) == (-math.inf, -math.inf)


def test_branching_entropy_is_base_two():
    """Eq. 6 by hand: two equal continuations is one bit, four is two."""
    assert A.entropy({"a": 1, "b": 1}) == pytest.approx(1.0)
    assert A.entropy({"a": 1, "b": 1, "c": 1, "d": 1}) == pytest.approx(2.0)
    assert A.entropy({"a": 2, "b": 1, "c": 1}) == pytest.approx(1.5)
    assert A.entropy({"a": 7}) == 0.0


def test_leaf_surgery_weights():
    """Eqs. 13-14 use LEAF_DF_WEIGHT and PRIORITY_SCORE_WEIGHT."""
    assert A.addition_gain(4, 3, 10) == pytest.approx(4 * 2 + 0.35 * 10)
    assert A.leaf_utility(4, 10) == pytest.approx(4 + 0.35 * 10)
    assert A.priority(2.0, 5.0) == pytest.approx(2.0 + 0.03 * 5.0)


def test_mined_statistics_match_appendix_a_by_brute_force(source):
    """The vectorized n-gram engine against Eqs. 5-7 recomputed naively with the reference
    functions of appendix_a: the same candidate spans pass the hard rejections, and every
    statistic of every span agrees."""
    docs = list(source())
    trainer = SupraTokTrainer(vocab_size=VOCAB_SIZE, min_frequency=2, holdout_fraction=0.2)
    trainer._read(lambda: iter(docs))
    mined = {span.words: span for span in trainer._mine().spans}
    seqs = [_CHUNK.findall(d) for d in docs]
    train = [q for d, q in zip(docs, seqs) if shard_of(d, 0.2) == "train"]
    held = [q for d, q in zip(docs, seqs) if shard_of(d, 0.2) == "held-out"]

    def grams(seq, n):
        for p in range(len(seq) - n + 1):
            g = tuple(seq[p:p + n])
            if all(_WORD.fullmatch(w) for w in g) and all(w.startswith(" ") for w in g[1:]):
                yield p, g

    unigrams = Counter(w for q in train for w in q if _WORD.fullmatch(w))
    total = sum(unigrams.values())
    df = Counter(w for q in train for w in set(q) if _WORD.fullmatch(w))
    occurrences, windows = {}, Counter()
    for q in train:
        for n in range(2, trainer.max_span + 1):
            for p, g in grams(q, n):
                occurrences.setdefault(g, []).append((q, p))
                windows[n] += 1
    held_occ = [Counter(g for n in range(2, trainer.max_span + 1) for _, g in grams(q, n))
                for q in held]

    def npmi(g):
        return A.pmi(len(occurrences.get(g, ())), windows[len(g)],
                     [unigrams[w] / total for w in g])[1]

    expected = {}
    for g, occ in occurrences.items():
        c_ho = sum(c[g] for c in held_occ)
        stop = sum(w.strip().lower() in trainer.stopwords for w in g) / len(g)
        pmi = A.pmi(len(occ), windows[len(g)], [unigrams[w] / total for w in g])[0]
        if len(occ) < A.MIN_COUNT or not c_ho or pmi < A.PMI_FLOOR or stop >= A.STOPWORD_CEILING:
            continue
        n = len(g)
        left = Counter(q[p - 1] if p else "" for q, p in occ)
        right = Counter(q[p + n] if p + n < len(q) else "" for q, p in occ)
        h_b = min(A.entropy(left), A.entropy(right))
        if A.sigmoid(1.75, h_b - A.TAU_BE) < A.BOUNDARY_GATE_FLOOR:
            continue
        expected[g] = {
            "train_count": len(occ), "holdout_count": c_ho,
            "holdout_df": sum(1 for c in held_occ if c[g]),
            "npmi": npmi(g), "boundary_entropy": h_b,
            "context_peak": max(A.peak(left), A.peak(right)),
            "avg_idf": sum(math.log((len(train) + 1) / (1 + df[w])) for w in g) / n,
            "internal_npmi": sum(npmi(g[i:i + 2]) for i in range(n - 1)) / (n - 1),
            "stopword_ratio": stop, "avg_word_len": sum(len(w.strip()) for w in g) / n,
            "char_len": len("".join(g))}
    assert expected and set(mined) == set(expected)
    for g, fields in expected.items():
        for name, value in fields.items():
            assert getattr(mined[g], name) == pytest.approx(value), (g, name)


def _mined_spans(source, **options):
    trainer = SupraTokTrainer(vocab_size=VOCAB_SIZE, min_frequency=2, **options)
    trainer._read(source)
    return trainer._mine().spans


def test_stopwords_replace_the_default_list(source):
    assert SupraTokTrainer(stopwords=[" The"]).stopwords == {"the"}
    assert any(s.stopword_ratio for s in _mined_spans(source))
    assert not any(s.stopword_ratio for s in _mined_spans(source, stopwords=[]))


def test_max_span_bounds_the_appendix_a_spans(source):
    assert max(len(s.words) for s in _mined_spans(source)) > 2
    assert {len(s.words) for s in _mined_spans(source, max_span=2)} == {2}


def test_expression_scores_no_span_that_is_one_token_or_holds_a_digit():
    """A span T already spells as one token saves nothing, and spans with digits are
    rejected outright (Appendix A's hard rejections)."""
    common = {"npmi": 0.9, "boundary_entropy": 3.0, "context_peak": 0.1, "avg_idf": 1.0,
              "internal_npmi": 0.9, "stopword_ratio": 0.0, "avg_word_len": 4.0}
    span = _Span((" new", " york"), 50, 10, 5, char_len=9, **common)
    trainer = SupraTokTrainer()
    assert trainer._score(span, 2) > 0
    assert trainer._score(span, 1) == 0
    assert trainer._score(span._replace(words=(" route", " 66")), 2) == 0


def test_leaf_surgery_retires_another_leaf_before_a_seed():
    """The seed " x y" never occurs in held-out text, so it is the leaf of least utility,
    yet surgery retires the next one ("za") to make room for " a b"."""
    trainer = SupraTokTrainer(vocab_size=400, min_frequency=1)
    trainer._set_alphabet([("abxyz ", 1)])
    trainer.seeds = {(" x", " y"): 50}
    trainer.heldout = [" a b" * 10, "za" * 5]
    trainer.verification = trainer.heldout
    merges = [(" ", "a"), (" ", "b"), (" ", "x"), (" ", "y"), (" x", " y"), ("z", "a")]
    common = {"npmi": 0.9, "boundary_entropy": 3.0, "context_peak": 0.1, "avg_idf": 1.0,
              "internal_npmi": 0.9, "stopword_ratio": 0.0, "avg_word_len": 1.0}
    spans = [_Span((" a", " b"), 50, 10, 5, char_len=4, **common)]
    out, _, accepted, retired, _ = trainer._expression(merges, spans, len(trainer._vocab(merges)))
    assert (accepted, retired) == (1, 1)
    assert (" x", " y") in out and ("z", "a") not in out


# --------------------------------------------- 6. script-aware processing

@pytest.mark.parametrize("char,supercategory", [
    ("a", "LM"), ("Ω", "LM"), ("中", "LM"), ("\u0301", "LM"),
    ("!", "PS"), ("$", "PS"), ("+", "PS"),
    ("7", "N"), ("٣", "N"),
    (" ", "Z"), ("\n", "Z"), ("\t", "Z"), ("\u00a0", "Z"),
    ("\x00", "C"), ("\u200b", "C"),
])
def test_supercategories(char, supercategory):
    assert S.classify(char) == supercategory


def test_bigram_gate_threshold_is_the_paper_minimum():
    assert S.MIN_BIGRAM_COUNT == 50
    assert S.BigramGate().min_count == 50


def test_bigram_gate_blocks_below_and_permits_above_min_count():
    """Section 3.4.1: a junction seen fewer than 50 times may not merge."""
    assert not S.BigramGate({"北京": 49}).allows("北", "京")
    assert S.BigramGate({"北京": 50}).allows("北", "京")
    assert S.BigramGate({"東京都": 0, "京都": 60}).allows("東京", "都市")


def test_bigram_gate_passes_non_boundary_free_pairs_ungated():
    """The mechanism exists for zh/ja/th only; everything else merges freely."""
    gate = S.BigramGate({})
    assert gate.allows("New", "York")
    assert gate.allows("中", "a")
    assert gate.allows(" ", "京")
    assert gate.allows("中", " ")


def test_mine_suffixes_finds_turkish_endings():
    stems = ["ev", "kitap", "göz", "araba", "okul", "masa", "kalem", "defter",
             "yol", "kapı", "çocuk", "bahçe", "şehir", "deniz", "orman"]
    text = " ".join(s + e for s in stems
                    for e in ("", "ler", "lar", "lerde", "larda", "lerin"))
    mined = S.mine_suffixes([text], "tr")
    assert {"ler", "lar"} <= set(mined[:8])
    assert {"de", "da"} <= set(mined)
    assert all(2 <= len(s) <= 5 for s in mined)
    assert len(mined) <= S.SUFFIX_TOP_K


def test_hamza_normalization_folds_the_carriers():
    assert S.normalize_hamza("أإآٱ") == "ا" * 4
    assert S.normalize_hamza("ؤ") == "و"
    assert S.normalize_hamza("ئ") == "ي"
    assert S.normalize_hamza("ءا") == "ءا"  # bare hamza and bare alef untouched


def test_bigram_gate_splits_at_refused_junctions():
    gate = S.BigramGate({"北京": 60, "大学": 60})
    assert gate.split("北京大学") == ["北京", "大学"]
    assert gate.split("New York") == ["New York"]


@pytest.mark.parametrize("script_aware", [True, False])
def test_training_gate_keeps_merges_off_rare_junctions(script_aware):
    """Section 3.4: 京大 is seen 10 times, below the minimum of 50, so with script-aware
    processing no token may span it."""
    docs = ["北京 大学"] * 60 + ["北京大学"] * 10
    tokenizer = train_tokenizer(lambda: iter(docs), vocab_size=300, min_frequency=2,
                                script_aware=script_aware, enable_seeds=False,
                                enable_multiword=False, enable_expression=False)
    pieces = [tokenizer.decode([i]) for i in tokenizer.encode("北京大学").ids]
    assert any("京大" in p for p in pieces) != script_aware


def test_multiword_does_not_join_across_cjk_sentence_ends(source):
    """Chinese puts no space after 。 or ，; the junction to the next clause is a boundary
    the gate has no bigram evidence for, so no token spans it. A word keeps its trailing
    punctuation, as " York," does."""
    rng = random.Random(1)
    clauses = ["你好。", "谢谢。", "是的。", "我们走吧。", "不是。", "好的，", "对，"]
    zh = [("".join(rng.choice(clauses) for _ in range(40)), "zh") for _ in range(200)]
    tokenizer = train_tokenizer(list(source()) + zh, vocab_size=900, min_frequency=2,
                                enable_expression=False)
    vocab = tokenizer.get_vocab()
    assert not [t for t in vocab if regex.search(r"[。，]\p{Han}", t)]
    assert [t for t in vocab if regex.search(r"\p{Han}[。，]", t)]


def test_multiword_does_not_rejoin_what_the_gate_split():
    """The gate's split leaves 北京 and 大学 with no space between them, which is not a word
    and what is attached to it: MULTIWORD may not join them either."""
    filler = _filler_words(400, 11)
    docs = [" ".join(filler[i % 400:i % 400 + 8]) + " 北京 大学" for i in range(300)]
    docs += ["北京大学 " + " ".join(filler[i % 400:i % 400 + 8]) for i in range(30)]
    tokenizer = train_tokenizer(lambda: iter(docs), vocab_size=1500, min_frequency=2,
                                enable_seeds=False, enable_expression=False)
    assert tokenizer.supratok_stats["pass2_merges"] > 0
    assert not [t for t in tokenizer.get_vocab() if "京大" in t]


def test_hamza_normalization_applies_in_training_and_encoding():
    """Section 3.4.1: hamza is folded before frequency counting and at encode time."""
    docs = ["أحمد إلى آخر"] * 20
    options = {"vocab_size": 300, "min_frequency": 2, "enable_seeds": False,
               "enable_multiword": False, "enable_expression": False}
    folded = SupraTokTrainer(normalize_hamza=True, **options)
    folded.train(docs)
    kept = SupraTokTrainer(**options)
    kept.train(docs)
    assert "أ" in kept.alphabet and "أ" not in folded.alphabet
    tokenizer = train_tokenizer(lambda: iter(docs), normalize_hamza=True, **options)
    assert tokenizer.decode(tokenizer.encode("أحمد إلى").ids) == "احمد الى"


def test_suffix_weighting_applies_to_tagged_turkish_documents():
    """Section 3.4 / Appendix E: suffixes are mined per language from its own documents."""
    stems = ["ev", "kitap", "göz", "araba", "okul", "masa", "kalem", "defter",
             "yol", "kapı", "çocuk", "bahçe", "şehir", "deniz", "orman"]
    text = " ".join(s + e for s in stems for e in ("", "ler", "lar", "lerde", "larda", "lerin"))
    options = {"vocab_size": 600, "min_frequency": 2, "enable_seeds": False,
               "enable_multiword": False, "enable_expression": False}
    tagged = SupraTokTrainer(**options)
    _, weighted = tagged.train([(text, "TR")] * 30)
    plain = SupraTokTrainer(**options)
    _, unweighted = plain.train([text] * 30)
    assert tagged.suffixes["tr"] and plain.suffixes == {}

    def rank(merges, token):
        return next((i for i, (a, b) in enumerate(merges) if a + b == token), len(merges))

    moves = [rank(unweighted, s) - rank(weighted, s) for s in tagged.suffixes["tr"]]
    assert min(moves) >= 0 and max(moves) > 0

    def extras(weight):
        trainer = SupraTokTrainer(**options, suffix_weight=weight)
        trainer._read(lambda: iter([(text, "tr")] * 30))
        return trainer._mine().extras

    assert extras(1) and extras(3) == [(s, 3 * c) for s, c in extras(1)]


def test_monolingual_boundary_free_scripts_train():
    """Unspaced Chinese and Thai corpora, where no word bigram reaches the screen, train
    and round-trip."""
    for text in ("北京大学是中国的大学北京大学", "ภาษาไทยง่ายมาก"):
        tokenizer = train_tokenizer([text] * 100, vocab_size=400, min_frequency=1)
        assert tokenizer.decode(tokenizer.encode(text).ids) == text


def test_turkish_fold_handles_the_decomposed_dotted_capital_i():
    assert S.fold("İSTANBUL", "tr") == S.fold("I\u0307STANBUL", "tr") == "istanbul"
    assert S.fold("IRMAK", "tr") == "ırmak"


def test_hamza_folds_composed_and_decomposed_letters():
    composed = "أإآؤئ"
    decomposed = unicodedata.normalize("NFD", composed)
    assert decomposed != composed
    for text in (composed, decomposed):
        assert S.normalize_hamza(text) == "اااوي"
        assert S.hamza_normalizer().normalize_str(text) == "اااوي"


def test_hamza_folds_decomposed_letters_with_vowel_marks():
    """Canonical order puts a vowel mark (fatha, shadda) before the hamza mark, so the
    decomposed letter folds through the marks, exactly as the composed one does."""
    for text in ("إَ", "أُ", "ؤَ", "ئِّ", "إَإ"):
        composed, decomposed = (unicodedata.normalize(f, text) for f in ("NFC", "NFD"))
        folded = S.normalize_hamza(composed)
        assert S.normalize_hamza(decomposed) == unicodedata.normalize("NFD", folded)
        assert S.hamza_normalizer().normalize_str(decomposed) == S.normalize_hamza(decomposed)


def test_alphabet_counts_the_corpus_not_the_suffix_weights():
    """A character below min_frequency stays out of T even inside a mined suffix."""
    stems = ["ev", "kapı", "masa", "okul", "araba", "kitap", "kalem", "bahçe"]
    docs = [(" ".join(stems * 3) + " evler kapılar masalar okullar arabalar", "tr")] * 10
    docs.append(("evğa kapığa masağa", "tr"))
    trainer = SupraTokTrainer(vocab_size=600, min_frequency=4, enable_expression=False)
    trainer.train(docs)
    assert any("ğ" in suffix for suffix in trainer.suffixes["tr"])
    assert "ğ" not in trainer.alphabet


def test_alphabet_leaves_characters_rarer_than_one_in_five_million_to_bytes():
    """On 100M training characters a character needs 20 occurrences, not min_frequency's 5."""
    trainer = SupraTokTrainer(vocab_size=1000)
    trainer._set_alphabet([(" a", 49_999_980), ("b", 21), ("c", 19)])
    assert "b" in trainer.alphabet
    assert "c" not in trainer.alphabet


def _attached_after(text: str, lang: str, test) -> set[tuple[str, str]]:
    """The (word, next) pre-token pairs joined across an attached junction for which
    `test(word, next)` holds."""
    trainer = SupraTokTrainer(enable_expression=False)
    trainer._read(lambda: iter([(text, lang)] * 3))
    junctions = trainer._mine().junctions
    pairs = [(trainer.types[trainer.ids[i - 1]], trainer.types[trainer.ids[i]])
             for i in range(1, len(trainer.ids)) if junctions[i]]
    return {pair for pair in pairs if test(*pair)}


def test_words_keep_the_opening_bracket_after_them():
    """A word and the bracket or quote after it are one whitespace-delimited word, in
    English (` print(`) as in Chinese or Japanese (的《, いわゆる「)."""
    assert _attached_after("call print(x) and len(y) now", "en",
                           lambda a, b: b[:1] == "(") == {(" print", "("), (" len", "(")}
    text = "他读了《红楼梦》和“三国”。北京（中国）很大。いわゆる「漫画」だ。"
    assert {b for _, b in _attached_after(text, "zh", lambda a, b: b[:1] in "《“（「")} \
        == {"《", "“", "（", "「"}


def test_multiword_does_not_join_punctuation_to_the_cjk_letter_after_it():
    """With no space after punctuation, the junction to a Chinese letter is where the
    space would be, whatever the punctuation (" “你", "(北")."""
    pairs = _attached_after("他说“你好”(北京)", "zh",
                            lambda a, b: regex.fullmatch(r"\p{P}+", a) is not None
                            and regex.match(r"\p{Han}", b) is not None)
    assert pairs == set()


def test_hamza_folds_through_two_marks_after_alef():
    text = "\u0627\u064B\u0651\u0654"
    assert unicodedata.normalize("NFD", text) == text
    assert S.normalize_hamza(text) == "\u0627\u064B\u0651"
    assert S.hamza_normalizer().normalize_str(text) == "\u0627\u064B\u0651"


# ------------------------------------------------------ 7. entropy curation

def test_npmi_of_a_span_that_is_every_occurrence_is_one():
    assert A.pmi(5, 5, [1.0])[1] == pytest.approx(1.0)


def test_entropy_filter_uses_character_bigrams_base_two():
    """Eq. 1 by hand: 'abab' has bigrams ab, ba, ab."""
    expected = -(2 / 3 * math.log2(2 / 3) + 1 / 3 * math.log2(1 / 3))
    assert EntropyFilter().calculate_entropy("abab") == pytest.approx(expected)
    assert EntropyFilter().calculate_entropy("a") == 0.0


def test_entropy_bins_use_the_paper_thresholds():
    """Section 3.2: H < 3.0 low, 3.0 <= H <= 4.5 medium, H > 4.5 high."""
    filt = EntropyFilter()
    assert filt.low_threshold == 3.0 and filt.high_threshold == 4.5
    assert filt._bin_for_entropy(2.999) == "low"
    assert filt._bin_for_entropy(3.0) == "medium"
    assert filt._bin_for_entropy(4.5) == "medium"
    assert filt._bin_for_entropy(4.501) == "high"
    assert filt.bin_of("a" * 40) == "low"
    assert filt.bin_of("abcdefghijklmnop" * 8) == "medium"
    assert filt.bin_of("the quick brown fox jumps over the lazy dog") == "high"


def test_three_bin_sampler_rates():
    assert ThreeBinSampler().rates == {"low": 0.1, "medium": 0.5, "high": 0.9}


def test_three_bin_sampler_is_seeded_and_reproducible():
    docs = [f"the quick brown fox jumps over the lazy dog {i}" for i in range(40)]
    seeded = ThreeBinSampler(seed=7)
    repeat = ThreeBinSampler(seed=7)
    other = ThreeBinSampler(seed=99)
    first = [seeded.should_sample(d) for d in docs]
    assert [repeat.should_sample(d) for d in docs] == first
    assert [other.should_sample(d) for d in docs] != first
    assert set(first) == {True, False}
    assert seeded.get_statistics()["bin_distribution"] == {"high": 40}


def test_streaming_loader_applies_the_table_e1_rule(tmp_path):
    """Non-empty stripped text of at least 50 characters."""
    rows = [{"text": "  " + "x" * 60 + "  ", "language": "tr"}, {"text": "short"},
            {"text": ""}, {"text": "y" * 50}]
    path = tmp_path / "docs.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    docs = list(StreamingDataLoader().stream_documents(str(path), with_language=True))
    assert docs == [("x" * 60, "tr"), ("y" * 50, "unknown")]


def test_streaming_loader_skips_lines_it_cannot_read(tmp_path):
    """A byte-order mark is accepted; lines that are not UTF-8, not JSON objects, nested
    too deep for the parser, text-less or hold surrogates are skipped and counted, and a
    language that is not a string is unknown."""
    good, other = "x" * 60, "y" * 60
    deep = b'{"text": "' + b"v" * 60 + b'", "m": ' + b"[" * 5000 + b"]" * 5000 + b"}"
    # How deep a nesting json can parse depends on the Python version.
    try:
        json.loads(deep)
        nested = [("v" * 60, "unknown")]
    except RecursionError:
        nested = []
    lines = [b"\xef\xbb\xbf" + json.dumps({"text": good}).encode(),
             b'{"text": "\xff' + b"z" * 60 + b'"}',
             json.dumps({"text": "z" * 60 + "\ud800"}).encode(),
             b"[1, 2]", b'{"language": "en"}', b"",
             (json.dumps({"text": "w" * 60}) + "\n").encode("utf-16-be"),
             deep,
             json.dumps({"text": other, "language": ["en"]}).encode()]
    path = tmp_path / "docs.jsonl"
    path.write_bytes(b"\n".join(lines) + b"\n")
    loader = StreamingDataLoader(language_filters=["unknown"])
    docs = list(loader.stream_documents(str(path), with_language=True))
    assert docs == [(good, "unknown"), *nested, (other, "unknown")]
    assert loader.stats["invalid_lines"] == 6 - len(nested)


def test_streaming_loader_passes_are_identical(tmp_path):
    rows = [{"text": "ab " * 40 if i % 2 else f"the quick brown fox {i} jumps over the lazy river"}
            for i in range(60)]
    path = tmp_path / "docs.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    loader = StreamingDataLoader(use_entropy_filtering=True)
    first = list(loader.stream_documents(str(path)))
    stats = dict(loader.stats)
    assert 0 < len(first) < 60
    assert list(loader.stream_documents(str(path))) == first
    assert loader.stats == stats


def test_streaming_loader_filters_languages_and_limits_lines(tmp_path):
    rows = [{"text": c * 60, "language": lang} for c, lang in zip("abcd", ["tr", "fi", "tr", "en"])]
    path = tmp_path / "docs.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    kept = StreamingDataLoader(language_filters=["tr"]).stream_documents(str(path))
    assert list(kept) == ["a" * 60, "c" * 60]
    # Language tags are read case-insensitively, as the trainer reads them.
    kept = StreamingDataLoader(language_filters=["TR"]).stream_documents(str(path))
    assert list(kept) == ["a" * 60, "c" * 60]
    first = StreamingDataLoader().stream_documents(str(path), max_documents=2)
    assert list(first) == ["a" * 60, "b" * 60]


def test_space_separated_stream_rejoins_pretokens():
    """One space between pre-tokens; line breaks become spaces too."""
    trainer = SupraTokTrainer(space_separated_stream=True)
    assert trainer._prepare("York, and\nthe end.") == "York , and the end ."
    assert trainer._prepare("the end.\n\nNext") == "the end . Next"
    assert trainer._prepare("alpha\x1cbeta") == "alpha \x1c beta"
    trainer._read(lambda: iter(["York, and\nthe end."] * 2))
    assert " ," in trainer.types and not any("\n" in t for t in trainer.types)
    assert SupraTokTrainer().space_separated_stream is False


def test_screen_defaults_follow_section_3_5():
    """The later-pass screen defaults to Section 3.5's PMI 5.0 and frequency 5."""
    default = SupraTokTrainer()
    assert (default.screening_pmi, default.screening_frequency) == (5.0, 5)


def test_sampler_keeps_each_entropy_bin_at_its_rate():
    """Section 3.2: 10% of low-, 50% of medium- and 90% of high-entropy documents."""
    rng = random.Random(0)
    texts = {"low": lambda: "ab" * 60,
             "medium": lambda: "".join(rng.choice("abc") for _ in range(300)),
             "high": lambda: "".join(rng.choice("abcdefghijklmnopqrstuvwxyz ") for _ in range(300))}
    sampler = ThreeBinSampler()
    for bucket, rate in (("low", 0.1), ("medium", 0.5), ("high", 0.9)):
        docs = [texts[bucket]() for _ in range(3000)]
        assert {sampler.entropy_filter.bin_of(d) for d in docs} == {bucket}
        assert abs(sum(map(sampler.should_sample, docs)) / len(docs) - rate) < 0.03


# ------------------------------------------------------------ 8. determinism

def test_training_is_deterministic(source, expression):
    _, vocab, merges = expression
    _, again_vocab, again_merges = _train(source)
    assert again_vocab == vocab
    assert again_merges == merges


def test_training_ignores_document_order():
    """The same documents in another order give the same tokenizer, as the hash-drawn
    shards do: ties between equally frequent merges are not broken by which word the
    corpus showed first."""
    docs = _tied_attached_corpus()
    results = []
    for seed in range(3):
        shuffled = list(docs)
        random.Random(seed).shuffle(shuffled)
        trainer = SupraTokTrainer(vocab_size=600, min_frequency=2, enable_seeds=False,
                                  enable_expression=False)
        results.append(trainer.train(shuffled))
        assert trainer.stats["pass2_merges"] > 0
    assert results[1] == results[0] and results[2] == results[0]


# ------------------------------------------------------- 9. SentencePiece export

def test_sentencepiece_export_matches_the_tokenizer(expression, source):
    """Section 3.6: on this corpus, the exported model segments text as the tokenizer does."""
    spm = pytest.importorskip("sentencepiece", reason="pip install -e '.[sentencepiece]'")
    _, vocab, merges = expression
    tokenizer = build_bpe_tokenizer(vocab, merges, [])
    processor = spm.SentencePieceProcessor(model_proto=to_sentencepiece(tokenizer))
    for text in list(source())[:50] + ROUNDTRIP_PROBES:
        expected = [t.replace(" ", "\u2581") for t in tokenizer.encode(text).tokens]
        assert processor.encode(text, out_type=str) == expected
        assert processor.decode(processor.encode(text)) == text


def test_sentencepiece_export_keeps_the_space_of_an_unspaced_corpus():
    """The space is always in T, so text with spaces exports exactly even when the
    training corpus had none."""
    spm = pytest.importorskip("sentencepiece", reason="pip install -e '.[sentencepiece]'")
    tokenizer = train_tokenizer(["abcabc"] * 50, vocab_size=300, min_frequency=1)
    assert " " in tokenizer.get_vocab()
    processor = spm.SentencePieceProcessor(model_proto=to_sentencepiece(tokenizer))
    assert processor.encode("abc abc", out_type=str) == ["abc", "\u2581", "abc"]
    assert processor.decode(processor.encode("abc abc")) == "abc abc"


def test_sentencepiece_export_keeps_special_and_added_tokens(source, tmp_path):
    """Added tokens, special or not, are units in both tokenizers; an `<unk>` special
    leaves the UNKNOWN piece another name."""
    spm = pytest.importorskip("sentencepiece", reason="pip install -e '.[sentencepiece]'")
    tokenizer = train_tokenizer(list(source())[:60], special_tokens=["<unk>", "<s>"],
                                vocab_size=400, min_frequency=2, enable_expression=False)
    tokenizer.add_tokens(["zzqq"])
    path = tmp_path / "supratok.model"
    data = to_sentencepiece(tokenizer, str(path))
    assert path.read_bytes() == data
    processor = spm.SentencePieceProcessor(model_file=str(path))
    assert processor.id_to_piece(processor.unk_id()) == "<unk1>"
    for text in ["<s>the company announced", "a <unk> b", "a zzqq b<s>"]:
        expected = [t.replace(" ", "\u2581") for t in tokenizer.encode(text).tokens]
        assert processor.encode(text, out_type=str) == expected
        assert processor.decode(processor.encode(text)) == text


def test_saved_tokenizer_reloads_identically(expression, tmp_path):
    _, vocab, merges = expression
    tokenizer = build_bpe_tokenizer(vocab, merges, ["<pad>"])
    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))
    reloaded = Tokenizer.from_file(str(path))
    for probe in ROUNDTRIP_PROBES + ["<pad> after a special"]:
        assert reloaded.encode(probe).ids == tokenizer.encode(probe).ids
        assert reloaded.decode(reloaded.encode(probe).ids) == probe.replace("<pad>", "")


# ------------------------------------------------ 9b. shortest-path segmentation

def test_alphabet_always_spells_byte_tokens():
    """The characters of '<0xNN>' are always in T, so text that spells a byte token is
    read as characters."""
    trainer = SupraTokTrainer(vocab_size=400, min_frequency=1, enable_expression=False)
    trainer.train(["北京大学"] * 30)
    assert set("<>x0123456789ABCDEF") <= set(trainer.alphabet)


def test_no_learned_token_spells_a_byte_token(source):
    """Text that spells a byte token ("<0x41>") is joined by no pass into that spelling,
    which would collide with the byte token itself."""
    filler = _filler_words(300, 9)
    docs = list(source()) + [f"<0x41>\n{filler[i % 300]}" if i % 2 else "<0x41>"
                             for i in range(400)]
    tokenizer = train_tokenizer(docs, vocab_size=800, min_frequency=2)
    assert tokenizer.decode(tokenizer.encode("<0x41> x").ids) == "<0x41> x"
    assert tokenizer.get_vocab()["<0x41>"] == BYTE_TOKENS.index("<0x41>")


def test_shortest_path_never_uses_more_tokens(expression, source):
    _, vocab, merges = expression
    bpe, shortest = build_bpe_tokenizer(vocab, merges, []), build_shortest_tokenizer(vocab, [])
    for text in list(source())[:40] + ROUNDTRIP_PROBES:
        ids = shortest.encode(text).ids
        assert len(ids) <= len(bpe.encode(text).ids)
        assert shortest.decode(ids) == text


def test_shortest_path_never_reads_text_as_a_byte(expression):
    shortest = build_shortest_tokenizer(expression[1], [])
    for text in ("a<0x41>b", "<0xFF><0x00>", "x <0x7a> y"):
        encoding = shortest.encode(text)
        assert shortest.decode(encoding.ids) == text
        assert not [t for t in encoding.tokens if t in BYTE_TOKENS]


def test_shortest_path_keeps_ids_and_special_tokens(expression):
    _, vocab, merges = expression
    bpe = build_bpe_tokenizer(vocab, merges, ["<pad>"])
    shortest = build_shortest_tokenizer(vocab, ["<pad>"])
    assert {t: i for t, i in shortest.get_vocab().items() if i <= len(vocab)} == bpe.get_vocab()
    encoding = shortest.encode("<pad> the United Nations")
    assert encoding.ids[0] == bpe.token_to_id("<pad>")
    assert shortest.decode(encoding.ids) == " the United Nations"


@pytest.fixture(scope="module")
def chinese_shortest():
    """A shortest-path tokenizer whose alphabet lacks 'u', 'n' and 'k'."""
    docs = ["北京大学是中国的大学北京大学"] * 50 + ["上海交通大学在上海"] * 50
    return train_tokenizer(docs, vocab_size=400, min_frequency=1, segmentation="shortest",
                           enable_expression=False)


def test_shortest_path_never_emits_an_id_outside_the_vocabulary(chinese_shortest):
    """Literal '<unk>' text is read as characters or bytes, never as an extra piece."""
    size = chinese_shortest.supratok_stats["vocab"]
    assert chinese_shortest.get_vocab_size() == size
    for text in ("<unk>", "北京<unk>大学"):
        ids = chinese_shortest.encode(text).ids
        assert max(ids) < size and chinese_shortest.decode(ids) == text


def test_sentencepiece_export_counts_text_outside_the_alphabet_alike(chinese_shortest):
    spm = pytest.importorskip("sentencepiece", reason="pip install -e '.[sentencepiece]'")
    processor = spm.SentencePieceProcessor(model_proto=to_sentencepiece(chinese_shortest))
    for text in ("<unk>", "北京<unk>大学", "Zürich 北京"):
        assert len(processor.encode(text)) == len(chinese_shortest.encode(text).ids)
        assert processor.decode(processor.encode(text)) == text


def test_shortest_path_rejects_a_vocabulary_that_cannot_be_lossless():
    with pytest.raises(InvalidBpeSpec):
        build_shortest_tokenizer({t: i for i, t in enumerate(BYTE_TOKENS + ["a", "b"])}, [])


def test_train_tokenizer_offers_shortest_path(source):
    docs = list(source())[:80]
    tokenizer = train_tokenizer(docs, vocab_size=500, min_frequency=2, segmentation="shortest",
                                enable_expression=False)
    assert json.loads(tokenizer.to_str())["model"]["type"] == "Unigram"
    assert tokenizer.get_vocab_size() == 500
    with pytest.raises(ValueError):
        train_tokenizer(docs, vocab_size=500, segmentation="greedy")


def test_shortest_path_applies_the_normalizer():
    docs = ["أحمد إلى آخر"] * 20
    tokenizer = train_tokenizer(docs, vocab_size=300, min_frequency=2, normalize_hamza=True,
                                segmentation="shortest", enable_seeds=False,
                                enable_multiword=False, enable_expression=False)
    assert tokenizer.decode(tokenizer.encode("أحمد إلى").ids) == "احمد الى"


def test_sentencepiece_export_of_a_shortest_path_tokenizer(expression, source):
    """Ties may break differently, but every text costs the same number of tokens."""
    spm = pytest.importorskip("sentencepiece", reason="pip install -e '.[sentencepiece]'")
    shortest = build_shortest_tokenizer(expression[1], ["<s>"])
    processor = spm.SentencePieceProcessor(model_proto=to_sentencepiece(shortest))
    for text in list(source())[:50] + ROUNDTRIP_PROBES + ["<s>a<0x41>b"]:
        assert len(processor.encode(text)) == len(shortest.encode(text).ids)
        assert processor.decode(processor.encode(text)) == text


# ------------------------------------------- 10. HuggingFace model construction

_SPEC = {t: i for i, t in enumerate(BYTE_TOKENS + ["a", "b", "ab"])}


@pytest.mark.parametrize("vocab,merges,specials", [
    ({t: i for t, i in _SPEC.items() if t != "<0x00>"}, [("a", "b")], []),
    (dict(_SPEC, b=256), [("a", "b")], []),
    ({t: i for t, i in _SPEC.items() if t != "ab"}, [("a", "b")], []),
    (dict(_SPEC, **{"<0xab>": 259}), [("a", "b")], []),
    (_SPEC, [("a", "b", "c")], []),
    (_SPEC, [("a", 1)], []),
    (_SPEC, [("a", "b")], ["<0xff>"]),
    (dict(_SPEC, **{"<0x+A>": 259}), [("a", "b")], []),
    (_SPEC, [("a", "b"), ("a", "b")], []),
    (dict(_SPEC, ab=300), [("a", "b")], []),
], ids=["missing byte", "duplicate id", "missing output", "byte spelling", "triple",
        "non-str", "byte special", "signed byte spelling", "duplicate merge", "id gap"])
def test_build_rejects_specs_that_would_corrupt_the_tokenizer(vocab, merges, specials):
    with pytest.raises(InvalidBpeSpec):
        build_bpe_tokenizer(vocab, merges, specials)


def test_special_tokens_never_take_an_existing_id():
    vocab = {t: i for i, t in enumerate(BYTE_TOKENS)}
    vocab.update({"a": 257, "b": 258})
    tokenizer = build_bpe_tokenizer(vocab, [], ["<pad>", "<pad>"], require_contiguous_ids=False)
    assert tokenizer.token_to_id("<pad>") == 259
    assert tokenizer.decode(tokenizer.encode("b").ids) == "b"


def test_build_accepts_any_iterable_of_merges():
    tokenizer = build_bpe_tokenizer(_SPEC, iter([["a", "b"]]), [])
    assert tokenizer.encode("ab").tokens == ["ab"]


def test_sentencepiece_export_parks_tokens_it_cannot_spell():
    """U+2581 is SentencePiece's space and NUL is not allowed in a piece."""
    spm = pytest.importorskip("sentencepiece", reason="pip install -e '.[sentencepiece]'")
    tokenizer = train_tokenizer(["ab\x00ab cd ef\u2581gh"] * 60, vocab_size=320, min_frequency=1)
    parked = [t for t in tokenizer.get_vocab() if "\x00" in t or "\u2581" in t]
    processor = spm.SentencePieceProcessor(model_proto=to_sentencepiece(tokenizer))
    assert {"\x00", "\u2581"} <= set("".join(parked))
    assert all(processor.id_to_piece(tokenizer.token_to_id(t)).startswith("<unused")
               for t in parked)
    expected = [t.replace(" ", "\u2581") for t in tokenizer.encode("ab cd abab").tokens]
    assert processor.encode("ab cd abab", out_type=str) == expected


@pytest.mark.parametrize("change", ["normalizer", "lstrip", "overlap"])
def test_sentencepiece_export_rejects_what_it_cannot_reproduce(change):
    pytest.importorskip("sentencepiece", reason="pip install -e '.[sentencepiece]'")
    tokenizer = train_tokenizer(["the cat sat on the mat"] * 50, special_tokens=["<s>"],
                                vocab_size=300, min_frequency=1)
    if change == "normalizer":
        tokenizer.normalizer = S.hamza_normalizer()
    elif change == "lstrip":
        tokenizer.add_tokens([AddedToken("cat", lstrip=True)])
    else:
        tokenizer.add_tokens(["<s>x"])
    with pytest.raises(ValueError):
        to_sentencepiece(tokenizer)


# ------------------------------------------------------------- input checks

@pytest.mark.parametrize("doc", [b"bytes", {"text": "x", "language": "en"}, ("a", "en", 3), 7,
                                 ("a", 5)])
@pytest.mark.parametrize("filtering", [False, True])
def test_documents_must_be_text_or_text_language_pairs(doc, filtering):
    with pytest.raises(TypeError, match="document"):
        SupraTokTrainer(vocab_size=400, use_entropy_filtering=filtering).train(["alpha beta", doc])


@pytest.mark.parametrize("kwargs", [{"suffix_weight": -1}, {"suffix_weight": 1.5},
                                    {"max_span": 1}, {"max_seed_ngram": 200},
                                    {"vocab_size": 270}, {"min_frequency": -1},
                                    {"min_frequency": 2.5}, {"min_frequency": None},
                                    {"min_frequency": True}, {"seed_min_frequency": None},
                                    {"screening_frequency": -1}, {"screening_pmi": None},
                                    {"screening_pmi": float("nan")},
                                    {"seed_pmi_threshold": "2"}])
def test_parameters_are_checked_before_reading(kwargs):
    with pytest.raises((TypeError, ValueError)):
        SupraTokTrainer(**kwargs)


def test_an_empty_corpus_is_an_error():
    with pytest.raises(ValueError, match="no text"):
        SupraTokTrainer(vocab_size=400).train([])


def test_shortfall_warning_counts_the_special_tokens(caplog):
    trainer = SupraTokTrainer(vocab_size=900, special_tokens=["<pad>"], min_frequency=2,
                              enable_expression=False)
    vocab, _ = trainer.train(["alpha beta gamma"] * 20)
    assert f"of {trainer.vocab_size}" in caplog.text
    assert f"reached {len(vocab) + 1} " in caplog.text


def test_training_releases_the_corpus_it_read(source):
    trainer = SupraTokTrainer(vocab_size=VOCAB_SIZE, min_frequency=2)
    trainer.train(source)
    assert len(trainer.ids) == 0 and trainer.heldout == [] and trainer.verification == []
    assert trainer.seeds and trainer.alphabet and trainer.stats["vocab"] == VOCAB_SIZE


def test_builders_reject_a_bare_string_for_special_tokens():
    """A string is a sequence of one-character special tokens, which would make every
    '<', 'p' or 'a' special."""
    vocab = {t: i for i, t in enumerate(BYTE_TOKENS + [" ", "a"] + sorted("<>x0123456789ABCDEF"))}
    with pytest.raises(TypeError):
        build_bpe_tokenizer(vocab, [], "<pad>")
    with pytest.raises(TypeError):
        build_shortest_tokenizer(vocab, "<pad>")


def test_merge_order_rejects_a_token_no_merge_forms():
    """'ab' is in the vocabulary but no merge forms it: merge order never emits it, and a
    SentencePiece export would give it the best score."""
    vocab = {t: i for i, t in enumerate(BYTE_TOKENS + ["a", "b", "c", "ab", "bc"])}
    with pytest.raises(InvalidBpeSpec, match="no merge"):
        build_bpe_tokenizer(vocab, [("b", "c")], [])


def test_supercategories_follow_the_pretokenizer_unicode_tables():
    """classify() reads the same Unicode tables as the trainer's pre-tokenizer, whatever
    Python's own unicodedata version, so the gate never disagrees with the pre-tokens."""
    def expected(ch):
        for name, pattern in (("LM", r"[\p{L}\p{M}]"), ("PS", r"[\p{P}\p{S}]"),
                              ("N", r"\p{N}"), ("Z", r"[\p{Z}\t\n\v\f\r\x85]")):
            if regex.fullmatch(pattern, ch):
                return name
        return "C"

    for cp in (0x41, 0x558, 0x5C8, 0x3002, 0x31350, 0x20, 0x0A, 0x7F, 0xE000, 0x1F600):
        assert S.classify(chr(cp)) == expected(chr(cp)), hex(cp)


# -------------------------------------------------- documented behavior, pinned

def test_score_combines_every_appendix_a_factor():
    """Eqs. 8-12 by hand, with every boost and penalty away from its neutral value."""
    def sig(k, x):
        return 1.0 / (1.0 + math.exp(-k * x))

    kw = dict(BASE, avg_idf=3.0, internal_npmi=0.7, context_peak=0.2, n_words=4,
              avg_word_len=7.0, stopword_ratio=0.25)
    expected = (3 * (10 + 0.60 * 5 + 0.18 * math.sqrt(100))   # g_hold
                * sig(6.0, 0.8 - 0.22) * sig(1.75, 2.0 - 1.15)  # g_N, g_B
                * (1 + 0.14 * 2.0) * (1 + 0.50 * 0.5)           # b_spec, b_con
                * max(0.30, 1 - 0.78 * 0.25 ** 1.35)            # p_stop
                * (1 + 0.08 * 2) * (1 + 0.04 * 2.0))            # b_len, b_cont
    assert A.score(**kw) == pytest.approx(expected)


def test_addition_gain_charges_at_least_one_token():
    """Eq. 13: A(s) = c_holdout * max(1, l_tok - 1) + 0.35 df_holdout."""
    assert A.addition_gain(5, 1, 2) == pytest.approx(5 + 0.35 * 2)
    assert A.addition_gain(5, 3, 2) == pytest.approx(10 + 0.35 * 2)


def test_screen_admits_a_bigram_at_exactly_the_frequency_floor():
    """Section 3.5: frequency >= 5, so a cohesive bigram seen exactly five times passes."""
    filler = _filler_words(400, 5)
    rng = random.Random(5)
    docs = [" ".join(rng.choice(filler) for _ in range(30)) for _ in range(60)]
    docs[:5] = [d + " zylphon quarnex " + d for d in docs[:5]]
    trainer = SupraTokTrainer(enable_seeds=False, enable_expression=False)
    trainer._read(lambda: iter(docs))
    assert (" zylphon", " quarnex") in trainer._mine().bigrams


def test_frequent_but_uncohesive_ngrams_are_not_seeds():
    """Eq. 2: a seed needs PMI >= 2.0 as well as frequency >= 100; words drawn at random
    co-occur at chance, PMI near 0."""
    rng = random.Random(7)
    docs = [" ".join(rng.choice(["alpha", "beta", "gamma", "delta"]) for _ in range(40))
            for _ in range(200)]
    trainer = SupraTokTrainer(enable_expression=False)
    trainer._read(lambda: iter(docs))
    assert trainer._mine().seeds == {}


def test_overlap_control_drops_a_candidate_inside_a_kept_seed():
    """With seed_overlap=1.0 a candidate is dropped only when every occurrence overlaps a
    kept seed: 'zyl phon quar' never occurs outside 'zyl phon quar nex'."""
    filler = _filler_words(400, 6)
    rng = random.Random(6)
    docs = [" ".join(rng.choice(filler) for _ in range(20)) + " zyl phon quar nex "
            + " ".join(rng.choice(filler) for _ in range(20)) for _ in range(150)]
    trainer = SupraTokTrainer(seed_overlap=1.0, enable_expression=False)
    trainer._read(lambda: iter(docs))
    seeds = trainer._mine().seeds
    assert (" zyl", " phon", " quar", " nex") in seeds
    assert (" zyl", " phon", " quar") not in seeds


def _scripted(monkeypatch, counts):
    """Replace the verification count with a script, one value per call."""
    it = iter(counts)
    monkeypatch.setattr(SupraTokTrainer, "_verification_tokens", lambda self, merges: next(it))
    monkeypatch.setattr(trainer_module, "EXPRESSION_CHUNK", 1)


def test_expression_accepts_while_every_chunk_helps(monkeypatch, source):
    _scripted(monkeypatch, range(10_000, 0, -1))
    trainer, _, _ = _train(source, enable_multiword=True, enable_expression=True)
    assert trainer.stats["pass3_accepted"] > 1


def test_expression_counts_a_tie_as_a_veto(monkeypatch, source):
    """EXPRESSION keeps a chunk only if verification tokens fall; unchanged is not enough."""
    _scripted(monkeypatch, [100] * 10_000)
    trainer, _, _ = _train(source, enable_multiword=True, enable_expression=True)
    assert trainer.stats["pass3_accepted"] == 0


def test_expression_stops_at_the_first_veto(monkeypatch, source):
    """EXPRESSION stops at the first chunk that does not lower the token count."""
    _scripted(monkeypatch, [100, 101] + list(range(99, 0, -1)) * 100)
    trainer, _, _ = _train(source, enable_multiword=True, enable_expression=True)
    assert trainer.stats["pass3_accepted"] == 0


def test_expression_measures_chunks_of_two_thousand_spans():
    assert trainer_module.EXPRESSION_CHUNK == 2000


def test_verification_counts_the_verification_shard():
    trainer = SupraTokTrainer()
    trainer._set_alphabet([(" ab", 1)])
    trainer.heldout, trainer.verification = ["a" * 10], ["b" * 3]
    assert trainer._verification_tokens([]) == 3


def test_held_out_and_verification_shards_are_equally_wide():
    shards = Counter(shard_of(f"document {i}", 0.1) for i in range(20_000))
    assert 1800 < shards["held-out"] < 2200 and 1800 < shards["verification"] < 2200


def test_leaf_surgery_never_displaces_a_seed(expression):
    trainer, vocab, _ = expression
    assert trainer.stats["surgery_retired"] > 0
    assert all("".join(seed) in vocab for seed in trainer.seeds)


def test_a_token_formed_by_two_merges_is_not_a_leaf():
    """Retiring one of two merges that form 'abc' would free no slot."""
    vocab = {t: i for i, t in enumerate(BYTE_TOKENS + ["a", "b", "c", "ab", "bc", "abc", "ca"])}
    merges = [("a", "b"), ("b", "c"), ("ab", "c"), ("a", "bc"), ("c", "a")]
    tk = build_bpe_tokenizer(vocab, merges, [])
    pool = SupraTokTrainer._leaves(merges, ["abc ca abc"], tk)
    assert [m for _, _, m in pool] == [("c", "a")]


def test_expression_takes_free_slots_before_displacing(monkeypatch):
    """When pass 1 runs out of candidates, T has free slots, and EXPRESSION fills them
    before it retires any leaf."""
    _scripted(monkeypatch, range(10_000, 0, -1))
    rng = random.Random(9)
    phrases = ["zyl phon quar", "bel mot dra", "kip sal vor"]
    filler = _filler_words(60, 9)
    docs = [" ".join(rng.choice(filler) for _ in range(8)) + f" {rng.choice(phrases)} "
            + " ".join(rng.choice(filler) for _ in range(8)) for _ in range(400)]
    tokenizer = train_tokenizer(docs, vocab_size=3000, min_frequency=2, enable_seeds=False,
                                enable_multiword=False)
    stats = tokenizer.supratok_stats
    assert stats["pass3_accepted"] > 0 and stats["surgery_retired"] == 0


def test_spans_with_three_stopwords_in_four_words_are_rejected():
    """Appendix A: a span whose stopword ratio reaches 0.75 is rejected outright."""
    filler = _filler_words(400, 4)
    rng = random.Random(4)
    docs = [" ".join(rng.choice(filler) for _ in range(15)) + " the of and zylphon "
            + " ".join(rng.choice(filler) for _ in range(15)) + " the quarnex vel zylphon "
            + " ".join(rng.choice(filler) for _ in range(15)) for _ in range(300)]
    trainer = SupraTokTrainer()
    trainer._read(lambda: iter(docs))
    spans = {s.words for s in trainer._mine().spans}
    assert (" the", " quarnex", " vel", " zylphon") in spans
    assert (" the", " of", " and", " zylphon") not in spans


def test_suffix_mining_keeps_two_hundred_candidates():
    """Appendix E: 200 heuristic suffix candidates per language."""
    rng = random.Random(3)
    stems = ["".join(rng.choice("bdfgklmnprstvz") + rng.choice("aeiou") for _ in range(3))
             for _ in range(60)]
    endings = sorted({"".join(rng.choice("bdfgklmnprstvz") + rng.choice("aeiou"))
                      + rng.choice("lmnr") for _ in range(2000)})[:300]
    words = set(stems) | {s + e for s in stems for e in endings}
    assert len(S.suffixes_from_types(words)) == 200


@pytest.mark.parametrize("spelling", ["<0xAB>end", "<0xab>"])
def test_no_merge_touches_a_byte_token_spelling(spelling):
    """A byte token is never a merge's input, and no merge output is spelled like one, in
    any case of its hex digits."""
    filler = _filler_words(300, 8)
    docs = [f"{spelling} {filler[i % 300]} zorblax quintor {filler[(i * 7) % 300]}"
            for i in range(400)]
    tokenizer = train_tokenizer(docs, vocab_size=1500, min_frequency=2)
    assert tokenizer.decode(tokenizer.encode(docs[0]).ids) == docs[0]


def test_multiword_tokens_reach_six_words():
    """MULTIWORD tokens reach max(max_seed_ngram, max_span) = 6 words and no more."""
    filler = _filler_words(400, 12)
    rng = random.Random(12)
    phrase = "zyl phon quar nex bel mot"
    docs = [" ".join(rng.choice(filler) for _ in range(20)) + f" {phrase} "
            + " ".join(rng.choice(filler) for _ in range(20)) for _ in range(400)]
    tokenizer = train_tokenizer(docs, vocab_size=1200, min_frequency=2, enable_seeds=False,
                                enable_expression=False)
    vocab = tokenizer.get_vocab()
    assert " " + phrase in vocab
    assert max(n_words(t) for t in vocab) <= 6


def test_hamza_folds_through_any_arabic_mark_before_it():
    """NFD also puts U+0670 (superscript alef) and other low-class marks before the hamza."""
    for text in ("أٰ", "إٖ", "ؤٰ"):
        decomposed = unicodedata.normalize("NFD", text)
        assert decomposed != text
        expected = unicodedata.normalize("NFD", S.normalize_hamza(text))
        assert S.normalize_hamza(decomposed) == expected
        assert S.hamza_normalizer().normalize_str(decomposed) == expected


def test_multiword_does_not_join_across_cjk_clause_punctuation():
    """Full-width and CJK punctuation end a clause whatever script follows it."""
    trainer = SupraTokTrainer(enable_expression=False)
    trainer._read(lambda: iter([("你好，Hello世界。谢谢。Python是好的，对，2023年。", "zh")] * 3))
    junctions = trainer._mine().junctions
    after = {trainer.types[trainer.ids[i]] for i in range(1, len(trainer.ids))
             if junctions[i] and trainer.types[trainer.ids[i - 1]] in ("，", "。")}
    assert after == set()
