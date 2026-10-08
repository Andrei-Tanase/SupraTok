# Implementation notes

How the implementation fills in details the paper leaves open. The paper's stated settings
are the defaults.

## Choices

1. **MULTIWORD screens word junctions.** The §3.5 screen (PMI ≥ 5.0, frequency ≥ 5) applies
   to each junction between two words, and MULTIWORD resumes BPE across junctions that pass,
   so tokens can span up to six words. Each merge joins the pieces on either side of a
   junction (` New` + ` York`, `ing` + ` to`).
2. **Seeds feed MULTIWORD.** Junctions inside occurrences of the mined seeds pass the screen
   too, so MULTIWORD can extend a seed (§3.3, "PMI-seeded MULTIWORD candidates").
3. **A word joins what is attached to it.** Appendix A's unit is the whitespace-delimited
   word, so ` York,` or ` don't` is one word: MULTIWORD joins a word to the punctuation or
   contraction after it without the screen.
4. **EXPRESSION is checked per chunk.** Spans are accepted by Appendix A's rule Net(s) > 0
   in Priority order, in chunks of 2,000; a chunk is kept only if it lowers the token count
   of a verification shard, and the pass stops at the first that does not.
5. **Held-out shards are drawn by hash** of each document's text, so they do not depend on
   document order.
6. **Byte-token spellings stay text.** `<`, `>`, `x`, `0`–`9` and `A`–`F` are always in the
   alphabet, and no merge forms a token spelled like a byte token.
7. **Shortest-path encoding** is optional (`segmentation="shortest"`).
8. **Rare characters use byte fallback.** A character enters the alphabet when it makes up at
   least one in 5 million training characters and occurs `min_frequency` times.
9. **A punctuation run keeps the line breaks after it** (`.\n\n` is one pre-token).
10. **The bigram gate binds MULTIWORD.** Pieces the gate splits are not rejoined.
11. **Unspaced punctuation ends a clause.** In Chinese, Japanese and Thai, a junction after
    CJK punctuation, or from punctuation to a letter of those scripts, is a word boundary.
12. **An apostrophe inside a word stays in it** (`l'homme`, `dell'anno`, `Ankara'da`);
    English contractions (`don` + `'t`) are split.
13. **A no-break space leads a word** as a space does (U+00A0, U+202F).

## Details

- **Pass sizes.** Figure 1 sets no quotas, so one BPE run over all candidates sizes the
  passes: merges inside one pre-token go to TRADITIONAL, merges joining two pre-tokens
  without a space go to MULTIWORD, and merges across a space are the cross-word share, for
  the seeds and then MULTIWORD. Slots MULTIWORD leaves unused go back to TRADITIONAL.
- **Seeds** count toward |T| from the start, are taken in order of the text they cover,
  count × (n − 1), and are skipped when their merges no longer fit. A candidate overlapping
  a kept seed in at least `seed_overlap` of its occurrences is dropped.
- **Words** are pre-tokens other than whitespace, with the space before them; phrases are
  runs of words joined by single spaces, so a line break or a double space ends a span.
- **Held-out shards** supply Appendix A's held-out counts and U(m), and are left out of its
  training counts; the verification shard checks EXPRESSION.
- **Leaf surgery** measures U(m) once, uses free slots before retiring a leaf, and never
  displaces a seed token.
- **Branching entropy** follows Appendix A's gate g_B. **idf** (Eq. 7) uses the natural log.
- **Suffix-aware weighting** counts the ending of each `tr` or `fi` word that matches one of
  200 suffixes mined from the corpus `suffix_weight` more times as a unit.
- **Hamza normalization** applies to training text and, through the normalizer, to input;
  such a tokenizer cannot be exported to SentencePiece.
- **Training** does not depend on document order, and the trainer releases the corpus it
  read when it finishes.
