'''Search-based baselines: DreamCoder (B3-a) and LILO (B3-b).

Both answer a question the LLM baselines cannot: not "can a model write this program?" but
"can *search* find it, and does a learned library help?"

The shape of it
---------------
A concept becomes a synthesis task whose examples are the placements a demonstration made
(`tasks.py`). A typed enumerator proposes programs in order of decreasing prior and keeps the
ones that reproduce those placements (`search.py`, judged by `evaluate.py`). Solutions are then
compressed into reusable abstractions which re-enter the grammar, so a concept out of reach in
one round may be in reach in the next (`stitch_bridge.py`); a neural model conditions the
grammar per task (`recognition.py`). `driver.py` is the loop over those phases, and
`harness.py` connects it to SPL's metrics.

What is searched is NOT what is emitted. The search explores a small typed combinator IR
(`ir.py`, semantics in `lattice.py`) where Hindley-Milner typing prunes hard; the winning term
is then *printed* deterministically into the Python concept class SPL scores (`lower.py`).
Searching Python source directly is what the LLM baselines do, and the contrast is deliberate.

B3-b is B3-a plus two hooks -- an LLM proposer and library auto-documentation (`lilo/`). Passing
neither reproduces B3-a exactly, which is what makes "they differ by exactly LILO's
contributions" a checkable claim rather than a hope.

Reading order is listed in the project README under Layout; each module depends only on the
ones above it there.
'''
