'''Symbolic (search-based) baselines: DreamCoder (B3-a) and LILO (B3-b).

See the plan for the design. In short: search runs over a restricted typed IR
(`ir.py`, semantics in `lattice.py`), and the winning term is printed deterministically into
the Python concept class SPL's metrics score (`lower.py`).
'''
