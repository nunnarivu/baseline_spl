'''Run configurations. One file per experiment; select with BASELINE_CONFIG.

Naming
------
`default`               the template every other file inherits from; copy it to start a new one
`*_smoke`               a few concepts and a short budget, for checking plumbing in minutes
`symbolic_sweep8`       eight concepts spanning the measured difficulty range
`determinism_pin`       a reproducible reference run, for verifying a refactor changed nothing
`r2_*`                  Round 2: continuous observations judged by SPL's own noise model,
                        rather than the integer lattice Round 1 handed the search
   `..._full16`         all 16 concepts, one task per concept (a program must work for every size)
   `..._demo16`         all 16 concepts, one task per demonstration (DreamCoder as published:
                        the program is closed and the size appears as a literal)
   `..._lilo16`         the same, run through B3-b
   `..._norecog`        that run with the recognition model switched off -- the ablation
   `..._rfix`           re-run after the recognition model was fixed; the earlier results are
                        kept because their solved counts stand, but their recognition
                        comparison does not
`r2_infer8`             inference only, scored against a library an earlier run left behind

Which settings must stay identical across baselines is machinery, not a knob: see
PARITY_CRITICAL in common/config.py.
'''
