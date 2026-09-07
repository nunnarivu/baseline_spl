'''
symbolic_sweep8.py

A longer B3-a run over eight concepts spanning the measured difficulty range, to answer two
questions the smoke runs could not:

  1. which concepts does search actually learn, given a real budget?
  2. where does the time go?

The eight are chosen to bracket the interesting boundary rather than to flatter the method:

  MDL ~12-15   row, column, tower, inverted_row, diagonal_45, diagonal_135
               solved in seconds even at a smoke budget
  MDL ~23 (library form), ~35 (inlined)   staircase, inverted_staircase
               out of reach without the library; the whole point of the loop is whether
               compressing the lines brings them within reach

Nothing at MDL 50+ (pyramid, x, arch_bridge, isosceles) is included: those are unreachable by
enumeration at any budget this machine can spend, so including them would only pad the run.
They belong in the full sweep, reported as unsolved.

    BASELINE_CONFIG=symbolic_sweep8 python -m baseline_spl.symbolic.dreamcoder.run

Results land in runs/symbolic_sweep8_dreamcoder_standard/, with per-stage timings in
search_stats.json under `seconds_by_stage`.
'''

from __future__ import annotations

from baseline_spl.configs.default import CODEGEN_MODEL, VLM_MODEL  # noqa: F401
from baseline_spl.configs.default import CapConfig as _CapConfig
from baseline_spl.configs.default import CommonConfig as _CommonConfig
from baseline_spl.configs.default import Demo2CodeConfig as _Demo2CodeConfig
from baseline_spl.configs.default import DreamCoderConfig as _DreamCoderConfig
from baseline_spl.configs.default import SayCanConfig as _SayCanConfig


class CommonConfig(_CommonConfig):
    concepts = ["row", "column", "tower", "inverted_row",
                "diagonal_45", "diagonal_135", "staircase", "inverted_staircase"]
    inference = False          # learning is what this run measures


class CapConfig(_CapConfig, CommonConfig):
    pass


class Demo2CodeConfig(_Demo2CodeConfig, CommonConfig):
    pass


class SayCanConfig(_SayCanConfig, CommonConfig):
    pass


class DreamCoderConfig(_DreamCoderConfig, CommonConfig):
    # 20 minutes per iteration. Enumeration cost grows as ~e^(0.79 x MDL), so this buys
    # description length only logarithmically -- the point is not to brute-force staircase but
    # to see whether the library moves it within reach of a budget like this one.
    enumeration_timeout = 1200.0
    search_iterations = 4

    # One worker per task. Parallelism is across tasks, so it engages from iteration 1 onward,
    # once the recognition model gives each task its own grammar.
    cpus = 8
