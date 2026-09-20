'''
rescore_run.py

Re-derive every term from the program a finished run already recorded, then lower, register
and score again -- into a NEW directory. The source run is only ever read.

Why this exists. `Solution.term` used to be a stored field written under an `if term is None`
guard, while `Solution.program` was derived from a mutable frontier. An LLM proposal accepted
in a later iteration replaced the program and left the earlier program's term in place, so 28
of 99 concepts in `lilo_gptluna_concept_level_depthimages_run1` were lowered from a term
belonging to a program the run had already discarded. All 28 scored `program_accuracy` 0.0
with the correct program sitting in the record beside them.

Nothing expensive has to be repeated. The wake/sleep loop reads `solution.programs`, never
`solution.term` -- `_corpus` (driver.py) feeds STITCH and the re-weighting from the lambda
programs alone -- so the search, the library and the LLM solves were never affected. Only the
final cheap stage (term -> lower -> register -> score) consumed the bad input, and every
program it needs is in `search_programs.yml`.

**No enumeration and no LLM call happens here.** The programs are inputs.

Run:
    BASELINE_CONFIG=lilo_concept python -m baseline_spl.tests.tools.rescore_run \\
        --source baseline_spl/runs/lilo_gptluna_concept_level_depthimages_run1 \\
        --into   lilo_run0_corrected
'''

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import OrderedDict

import yaml

# Artifacts the search produced that this script does not recompute, copied across verbatim.
CARRY_OVER = ("search_stats.json", "lilo_stats.json", "library.json", "demo_selection.json")
# Per-concept fields that come from the LLM ledger, not from scoring: `_annotate_record`
# merges them during a real run, and there is no proposer here to merge them again.
PRESERVE_FIELDS = ("prompt_tokens", "completion_tokens", "num_llm_calls", "num_cache_hits")


def load_records(source: str) -> dict:
    path = os.path.join(source, "search_programs.yml")
    if not os.path.exists(path):
        raise SystemExit(f"No search_programs.yml in {source}")
    return yaml.safe_load(open(path, encoding="utf-8")) or {}


def rebuild_solutions(records: dict, tasks, log) -> "object":
    '''A RunResult whose solutions hold the recorded programs -- and therefore, now that
    `term` derives itself, the right terms.'''
    from dreamcoder.program import Program

    from baseline_spl.symbolic import driver
    from baseline_spl.symbolic.search import Solution

    arity_of = {t.name: t.arity for t in tasks}
    result = driver.RunResult()
    untranslatable = []
    for concept, record in records.items():
        source = record.get("program")
        if not source:
            continue
        try:
            program = Program.parse(source)
        except Exception as exc:  # noqa: BLE001
            log(f"  {concept}: recorded program will not parse ({exc}); skipping")
            continue

        solution = Solution(task=concept, arity=arity_of.get(concept, 0))
        prior = -float(record.get("mdl") or 0.0)
        if record.get("status") == "solved":
            solution.frontier = [(prior, program)]
            solution.distance = 0.0
        else:
            # Keep a near miss a near miss: it is lowered and recorded the same way the run
            # did, so the corrected numbers stay comparable column for column.
            solution._approximate = (prior, program)
            solution.distance = float(record.get("search_score") or 1.0)
        if solution.term is None:
            untranslatable.append(concept)
        result.solutions[concept] = solution
        if record.get("solved_by"):
            result.solved_by[concept] = record["solved_by"]

    if untranslatable:
        log(f"  {len(untranslatable)} recorded program(s) will not translate: "
            f"{', '.join(untranslatable)}")
    return result


def merge_preserved(source: str, target: str) -> None:
    '''Carry the LLM cost columns over, so the corrected metrics lose nothing the run had.'''
    src_path = os.path.join(source, "training_metrics.json")
    dst_path = os.path.join(target, "training_metrics.json")
    if not (os.path.exists(src_path) and os.path.exists(dst_path)):
        return
    old = json.load(open(src_path, encoding="utf-8"))
    new = json.load(open(dst_path, encoding="utf-8"))
    for concept, record in new.items():
        previous = old.get(concept) or {}
        for field in PRESERVE_FIELDS:
            if field in previous and field not in record:
                record[field] = previous[field]
    json.dump(new, open(dst_path, "w", encoding="utf-8"), indent=2)


def write_correction_note(source: str, target: str) -> None:
    old_path = os.path.join(source, "training_metrics.json")
    new_path = os.path.join(target, "training_metrics.json")
    old = json.load(open(old_path, encoding="utf-8")) if os.path.exists(old_path) else {}
    new = json.load(open(new_path, encoding="utf-8")) if os.path.exists(new_path) else {}

    def accuracy(table, concept):
        return (table.get(concept) or {}).get("program_accuracy")

    # Only what this pass actually re-scored, so a partial run reports on its own concepts
    # rather than counting the ones it never touched as having lost their score.
    concepts = sorted(set(new))
    changed = [(c, accuracy(old, c), accuracy(new, c))
               for c in concepts if accuracy(old, c) != accuracy(new, c)]
    before = sum(1 for c in concepts if accuracy(old, c) == 1.0)
    after = sum(1 for c in concepts if accuracy(new, c) == 1.0)

    lines = [
        "# Correction of a stale-term defect",
        "",
        f"Source run (unmodified): `{source}`",
        "",
        "## What was wrong",
        "",
        "`Solution.term` was a stored field written under an `if term is None` guard, while",
        "`Solution.program` was derived from a mutable frontier. When an LLM proposal was",
        "accepted in a later iteration it replaced the program, but the guard refused to",
        "refresh the term, so the concept was lowered, registered and scored from a term",
        "belonging to a program the run had already discarded.",
        "",
        "## What was redone",
        "",
        "Only the post-search stage: each term was re-derived from the program the source run",
        "already recorded in `search_programs.yml`, then lowered, registered and scored",
        "through the same `_register_and_score` path a real run uses.",
        "",
        "**No enumeration and no LLM call was repeated.** The search, the STITCH library and",
        "the grammar weights were never affected -- the wake/sleep loop reads",
        "`solution.programs`, never `solution.term`. `search_stats.json`, `lilo_stats.json`",
        "and `library.json` are copied from the source run byte for byte.",
        "",
        "## Result",
        "",
        f"- concepts at `program_accuracy == 1.0` before: **{before}**",
        f"- concepts at `program_accuracy == 1.0` after:  **{after}**",
        f"- concepts whose score changed: **{len(changed)}**",
        "",
        "| concept | before | after |",
        "|---|---|---|",
    ]
    lines += [f"| {c} | {b} | {a} |" for c, b, a in changed]
    lines.append("")
    open(os.path.join(target, "CORRECTION.md"), "w", encoding="utf-8").write("\n".join(lines))
    print(f"\nprogram_accuracy == 1.0:  before {before}   after {after}   "
          f"({len(changed)} concept(s) changed)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="finished run directory (read only)")
    parser.add_argument("--into", required=True, help="new run NAME under baseline_spl/runs/")
    parser.add_argument("--only", nargs="*", default=None,
                        help="re-score just these concepts (for a quick check)")
    arguments = parser.parse_args()

    from baseline_spl.common.config import BaselineConfig
    from baseline_spl.common.harness import log
    from baseline_spl.config import LiloConfig
    from baseline_spl.symbolic.bridge import grammar as build_grammar
    from baseline_spl.symbolic.dreamcoder.run import share_sketch_cache
    from baseline_spl.symbolic.harness import SearchHarness
    from baseline_spl.symbolic.tasks import build_tasks

    source = os.path.abspath(arguments.source)
    records = load_records(source)

    configs = BaselineConfig.from_run_config(LiloConfig, arguments.into)
    if os.path.abspath(configs.run_dir) == source:
        raise SystemExit("Refusing to write into the source run directory.")
    share_sketch_cache(configs, LiloConfig)

    # B3-a's harness: it owns `_register_and_score`, and no proposer is needed to re-score.
    # The LLM cost columns are merged back from the source run afterwards.
    harness = SearchHarness(configs)
    cfg = harness.configs

    concepts = [c for c in (cfg.concepts_to_learn or cfg.concepts_space) if c in records]
    if arguments.only:
        concepts = [c for c in concepts if c in set(arguments.only)]
    log(f"Re-scoring {len(concepts)} concept(s) recorded in {source}")
    demos_by_concept = harness._load_demos(concepts, summarise=True)
    pending = OrderedDict((c, d) for c, d in demos_by_concept.items() if d)
    sketches = {c: [d.get("sketch_info") for d in demos] for c, demos in pending.items()}

    tasks, used_demos, skipped = build_tasks(
        pending, sketches,
        granularity=getattr(cfg, "task_granularity", "concept"),
        selection=getattr(cfg, "demo_selection", "distinct_params"),
        demos_per_concept=cfg.num_demos_per_concept,
        observation_mode=harness.observation_mode,
        executor=harness.spl.executor if harness.observation_mode == "continuous" else None)

    # Score the demos the source run scored, not a fresh selection, or the two runs' numbers
    # would differ for a second reason on top of the one being corrected.
    recorded_selection = os.path.join(source, "demo_selection.json")
    if os.path.exists(recorded_selection):
        used_demos = json.load(open(recorded_selection, encoding="utf-8"))
        log("Reusing the source run's demo selection.")

    for concept, reason in skipped.items():
        log(f"<{concept}>: not expressible for this baseline -- {reason}")
        pending.pop(concept, None)

    # Before any `Program.parse`. The recorded programs carry integer literals that only
    # exist as primitives once the grammar has been built to that ceiling -- without this,
    # `smiley`'s `(move RIGHT 3 ...)` raises ParseFailure and the concept is recorded as
    # "no program found" despite its program sitting right there in the file.
    ceiling = harness._literal_ceiling(tasks, cfg)
    build_grammar(getattr(cfg, "grammar_level", "standard"), int_literals_upto=ceiling)
    log(f"Grammar rebuilt with integer literals up to {ceiling} so the recorded programs parse.")

    result = rebuild_solutions(records, tasks, log)
    harness._register_and_score(pending, used_demos, result, tasks, cfg)
    if cfg.concept_save_path:
        harness.spl.save(cfg.concept_save_path)

    for name in CARRY_OVER:
        origin = os.path.join(source, name)
        if os.path.exists(origin):
            shutil.copy2(origin, os.path.join(cfg.run_dir, name))

    merge_preserved(source, cfg.run_dir)
    write_correction_note(source, cfg.run_dir)
    print(f"Corrected run written to {cfg.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
