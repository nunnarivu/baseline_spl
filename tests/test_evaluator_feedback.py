'''
test_evaluator_feedback.py

use_evaluator_feedback adds execution feedback to codegen's retry loop. With a stub backend
and a stub evaluator (no network, no SPL), check that:

  1. a failing class is retried with the evaluator's report, and a passing one is returned;
  2. a bookkeeping-only failure gets the bookkeeping prompt;
  3. when nothing passes, the best-scoring class comes back within max_retries + 1 calls;
  4. with no evaluator, the loop is unchanged: the first valid class, one call;
  5. the harness refuses the knob with sketch_mode='none' or use_demo=False.

The evaluator itself (common/evaluator.py) needs SPL and real demonstrations; it was
checked against SPL's learned library, not here.

Run: python -m baseline_spl.tests.test_evaluator_feedback
'''

from __future__ import annotations

import sys
import types

from baseline_spl.common import codegen

CLASS = '''```python
class zigzag:
    # attempt {n}
    def __init__(self, length: int, objects: list):
        self.length, self.objects = length, objects
    def construct(self):
        pass
    @staticmethod
    def argument_sampler():
        yield (1, None)
    @property
    def plan(self):
        return []
    @property
    def blocks(self):
        return []
    @property
    def substructures(self):
        return []
    @property
    def key_blocks(self):
        return []
    @property
    def actions(self):
        return []
```'''


class StubConversation:
    def __init__(self, backend):
        self.backend = backend

    def ask(self, prompt, max_tokens=None):
        self.backend.prompts.append(prompt)
        return CLASS.format(n=len(self.backend.prompts))


class StubBackend:
    def __init__(self):
        self.prompts = []

    def start_conversation(self, system, images=None):
        return StubConversation(self)


class StubEvaluator:
    '''Replays (passed, score, tag) per call and keeps the best code, as ClassEvaluator does.'''

    def __init__(self, results):
        self.results = list(results)
        self.best_code, self.best_score = None, float("-inf")

    def __call__(self, code):
        passed, score, tag = self.results.pop(0)
        if self.best_code is None or score > self.best_score:
            self.best_code, self.best_score = code, score
        return passed, score, f"{tag}:\nDemonstration 1: blocks NOT close: block 3."


def _run(evaluator, max_retries=3):
    backend = StubBackend()
    code = codegen.generate_with_retries(backend, "system", "task", wanted_name="zigzag",
                                         max_retries=max_retries, evaluator=evaluator,
                                         log=lambda *_: None)
    return code, backend.prompts


def check_fail_then_pass(failures):
    code, prompts = _run(StubEvaluator([(False, -2.0, "FAILED"), (True, -0.1, "OK")]))
    if len(prompts) != 2 or "# attempt 2" not in (code or ""):
        failures.append(f"fail->pass: expected attempt 2 after 2 calls, got {len(prompts)} calls")
    elif "did not reproduce them" not in prompts[1] or "block 3" not in prompts[1]:
        failures.append("fail->pass: the retry prompt does not carry the evaluator's report")
    else:
        print("  OK   a failing class is retried with the report; the passing one is returned")


def check_bookkeeping_prompt(failures):
    _code, prompts = _run(StubEvaluator([(False, -0.1, "BOOKKEEPING"), (True, -0.1, "OK")]))
    if "bookkeeping is wrong" not in prompts[1]:
        failures.append("a bookkeeping-only failure did not get the bookkeeping prompt")
    else:
        print("  OK   a bookkeeping-only failure gets the bookkeeping prompt")


def check_best_when_nothing_passes(failures):
    scores = [-3.0, -1.0, -2.0, -5.0]
    code, prompts = _run(StubEvaluator([(False, s, "FAILED") for s in scores]), max_retries=3)
    if len(prompts) != 4:
        failures.append(f"never-pass: expected 4 calls (max_retries + 1), got {len(prompts)}")
    elif "# attempt 2" not in (code or ""):
        failures.append("never-pass: did not return the best-scoring class (attempt 2)")
    else:
        print("  OK   nothing passes: the best-scoring class is returned within the budget")


def check_no_evaluator_unchanged(failures):
    code, prompts = _run(None)
    if len(prompts) != 1 or "# attempt 1" not in (code or ""):
        failures.append(f"no evaluator: expected the first class after 1 call, got {len(prompts)}")
    else:
        print("  OK   without an evaluator the first valid class is returned after one call")


def check_harness_guards(failures):
    from baseline_spl.common.harness import BaselineHarness
    from baseline_spl.VLM.saycan.agent import SayCanAgent

    agent = types.SimpleNamespace(generate=lambda *a: None)
    for label, overrides in (("sketch_mode='none'", {"sketch_mode": "none"}),
                             ("use_demo=False", {"use_demo": False})):
        configs = types.SimpleNamespace(load_concept_checkpoint=None, use_evaluator_feedback=True,
                                        sketch_mode="corrected", use_demo=True)
        vars(configs).update(overrides)
        try:
            BaselineHarness(configs, agent)
            failures.append(f"the harness accepted use_evaluator_feedback with {label}")
        except ValueError:
            print(f"  OK   use_evaluator_feedback with {label} is refused")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{label}: expected ValueError before SPL is built, got {type(exc).__name__}: {exc}")
    # SayCan is exempt because it generates no class; the guard keys on that.
    if hasattr(SayCanAgent, "generate"):
        failures.append("SayCanAgent has generate(), so the guard would refuse SayCan runs")
    else:
        print("  OK   SayCan generates no class, so the guard does not apply to it")


def main() -> int:
    failures = []
    check_fail_then_pass(failures)
    check_bookkeeping_prompt(failures)
    check_best_when_nothing_passes(failures)
    check_no_evaluator_unchanged(failures)
    check_harness_guards(failures)

    print()
    if failures:
        for f in failures:
            print(f"  FAIL {f}")
        return 1
    print("all evaluator-feedback checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
