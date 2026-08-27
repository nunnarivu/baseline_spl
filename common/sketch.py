'''
sketch.py

Thin wrapper over SPL's ``SketchAgent``, shared by every method.

All methods — SPL and each baseline — parse instructions with the same agent and the
same model, so a baseline failure is a *learning* failure rather than a parsing one,
and the concept name a baseline registers is the name a test instruction resolves to.
The agent is the one ``SPL.__init__`` already built; because ``BaselineConfig`` gave
it a per-variant ``cache_dir``, no cache state is shared with SPL.

The initialized-sketch string produced here is injected into each baseline's
code-generation prompt, mirroring what SPL's Generalize stage receives — otherwise SPL
would enjoy an unearned advantage in knowing the target signature, and the baselines'
``program_accuracy`` would land on "undecided" (see check_program_equivalence, which
requires matching integer-argument arity).
'''

from __future__ import annotations

from typing import Any, Dict, Optional


class SharedSketch:
    '''Input: spl: SPL - a constructed SPL instance whose sketch_agent we borrow.'''

    def __init__(self, spl):
        self.spl = spl
        self.agent = spl.sketch_agent

    def signature(self, instruction: str) -> Dict[str, Any]:
        '''Instruction -> SPL's sketch_info schema:
        ``{'concept': str, 'arguments': {name: {'type': str, 'value': Any}}}``.'''
        return self.agent.extract_function_signature(instruction)

    @staticmethod
    def values(sketch_info: Dict[str, Any]) -> Dict[str, Any]:
        '''The argument name -> value mapping used to instantiate the concept.'''
        return {name: spec['value'] for name, spec in sketch_info['arguments'].items()}

    @staticmethod
    def attributes(sketch_info: Dict[str, Any]) -> Dict[str, type]:
        '''The argument name -> python type mapping ``register_inductive_concepts``
        expects. Types arrive from the LLM as strings ('int', 'list'), which is what
        SPL.learn itself evals.'''
        return {name: eval(spec['type']) for name, spec in sketch_info['arguments'].items()}

    def corrected(self, sketch_infos, demos):
        '''SPL's own cross-demonstration correction: make the sketches agree on one
        concept name and argument schema, renaming when a name collides with a library
        concept that behaves differently. Returns them unchanged, with no LLM call, when
        they already agree.'''
        return self.spl.validate_and_correct_sketch(sketch_infos, demos)

    def ground(self, instruction: str, concept: str, attributes: Dict[str, type],
               retries: int = 2, log=print) -> Optional[Dict[str, Any]]:
        '''Sketch an instruction onto an already-registered concept.

        Used by the no-sketch condition: the class is registered first, so the sketch
        agent sees its name and argument names in context (``_similar_concepts`` scores
        against exactly those) and should ground the instruction onto it.

        The cache is keyed by instruction alone, so an entry written against a previous
        run's class name is stale under sketch_mode="none", where the name changes on
        every relearn. A mismatch drops that entry and bypasses the cache for the retry.

        Returns None (with a warning) if it never grounds, so the caller can carry on with
        the demonstrations that did.
        '''
        expected = set(attributes)
        original_use_cache = self.agent.use_cache
        problem = "no attempt made"
        try:
            for attempt in range(retries + 1):
                try:
                    info = self.agent.extract_function_signature(instruction)
                except Exception as exc:  # noqa: BLE001
                    problem = f"sketch raised: {exc}"
                else:
                    got = set(info.get("arguments", {}))
                    if info.get("concept") == concept and got == expected:
                        return info
                    problem = (f"sketch produced {info.get('concept')}({sorted(got)}), "
                               f"expected {concept}({sorted(expected)})")
                self._drop_stale(instruction, log)
                if attempt < retries:
                    log(f"[sketch] {problem}; re-asking ({attempt + 1}/{retries})")
                    self.agent.use_cache = False   # otherwise the retry replays the cache
        finally:
            self.agent.use_cache = original_use_cache

        log(f"[sketch] WARNING could not ground '{instruction}' onto {concept}: {problem}. "
            f"Skipping this demonstration's metrics.")
        return None

    def _drop_stale(self, instruction: str, log=print) -> None:
        '''Remove a cached sketch that failed to ground. Done on the last attempt too:
        the agent re-caches what it just produced, and a wrong answer left behind would
        be served to the next run without ever reaching the LLM.'''
        try:
            if self.agent.remove_cache(instruction):
                log(f"[sketch] dropped the stale cached sketch for '{instruction}'")
        except Exception as exc:  # noqa: BLE001
            log(f"[sketch] could not drop the cached sketch for '{instruction}': {exc}")

    def initialized_sketch(self, sketch_info: Dict[str, Any]) -> str:
        '''The two-line ``<concept>_1 = <concept>(...)`` / ``.construct()`` snippet,
        via the same helper SPL uses so the baselines see an identical string.'''
        return self.spl._get_class_initialization(
            sketch_info['concept'], self.values(sketch_info))
