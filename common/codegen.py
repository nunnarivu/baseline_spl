'''
codegen.py

The structural-validation and retry loop shared by every baseline.

Centralised deliberately: all baselines must get the *same* retry budget and the same
acceptance criterion, otherwise a difference in results could be a difference in how
many second chances each got. Validation reuses SPL's own
``GeneralizeAgent._validate_class_code`` (a staticmethod) so the required class shape can
never drift between SPL and the baselines.

By default baselines retry on parse/structure failures only: SPL's execution-grounded
evaluator (``SPL._build_concept_class_evaluator``) is an SPL contribution. With
``use_evaluator_feedback`` the loop also runs each class on the demonstrations
(common/evaluator.py) and re-prompts with the report, within the same retry budget.
'''

from __future__ import annotations

import ast
from typing import Dict, List, Optional, Tuple

from SPL.config.spl_config import ALL_CONCEPTS
from SPL.model.generalize import ConceptClassParseError, GeneralizeAgent
from SPL.model.text_utils import extract_code_block
from baseline_spl.common.harness import log

# SPL's evaluation retry prompts (GeneralizeAgent._build_eval_retry_prompt and
# _build_bookkeeping_prompt), reworded: a baseline never saw plans or execution traces.
_OUTPUT_ONLY = ("Output ONLY the class source code inside a single fenced ```python ... ``` "
                "block, with the same required methods. Do not include any explanation or "
                "example usage.")
EVAL_RETRY_PROMPT = (
    "# Your previous class was run on the demonstrations and did not reproduce them\n"
    "Evaluation report: {report}\n\n"
    "Focus your fix on the blocks the report names. Fix the general rule that produces them "
    "so it holds for every argument value; do not special-case specific argument values "
    "(e.g. `if self.length == 5` or a dict keyed on lengths). " + _OUTPUT_ONLY)
BOOKKEEPING_RETRY_PROMPT = (
    "# Your previous class reproduces the demonstrations, but its bookkeeping is wrong\n"
    "`blocks`, `key_blocks` or a substructure's `blocks` do not match the objects "
    "construct() actually placed.\n"
    "Evaluation report: {report}\n\n"
    "Keep the construction logic. Track the objects construct() places (including those "
    "placed by substructures) and build `blocks` and `key_blocks` from them, not from the "
    "`objects` passed in. " + _OUTPUT_ONLY)

# Annotations we can map to a real type. Anything else becomes `object`, which makes
# check_program_equivalence report "undecided" rather than guess at an arity match.
_ANNOTATION_TYPES = {"int": int, "float": float, "list": list, "str": str, "bool": bool}


def validate(code: str) -> None:
    '''Raise ConceptClassParseError unless the code parses, defines a class and carries
    every method SPL requires.'''
    GeneralizeAgent._validate_class_code(code)


def _target_class(tree: ast.Module, code: str, wanted: Optional[str] = None) -> ast.ClassDef:
    '''The concept class among the module's top-level classes.

    Chosen by shape -- a `construct` method and an `__init__` taking `objects` -- rather
    than by position. Models that also define a substructure helper class put the concept
    class second as often as first, and the rename, the signature and the name it is
    registered under all have to agree on which class is the concept.

    A class already named `wanted` wins outright. Without that, a helper class named after
    a real concept is equally well shaped, and renaming *it* to `wanted` would leave two
    classes with the same name, the second silently shadowing the first.
    '''
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    if not classes:
        raise ConceptClassParseError("no class definition found", code=code)

    if wanted is not None:
        exact = next((c for c in classes if c.name == wanted), None)
        if exact is not None:
            return exact

    def shaped(cls: ast.ClassDef) -> bool:
        init = next((n for n in cls.body
                     if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
        return (init is not None
                and any(a.arg == "objects" for a in init.args.args)
                and any(isinstance(n, ast.FunctionDef) and n.name == "construct"
                        for n in cls.body))

    # Falling back to the first class keeps the error messages below pointing at something
    # concrete when nothing matches, instead of raising a less useful "no class" here.
    return next((c for c in classes if shaped(c)), classes[0])


def class_signature(code: str) -> Tuple[str, Dict[str, type]]:
    '''(class name, {argument: type}) read off the generated class's __init__.

    Used when no sketch was given, so the name and arguments are whatever the model
    invented and have to be recovered from the code itself.
    '''
    cls = _target_class(ast.parse(code), code)

    init = next((n for n in cls.body
                 if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
    if init is None:
        raise ConceptClassParseError(f"class {cls.name} has no __init__", code=code)

    arguments: Dict[str, type] = {}
    for arg in init.args.args:
        if arg.arg == "self":
            continue
        annotation = getattr(arg.annotation, "id", None)
        arguments[arg.arg] = _ANNOTATION_TYPES.get(annotation, object)

    if "objects" not in arguments:
        raise ConceptClassParseError(
            f"class {cls.name}.__init__ has no `objects` argument; the harness supplies "
            f"the object ids through it, so the class cannot be executed without one.",
            code=code)
    return cls.name, arguments


def ensure_class_name(code: str, wanted: str) -> str:
    '''Rename the generated class to the name the shared sketch assigned.

    The concept name comes from the sketch, and ``SPL._get_class_initialization`` builds
    ``<concept>_1 = <concept>(...)`` from it, so a class named anything else is
    unusable. Models occasionally CamelCase it. Renaming via the AST (rather than a
    textual substitution) also fixes internal self-references, which matters for the
    recursive concepts.
    '''
    tree = ast.parse(code)
    target = _target_class(tree, code, wanted)
    renames = {} if target.name == wanted else {target.name: wanted}

    # Any OTHER top-level class named after a real concept is neutralised.
    # register_inductive_concepts execs the WHOLE source into the library namespace
    # (SPL/model/concept_library.py), so a leftover `class tower` would bind `tower` there
    # as a live callable even under concept_name="reversed", and a later concept could
    # call it. Renaming keeps the model's own substructure working while keeping the real
    # name out of the library.
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node is not target and node.name in ALL_CONCEPTS:
            renames[node.name] = f"_local_{node.name}"
            log(f"[codegen] neutralised extra class {node.name!r} -> {renames[node.name]!r}")

    if not renames:
        return code   # unchanged source, so the model's own comments survive

    class _Rename(ast.NodeTransformer):
        def visit_ClassDef(self, node: ast.ClassDef):
            self.generic_visit(node)
            node.name = renames.get(node.name, node.name)
            return node

        def visit_Name(self, node: ast.Name):
            node.id = renames.get(node.id, node.id)
            return node

    renamed = ast.fix_missing_locations(_Rename().visit(tree))
    return ast.unparse(renamed)


def generate_with_retries(backend, system_prompt: str, user_prompt: str, *,
                          wanted_name: Optional[str] = None, max_retries: int = 3,
                          max_tokens: int = 6000, images=None, evaluator=None,
                          log=print) -> Optional[str]:
    '''Ask for a concept class, retrying on structural failure and, with an ``evaluator``
    (common/evaluator.py), on failing to reproduce the demonstrations.

    ``images`` (PNG bytes) routes the request to the vision model instead; they stay
    attached across retries, since a retry that dropped them would be answering a
    different question.

    ``wanted_name`` renames the class to the sketch's concept. Pass None when no sketch
    was given: the model's own class name is then the concept name.

    Output: the first class that validates (and passes the evaluator), else the evaluator's
    best-scoring class, else None if every attempt failed to parse.
    '''
    # One conversation: the task, the demonstration and any images go out once, and a
    # retry sends only what was wrong. Resending everything would repeat a ~2,300-token
    # system prompt and every keyframe on each of the four attempts, and would also
    # diverge from SPL, whose GeneralizeAgent chains its retries the same way.
    conversation = backend.start_conversation(system_prompt, images=images)
    prompt = user_prompt
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            response = conversation.ask(prompt, max_tokens=max_tokens)
        except Exception as exc:  # noqa: BLE001
            log(f"[codegen] LLM call failed on attempt {attempt + 1}: {exc}")
            last_error = exc
            continue

        try:
            code = extract_code_block(response, dedent=True)
            validate(code)
            if wanted_name:
                code = ensure_class_name(code, wanted_name)
                validate(code)  # renaming must not have broken the required shape
            else:
                class_signature(code)  # must expose a usable __init__ with `objects`
        except (ConceptClassParseError, ValueError, SyntaxError) as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            log(f"[codegen] attempt {attempt + 1}/{max_retries + 1} invalid: {exc}")
            # Only the correction: the conversation still holds the task and the answer.
            prompt = (f"# Your previous answer was rejected\n"
                      f"{GeneralizeAgent._build_retry_prompt(exc)}")
            continue

        if evaluator is None:
            return code
        passed, _score, report = evaluator(code)
        if passed:
            return code
        last_error = report
        if attempt >= max_retries:
            break
        log(f"[codegen] attempt {attempt + 1}/{max_retries + 1} did not reproduce the demonstrations:\n{report}")
        template = BOOKKEEPING_RETRY_PROMPT if report.startswith("BOOKKEEPING:") else EVAL_RETRY_PROMPT
        prompt = template.format(report=report)

    if evaluator is not None and evaluator.best_code is not None:
        log(f"[codegen] no class passed the evaluator in {max_retries + 1} attempts; "
            f"keeping the best-scoring one (score {evaluator.best_score:.6f}).")
        return evaluator.best_code
    log(f"[codegen] giving up after {max_retries + 1} attempts. Last error: {last_error}")
    return None
