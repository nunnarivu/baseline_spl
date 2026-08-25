'''
codegen.py

The structural-validation and retry loop shared by every baseline.

Centralised deliberately: all baselines must get the *same* retry budget and the same
acceptance criterion, otherwise a difference in results could be a difference in how
many second chances each got. Validation reuses SPL's own
``GeneralizeAgent._validate_class_code`` (a staticmethod) so the required class shape can
never drift between SPL and the baselines.

Note what is deliberately absent: SPL's execution-grounded evaluator
(``SPL._build_concept_class_evaluator``, which runs the candidate on the demonstration,
scores its reward and re-prompts with a per-block divergence report). That loop is an SPL
contribution; baselines retry on parse/structure failures only.
'''

from __future__ import annotations

import ast
from typing import Dict, List, Optional, Tuple

from SPL.model.generalize import ConceptClassParseError, GeneralizeAgent
from SPL.model.text_utils import extract_code_block

# Annotations we can map to a real type. Anything else becomes `object`, which makes
# check_program_equivalence report "undecided" rather than guess at an arity match.
_ANNOTATION_TYPES = {"int": int, "float": float, "list": list, "str": str, "bool": bool}


def validate(code: str) -> None:
    '''Raise ConceptClassParseError unless the code parses, defines a class and carries
    every method SPL requires.'''
    GeneralizeAgent._validate_class_code(code)


def class_signature(code: str) -> Tuple[str, Dict[str, type]]:
    '''(class name, {argument: type}) read off the generated class's __init__.

    Used when no sketch was given, so the name and arguments are whatever the model
    invented and have to be recovered from the code itself.
    '''
    tree = ast.parse(code)
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    if not classes:
        raise ConceptClassParseError("no class definition found", code=code)
    cls = classes[0]

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
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    if not classes:
        raise ConceptClassParseError("no class definition found", code=code)
    original = classes[0].name
    if original == wanted:
        return code

    class _Rename(ast.NodeTransformer):
        def visit_ClassDef(self, node: ast.ClassDef):
            self.generic_visit(node)
            if node.name == original:
                node.name = wanted
            return node

        def visit_Name(self, node: ast.Name):
            if node.id == original:
                node.id = wanted
            return node

    renamed = ast.fix_missing_locations(_Rename().visit(tree))
    return ast.unparse(renamed)


def generate_with_retries(backend, system_prompt: str, user_prompt: str, *,
                          wanted_name: Optional[str] = None, max_retries: int = 3,
                          max_tokens: int = 6000, images=None,
                          log=print) -> Optional[str]:
    '''Ask for a concept class, retrying only on structural failure.

    ``images`` (PNG bytes) routes the request to the vision model instead; they stay
    attached across retries, since a retry that dropped them would be answering a
    different question.

    ``wanted_name`` renames the class to the sketch's concept. Pass None when no sketch
    was given: the model's own class name is then the concept name.

    Output: the validated class source, or None if every attempt failed to parse.
    '''
    prompt = user_prompt
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            if images:
                response = backend.call_vlm(system_prompt, prompt, images,
                                            max_tokens=max_tokens)
            else:
                response = backend.call_text(system_prompt, prompt, max_tokens=max_tokens)
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
            return code
        except (ConceptClassParseError, ValueError, SyntaxError) as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            log(f"[codegen] attempt {attempt + 1}/{max_retries + 1} invalid: {exc}")
            prompt = (f"{user_prompt}\n\n"
                      f"# Your previous answer was rejected\n"
                      f"{GeneralizeAgent._build_retry_prompt(exc)}\n\n"
                      f"Previous answer:\n```python\n{locals().get('code', response)[:4000]}\n```")

    log(f"[codegen] giving up after {max_retries + 1} attempts. Last error: {last_error}")
    return None
