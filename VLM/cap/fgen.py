'''
fgen.py

Code-as-Policies' signature move: hierarchical code generation.

When the generated program calls a helper it never defined, CaP asks the LLM to write
that helper, then recurses into the helper's own body. Upstream's implementation lives
in ``VLM/demo2code/scripts/overall_helpers/lmp.py:LMPFGen``; it is reimplemented here
rather than imported because upstream hard-codes robotouille output paths and
star-imports ``shapely``/``astunparse`` for machinery we do not use. The algorithm —
parse for undefined calls, generate each, recurse one level deeper — follows it.

Generated helpers are prepended to the class source, so they are exec'd into the concept
library namespace alongside the class (``ConceptLibrary.register_inductive_concepts``
does ``exec(class_code, self.operators)``) and travel with the stored code when
``utils/metrics.py`` replays the program on the ideal executor.
'''

from __future__ import annotations

import ast
import builtins
from typing import Dict, List, Sequence, Set

from SPL.model.text_utils import extract_code_block


FGEN_SYSTEM = '''You write small Python helper functions for a robot block-construction program.
You will be given the signature of a function that is called but not yet defined, plus the
action DSL it may use. Write only that function.

Output ONLY the function definition inside a single ```python ... ``` block.
No explanation, no example usage. Do not redefine the DSL primitives.'''


def _defined_names(tree: ast.AST) -> Set[str]:
    '''Top-level function and class names bound by this module.'''
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
    return names


def _called_names(tree: ast.AST) -> Dict[str, int]:
    '''Bare function calls (``foo(...)``) mapped to their positional-argument count.
    Attribute calls such as ``self.foo()`` or ``np.array()`` are ignored: they are
    resolved by an object, not by a free name.'''
    calls: Dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            calls.setdefault(node.func.id, len(node.args))
    return calls


def undefined_calls(code: str, known: Sequence[str]) -> Dict[str, int]:
    '''Names the code calls but never defines, excluding builtins, the DSL primitives
    and anything already in the concept library.'''
    tree = ast.parse(code)
    excluded = set(dir(builtins)) | set(known) | _defined_names(tree)
    return {name: nargs for name, nargs in _called_names(tree).items() if name not in excluded}


def expand_helpers(code: str, *, backend, known: Sequence[str], dsl_doc: str,
                   max_depth: int = 2, log=print) -> str:
    '''Recursively generate any helper the code calls but does not define.

    Input:  code      : the generated class source
            backend   : LLMBackend, for the generation calls
            known     : names already available at exec time (library operators)
            dsl_doc   : DSL documentation to include in each helper prompt
            max_depth : recursion limit, mirroring CaP's bounded expansion
    Output: the source with every generated helper prepended.
    '''
    generated: List[str] = []
    known_set = set(known)

    current = code
    for depth in range(max_depth):
        try:
            missing = undefined_calls(current, known_set)
        except SyntaxError as exc:
            log(f"[cap.fgen] cannot parse code at depth {depth}: {exc}")
            break
        if not missing:
            break

        new_sources = []
        for name, nargs in missing.items():
            signature = f"{name}({', '.join(f'arg{i + 1}' for i in range(nargs))})"
            log(f"[cap.fgen] depth {depth}: generating helper {signature}")
            user = (f"{dsl_doc}\n\n"
                    f"Names already available (do not redefine): {sorted(known_set)}\n\n"
                    f"Define the function: {signature}")
            try:
                response = backend.call_text(FGEN_SYSTEM, user, max_tokens=1500)
                source = extract_code_block(response, dedent=True)
                ast.parse(source)
            except Exception as exc:  # noqa: BLE001
                log(f"[cap.fgen] failed to generate {name}: {exc}")
                continue
            new_sources.append(source)
            known_set.add(name)

        if not new_sources:
            break
        generated.extend(new_sources)
        # Recurse into the newly written bodies, which may call further helpers.
        current = "\n\n".join(new_sources)

    if not generated:
        return code
    return "\n\n".join(generated) + "\n\n" + code
