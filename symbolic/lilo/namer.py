'''
namer.py

LILO's third contribution: auto-documentation of the learned library.

Re-templated from `src/models/library_namer.py:60-110` rather than imported -- upstream's
class depends on `LAPSGrammar.show_program` and on `openai<1.0`, neither of which exists
here. The prompt shape is theirs: show the library, show the target abstraction with its
type, body and usage examples, ask for JSON with a unique underscore-separated
`readable_name` and a `description`, and allow `null` when no good name exists.

Why this is load-bearing rather than cosmetic: `proposer.substitute_aliases` resolves these
names back to their anonymous bodies, so a named abstraction is one the model can actually
*write*. Compare `(line_of_blocks RIGHT $1 $0)` with the alternative it replaces --
`(#(lambda (lambda (lambda (loop (sub $1 1) (lambda (lambda (place (shift $4 $0)))) (place $0))))) RIGHT $1 $0)`.
Naming is what makes a grown library usable by a language model at all.
'''

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Sequence

SYSTEM = '''You are writing software documentation for a library of functions that build
block structures on a 3-D lattice.

The functions are written in a small typed lambda calculus with de Bruijn indices. They were
discovered automatically by compressing programs that solved real construction tasks, so each
one captures a pattern that recurred across several structures.

Your job is to give a function a human-readable name and a one-sentence description.

Rules:
- `readable_name` must be underscore_separated, lowercase, and contain no spaces.
- It must be unique: not the name of any function already in the library.
- Name what the function BUILDS or DOES, judging from its body and its usage examples --
  not how it is implemented.
- If you cannot come up with a good name, set `readable_name` to null.
- Reply with ONLY the JSON object, no prose and no markdown fence.
'''


def _docstring(name: str, abstraction, description: Optional[str] = None) -> str:
    try:
        signature = str(abstraction.infer())
    except Exception:  # noqa: BLE001
        signature = "?"
    text = f"{name} :: {signature}\n{abstraction}"
    if description:
        text += f"\ndescription: {description}"
    return text


def _parse_reply(text: str) -> Optional[dict]:
    '''Pull the JSON object out of a reply, fence or no fence.'''
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except (ValueError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


class LibraryNamer:
    '''Names and describes each newly accepted abstraction.

    Input: backend - common.llm_backend.LLMBackend
    '''

    def __init__(self, backend, *, max_tokens: int = 600, log=print):
        self.backend = backend
        self.max_tokens = max_tokens
        self.log = log
        self.documentation: Dict[str, dict] = {}   # str(abstraction) -> {name, description}
        self.calls = 0
        self.failures = 0

    def __call__(self, abstractions: Sequence, corpus: Sequence) -> Dict[str, dict]:
        '''The `document` hook `driver.run` calls, after compression accepts abstractions.'''
        fresh: Dict[str, dict] = {}
        for index, abstraction in enumerate(abstractions):
            body = str(abstraction)
            if body in self.documentation:
                continue
            anonymous = f"fn_{len(self.documentation) + len(fresh)}"
            doc = self._name_one(anonymous, abstraction, corpus)
            if doc is not None:
                fresh[body] = doc
        self.documentation.update(fresh)
        return fresh

    # ------------------------------------------------------------------ #

    def _name_one(self, anonymous: str, abstraction, corpus: Sequence) -> Optional[dict]:
        taken = {d["readable_name"] for d in self.documentation.values()
                 if d.get("readable_name")}
        query = self._prompt(anonymous, abstraction, corpus, taken)
        try:
            reply = self.backend.call_text(SYSTEM, query, max_tokens=self.max_tokens)
        except Exception as exc:  # noqa: BLE001 - naming must never end the run
            self.log(f"  naming {anonymous} failed: {exc}")
            self.failures += 1
            return None
        self.calls += 1

        payload = _parse_reply(reply)
        if payload is None:
            self.failures += 1
            return None
        name = payload.get("readable_name")
        if not name or not isinstance(name, str):
            self.failures += 1                     # the model declined; fn_N stands
            return None
        name = re.sub(r"\W+", "_", name.strip()).strip("_").lower()
        if not name or name in taken:
            self.failures += 1
            return None
        return {"anonymous_name": anonymous, "readable_name": name,
                "description": str(payload.get("description") or "").strip(),
                "body": str(abstraction)}

    def _prompt(self, anonymous: str, abstraction, corpus: Sequence,
                taken: set) -> str:
        lines: List[str] = []
        if self.documentation:
            lines.append("# Functions already in the library")
            lines.append("")
            for body, doc in self.documentation.items():
                lines.append(_docstring(doc["readable_name"], _Body(body),
                                        doc.get("description")))
                lines.append("")
        lines.append("# The function to name")
        lines.append("")
        lines.append(_docstring(anonymous, abstraction))
        lines.append("")

        body = str(abstraction)
        usages = [(name, str(program)) for name, program in corpus if body in str(program)]
        if usages:
            lines.append("# Where it is used")
            lines.append("")
            for name, program in usages[:6]:
                lines.append(f"    building a {name}:")
                lines.append(f"    {program}")
            lines.append("")
        if taken:
            lines.append(f"Names already taken: {', '.join(sorted(taken))}")
            lines.append("")
        lines.append("Reply with only this JSON object:")
        lines.append('{"anonymous_name": "%s", "readable_name": ..., "description": ...}'
                     % anonymous)
        return "\n".join(lines)


class _Body:
    '''Renders an already-stringified abstraction back through `_docstring`.'''

    def __init__(self, body: str):
        self.body = body

    def __str__(self) -> str:
        return self.body

    def infer(self):
        from baseline_spl.symbolic._dreamcoder import Program
        return Program.parse(self.body).infer()
