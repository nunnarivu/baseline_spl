'''
_dreamcoder.py

Import DreamCoder's pure-Python core without executing its package __init__.

`third_party/lilo/dreamcoder/__init__.py` exists to remap module paths for old pickle files,
and to do it it imports *every* domain -- tower, logo, list, regex, misc/algolisp,
misc/RobustFill. Two problems with letting that run:

  1. It drags in dill, pathos, pregex and the rest of the experiment stack, none of which we
     use.
  2. `Primitive.GLOBALS` is a process-global name -> primitive dict (program.py:682), so
     importing those domains registers 176 primitives into it. Several of their names collide
     with ours -- the tower domain alone registers `Primitive(str(j), tint, j)` for j in
     1..49, i.e. "1" and "2" with *its* tint -- and the last registration wins. That would
     silently corrupt `Program.parse` for our grammar.

So we install a stub package pointing at the same directory. `import dreamcoder.grammar` then
resolves the submodule through __path__ without running the real __init__, and only the chain
we actually need loads: type -> utilities -> program -> frontier/task -> grammar.

This must run before anything else imports `dreamcoder`, which it does: every use in this
package goes through here.
'''

from __future__ import annotations

import sys
import types
from pathlib import Path

LILO_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "lilo"
DREAMCODER_DIR = LILO_ROOT / "dreamcoder"


def _install_stub_package() -> None:
    existing = sys.modules.get("dreamcoder")
    if existing is not None:
        if getattr(existing, "_spl_stub", False):
            return
        # Someone imported the real package first; the GLOBALS pollution described above has
        # already happened, so fail loudly rather than produce silently wrong programs.
        raise ImportError(
            "the real `dreamcoder` package is already imported, which pollutes "
            "Primitive.GLOBALS with other domains' primitives. Import "
            "baseline_spl.symbolic._dreamcoder before anything that imports dreamcoder.")

    if not DREAMCODER_DIR.is_dir():
        raise ImportError(f"DreamCoder source not found at {DREAMCODER_DIR}")

    if str(LILO_ROOT) not in sys.path:
        sys.path.insert(0, str(LILO_ROOT))

    stub = types.ModuleType("dreamcoder")
    stub.__path__ = [str(DREAMCODER_DIR)]
    stub.__doc__ = "Stub package: see baseline_spl/symbolic/_dreamcoder.py"
    stub._spl_stub = True
    sys.modules["dreamcoder"] = stub


_install_stub_package()

from dreamcoder.grammar import Grammar                                   # noqa: E402
from dreamcoder.program import (Abstraction, Application, Index,         # noqa: E402
                                Invented, Primitive, Program)
from dreamcoder.task import Task                                        # noqa: E402
from dreamcoder.frontier import Frontier, FrontierEntry                 # noqa: E402
from dreamcoder.type import arrow, baseType                             # noqa: E402

__all__ = ["Grammar", "Abstraction", "Application", "Index", "Invented", "Primitive",
           "Program", "Task", "Frontier", "FrontierEntry", "arrow", "baseType",
           "registered_primitive_names"]


def registered_primitive_names() -> set:
    '''What is currently in Primitive.GLOBALS. Asserted in test_bridge to prove no other
    domain's primitives leaked in.'''
    return set(Primitive.GLOBALS)
