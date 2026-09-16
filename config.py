'''
config.py

Loads the run configuration named by the BASELINE_CONFIG environment variable from
``baseline_spl/configs/``, defaulting to ``default``:

    BASELINE_CONFIG=cap_nodemo python -m baseline_spl.VLM.cap.run

Settings live in the config files; this module only selects one, so
``from baseline_spl.config import CapConfig`` always gives the active experiment's
classes. common/config.py turns them into the SPL-compatible config object.
'''

from __future__ import annotations

import importlib
import os
from pathlib import Path

DEFAULT_CONFIG = "default"
CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def available_configs() -> list:
    return sorted(p.stem for p in CONFIG_DIR.glob("*.py") if not p.stem.startswith("_"))


def load(name: str = None):
    '''Import a config module by name, failing with the list of what exists.

    Falls back to `baseline_spl/tests/configs/`, which holds the smoke and golden configs.
    Keeping those out of `configs/` means the directory lists the experiments you actually
    run, not the scaffolding that tests the code.
    '''
    name = name or os.environ.get("BASELINE_CONFIG") or DEFAULT_CONFIG
    try:
        return importlib.import_module(f"baseline_spl.configs.{name}")
    except ImportError:
        pass
    try:
        return importlib.import_module(f"baseline_spl.tests.configs.{name}")
    except ImportError as exc:
        raise ImportError(
            f"BASELINE_CONFIG={name!r} does not name a file in {CONFIG_DIR} "
            f"or {CONFIG_DIR.parent / 'tests' / 'configs'}. "
            f"Available: {', '.join(available_configs())}"
        ) from exc


ACTIVE_CONFIG = os.environ.get("BASELINE_CONFIG") or DEFAULT_CONFIG
_module = load(ACTIVE_CONFIG)

CommonConfig = _module.CommonConfig
CapConfig = _module.CapConfig
Demo2CodeConfig = _module.Demo2CodeConfig
SayCanConfig = _module.SayCanConfig
# Config files predating the symbolic baselines do not define these.
DreamCoderConfig = getattr(_module, "DreamCoderConfig", None)
LiloConfig = getattr(_module, "LiloConfig", None)
CODEGEN_MODEL = _module.CODEGEN_MODEL
VLM_MODEL = _module.VLM_MODEL
