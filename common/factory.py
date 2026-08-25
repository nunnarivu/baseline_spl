'''
factory.py

Turns a run-config class from the active config file into a ready-to-run harness.

Shared by every baseline's run.py so the wiring — parity check, run directory, backend,
agent — is defined once.
'''

from __future__ import annotations

from baseline_spl.common.config import BaselineConfig
from baseline_spl.common.harness import BaselineHarness
from baseline_spl.common.llm_backend import LLMBackend


def resolve_run_name(run_cfg, baseline: str) -> str:
    '''Output directory under runs/.

    Defaults to "<config file>_<baseline>" so two config files can never overwrite each
    other's results, and the directory name says which experiment produced it. An
    explicit run_name in the config wins.
    '''
    if run_cfg.run_name:
        return run_cfg.run_name
    from baseline_spl.config import ACTIVE_CONFIG

    return f"{ACTIVE_CONFIG}_{baseline}"


def build(run_cfg, baseline: str, make_agent) -> BaselineHarness:
    '''Input: run_cfg    - a class from the active config file
              baseline   - short name of this baseline, e.g. 'cap' or 'demo2code_text'
              make_agent - callable (configs, backend) -> agent
    '''
    configs = BaselineConfig.from_run_config(run_cfg, resolve_run_name(run_cfg, baseline))
    backend = LLMBackend(configs)
    agent = make_agent(configs, backend)
    return BaselineHarness(configs, agent)


def run(run_cfg, baseline: str, make_agent) -> None:
    '''Build and execute the learn / inference phases the run config asks for.'''
    if not (run_cfg.learn or run_cfg.inference):
        raise ValueError("Set learn and/or inference to True in the active config file")

    harness = build(run_cfg, baseline, make_agent)
    if run_cfg.learn:
        harness.learn_all()
    if run_cfg.inference:
        harness.infer_all()
