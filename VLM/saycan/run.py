'''
run.py

Runs the SayCan baseline. Settings come from the SayCanConfig of the config file named by
BASELINE_CONFIG (default: configs/default.py). No command-line arguments.

    BASELINE_CONFIG=my_experiment python -m baseline_spl.VLM.saycan.run
'''

from __future__ import annotations

from baseline_spl.common.config import BaselineConfig
from baseline_spl.common.factory import resolve_run_name
from baseline_spl.common.llm_backend import LLMBackend
from baseline_spl.config import SayCanConfig
from baseline_spl.VLM.saycan.agent import SayCanAgent
from baseline_spl.VLM.saycan.harness import SayCanHarness


def main() -> None:
    if not (SayCanConfig.learn or SayCanConfig.inference):
        raise ValueError("Set learn and/or inference to True in the active config file")

    run_name = resolve_run_name(SayCanConfig, f"saycan_{SayCanConfig.demo_modality}")
    configs = BaselineConfig.from_run_config(SayCanConfig, run_name)
    harness = SayCanHarness(configs, SayCanAgent(configs, LLMBackend(configs)))

    if SayCanConfig.learn:
        harness.learn_all()
    if SayCanConfig.inference:
        harness.infer_all()


if __name__ == "__main__":
    main()
