'''
run.py

Runs the Code-as-Policies baseline. Settings come from the CapConfig of the config file
named by BASELINE_CONFIG (default: configs/default.py). No command-line arguments.

    BASELINE_CONFIG=my_experiment python -m baseline_spl.VLM.cap.run
'''

from __future__ import annotations

from baseline_spl.common.factory import run
from baseline_spl.config import CapConfig
from baseline_spl.VLM.cap.agent import CodeAsPoliciesAgent


def main() -> None:
    baseline = "cap_nodemo" if not CapConfig.use_demo else f"cap_{CapConfig.demo_modality}"
    if CapConfig.sketch_mode == "none":
        baseline += "_nosketch"
    run(CapConfig, baseline, CodeAsPoliciesAgent)


if __name__ == "__main__":
    main()
