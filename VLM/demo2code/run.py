'''
run.py

Runs the Demo2Code baseline. Settings come from baseline_spl/config.py
(Demo2CodeConfig); there are no command-line arguments. Set `variant` there to choose
between the text and VLM pipelines.

    python -m baseline_spl.VLM.demo2code.run
'''

from __future__ import annotations

from baseline_spl.common.factory import run
from baseline_spl.config import Demo2CodeConfig


def make_agent(configs, backend):
    if Demo2CodeConfig.variant == "vlm":
        from baseline_spl.VLM.demo2code.agent_vlm import Demo2CodeVLMAgent
        return Demo2CodeVLMAgent(configs, backend)
    from baseline_spl.VLM.demo2code.agent_text import Demo2CodeTextAgent
    return Demo2CodeTextAgent(configs, backend)


def main() -> None:
    variant = Demo2CodeConfig.variant
    if variant not in ("text", "vlm"):
        raise ValueError(f"Demo2CodeConfig.variant must be 'text' or 'vlm', got {variant!r}")
    baseline = f"demo2code_{variant}"
    if Demo2CodeConfig.sketch_mode == "none":
        baseline += "_nosketch"
    run(Demo2CodeConfig, baseline, make_agent)


if __name__ == "__main__":
    main()
