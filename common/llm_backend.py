'''
llm_backend.py

A drop-in replacement for Demo2Code's ``call_openai_api`` that routes through SPL's
own ``openaiClient``.

Upstream (``VLM/demo2code/scripts/overall_helpers/openai_helper.py``) hard-codes
``gpt-3.5-turbo`` / ``gpt-3.5-turbo-16k`` and talks to the SDK directly. We keep its
message assembly byte-for-byte — system message split on ``<end_of_system_message>``,
few-shot examples split on ``<end_of_example_user_query>`` into alternating
user/assistant turns — and change only where the request is sent. That buys:

  * model parity with SPL (the same ``llm_model`` for every method),
  * a response cache, so re-running a baseline costs nothing,
  * per-call token accounting for the cost table,

while leaving the claim honest: the authors' released implementation, with only the
LLM backend substituted.

``install()`` monkeypatches the upstream modules that already imported the symbol.
'''

from __future__ import annotations

import hashlib
import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# Parameters the reasoning-model families reject. We start optimistic and strip
# adaptively from the error message, so a new model does not need a code change.
_ALWAYS_STRIP_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def _messages_from_upstream_prompt(main_prompt: str,
                                   raw_examples_list: Optional[Sequence[str]],
                                   user_query: str) -> List[Dict[str, str]]:
    '''Rebuild upstream's message list exactly (see openai_helper.call_openai_api).'''
    sys_msg, _, prompt = main_prompt.partition("<end_of_system_message>")
    messages = [{"role": "system", "content": sys_msg.strip("\n")},
                {"role": "user", "content": prompt.strip("\n")}]
    for example_text in (raw_examples_list or []):
        example_query, _, example_response = example_text.strip("\n").partition(
            "<end_of_example_user_query>")
        messages.append({"role": "user", "content": example_query.strip("\n")})
        messages.append({"role": "assistant", "content": example_response.strip("\n")})
    messages.append({"role": "user", "content": user_query.strip().strip("\n")})
    return messages


class LLMBackend:
    '''Cached, token-accounting LLM/VLM backend shared by every baseline.

    Input: configs: BaselineConfig - supplies api key, model names and cache dir.
    '''

    def __init__(self, configs):
        from SPL.model.llm.openai import openaiClient

        self.configs = configs
        # Parity is with SPL's *Generalize* stage: that is the model SPL uses to write
        # the concept class, so it is the model a baseline must be compared against.
        # (SketchConfig.llm_model is a different, cheaper model used only for parsing.)
        self.model = configs.generalize_config.llm_model
        self.vlm_model = getattr(configs, "vlm_model", "gpt-5")
        self.client = openaiClient(configs.generalize_config.llm_api_key,
                                   organization=getattr(configs.generalize_config, "organization", None),
                                   service_tier=getattr(configs, "service_tier", None))
        self.client.set_model(self.model)

        self.cache_path = Path(configs.llm_cache_dir) / "llm_cache.json"
        self._cache: Dict[str, Any] = self._load_cache()

        # Token ledger. The harness snapshots this per concept for the cost table.
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.num_calls = 0
        self.num_cache_hits = 0

    # ------------------------------------------------------------------ #
    # Cache
    # ------------------------------------------------------------------ #
    def _load_cache(self) -> dict:
        if not self.cache_path.exists():
            return {}
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"[LLMBackend] could not read cache ({exc}); starting fresh.")
            return {}

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, indent=1)

    @staticmethod
    def _key(model: str, messages: list, params: dict, image_digests: Sequence[str] = ()) -> str:
        blob = json.dumps({"model": model, "messages": messages, "params": params,
                           "images": list(image_digests)}, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()

    def reset_ledger(self) -> None:
        self.prompt_tokens = self.completion_tokens = self.num_calls = self.num_cache_hits = 0

    def ledger(self) -> dict:
        return {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
                "num_llm_calls": self.num_calls, "num_cache_hits": self.num_cache_hits}

    # ------------------------------------------------------------------ #
    # Core request
    # ------------------------------------------------------------------ #
    def _request(self, messages: list, params: dict, *, model: str, endpoint: str,
                 images: Optional[Sequence[bytes]] = None) -> str:
        image_digests = [hashlib.sha1(im).hexdigest() for im in (images or [])]
        key = self._key(model, messages, params, image_digests)
        if key in self._cache:
            self.num_cache_hits += 1
            return self._cache[key]["text"]

        attempt_params = dict(params)
        attempt_endpoint = endpoint
        last_exc = None
        for _ in range(8):  # each round fixes at most one incompatibility
            try:
                response = self.client.generate_response(
                    messages=messages,
                    model=model,
                    images=list(images) if images else None,
                    endpoint=attempt_endpoint,
                    return_raw=True,
                    **attempt_params,
                )
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                message = str(exc)
                # Some models (e.g. the *-codex family) are Responses-only, while
                # upstream Demo2Code is written against chat completions. Switch rather
                # than forcing the caller to know which family a model belongs to.
                if attempt_endpoint == "chat" and "v1/chat/completions" in message:
                    warnings.warn(f"[LLMBackend] {model} is Responses-only; switching endpoint.")
                    attempt_endpoint = "responses"
                    attempt_params = self._retarget_params(attempt_params, "responses")
                    continue
                dropped = self._strip_rejected_param(attempt_params, message)
                if dropped is None:
                    raise
                warnings.warn(f"[LLMBackend] {model} rejected '{dropped}'; retrying without it.")
        else:
            raise RuntimeError(f"[LLMBackend] exhausted parameter retries: {last_exc}")

        text, usage = self._extract(response, attempt_endpoint)
        if not text:
            # Reasoning models can spend the whole budget on hidden reasoning tokens and
            # return empty content. Silently caching "" would poison every later run.
            warnings.warn(f"[LLMBackend] {model} returned empty content "
                          f"(usage={usage}); not caching.")
            self.num_calls += 1
            return text
        self.num_calls += 1
        if usage:
            self.prompt_tokens += usage.get("prompt_tokens", 0)
            self.completion_tokens += usage.get("completion_tokens", 0)

        self._cache[key] = {"text": text, "usage": usage, "model": model}
        self._save_cache()
        return text

    @staticmethod
    def _retarget_params(params: dict, endpoint: str) -> dict:
        '''Translate token/stop parameters between the chat and responses endpoints.'''
        out = dict(params)
        if endpoint == "responses":
            for name in ("max_tokens", "max_completion_tokens"):
                if name in out:
                    out["max_output_tokens"] = out.pop(name)
            out.pop("stop", None)
        else:
            if "max_output_tokens" in out:
                out["max_completion_tokens"] = out.pop("max_output_tokens")
        return out

    @staticmethod
    def _strip_rejected_param(params: dict, message: str) -> Optional[str]:
        '''Remove the parameter an API error complains about. Returns its name, or
        None when the error is not about an unsupported parameter (so the caller
        re-raises instead of looping).'''
        lowered = message.lower()
        if "unsupported" not in lowered and "not supported" not in lowered \
                and "unrecognized" not in lowered and "invalid_request" not in lowered:
            return None
        # 'max_tokens' is renamed rather than dropped on the reasoning families.
        if "max_tokens" in lowered and "max_completion_tokens" in lowered and "max_tokens" in params:
            params["max_completion_tokens"] = params.pop("max_tokens")
            return "max_tokens -> max_completion_tokens"
        for name in ("temperature", "stop", "max_tokens", "max_completion_tokens", "top_p"):
            if name in lowered and name in params:
                params.pop(name)
                return name
        return None

    @staticmethod
    def _extract(response: Any, endpoint: str) -> Tuple[str, dict]:
        usage_obj = getattr(response, "usage", None)
        usage = {}
        if usage_obj is not None:
            usage = {
                "prompt_tokens": getattr(usage_obj, "prompt_tokens", None)
                or getattr(usage_obj, "input_tokens", 0) or 0,
                "completion_tokens": getattr(usage_obj, "completion_tokens", None)
                or getattr(usage_obj, "output_tokens", 0) or 0,
            }
        if endpoint == "chat":
            return (response.choices[0].message.content or "").strip(), usage
        text = getattr(response, "output_text", None)
        if text is None:
            chunks = []
            for item in getattr(response, "output", []) or []:
                for content in getattr(item, "content", []) or []:
                    if getattr(content, "text", None):
                        chunks.append(content.text)
            text = "".join(chunks)
        return (text or "").strip(), usage

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def call_openai_api(self, main_prompt, raw_examples_list, user_query, stop_list=[],
                        temperature=0.7, max_tokens=2000, gpt_model=None):
        '''Signature-compatible with upstream ``openai_helper.call_openai_api``.

        ``gpt_model`` is accepted but ignored: upstream hard-codes gpt-3.5 at several
        call sites, and the whole point of this backend is that every method uses the
        same model as SPL.
        '''
        del gpt_model  # deliberately ignored — see docstring
        messages = _messages_from_upstream_prompt(main_prompt, raw_examples_list, user_query)
        params: Dict[str, Any] = {"max_tokens": max_tokens}
        if not self.model.startswith(_ALWAYS_STRIP_PREFIXES):
            params["temperature"] = temperature
            if stop_list:
                params["stop"] = list(stop_list)
        elif "max_tokens" in params:
            params["max_completion_tokens"] = params.pop("max_tokens")
        return self._request(messages, params, model=self.model, endpoint="chat")

    def call_text(self, system_message: str, user_query: str, *, max_tokens: int = 4000) -> str:
        '''A plain single-turn text call, for our own agents' prompts.'''
        messages = [{"role": "system", "content": system_message},
                    {"role": "user", "content": user_query}]
        params: Dict[str, Any] = ({"max_completion_tokens": max_tokens}
                                  if self.model.startswith(_ALWAYS_STRIP_PREFIXES)
                                  else {"max_tokens": max_tokens})
        return self._request(messages, params, model=self.model, endpoint="chat")

    def call_vlm(self, system_message: str, user_query: str, images: Sequence[bytes],
                 *, max_tokens: int = 6000) -> str:
        '''A vision call. Images are raw PNG bytes; ``openaiClient`` turns them into
        data URIs. Uses the Responses endpoint, which handles image input natively.'''
        messages = [{"role": "system", "content": system_message},
                    {"role": "user", "content": user_query}]
        params = {"max_output_tokens": max_tokens}
        return self._request(messages, params, model=self.vlm_model,
                             endpoint="responses", images=images)


def load_upstream_code_generator(backend: LLMBackend, configs=None):
    '''Import the authors' ``code_gen_helper`` with our backend wired in.

    Two obstacles, neither of which requires editing their files:

      * ``code_gen_helper`` does ``from scripts.overall_helpers.lmp import
        call_openai_api``, and ``lmp.py`` star-imports ``shapely`` and ``astunparse``
        (absent here) purely for the CodeAsPolicies ``LMPFGen`` machinery we do not
        use. We register a tiny shim module under that import path first, so the real
        ``lmp`` — and its unused dependencies — never load.
      * ``openai_helper`` reads ``os.environ["OPENAI_API_KEY"]`` at import and raises
        KeyError if unset. The shim means it is never imported, but we set the variable
        anyway so a future import path cannot trip on it.

    Because the shim already supplies ``call_openai_api``, ``code_gen_helper`` binds our
    function at import time; the explicit rebind afterwards is belt-and-braces for the
    case where the module was imported earlier in the process.

    Returns the imported ``code_gen_helper`` module.
    '''
    import sys
    import types
    from baseline_spl.common.config import BASELINE_ROOT

    upstream = BASELINE_ROOT / "VLM" / "demo2code"
    if not (upstream / "scripts").is_dir():
        raise FileNotFoundError(f"Demo2Code clone not found at {upstream}")
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))

    if not os.environ.get("OPENAI_API_KEY"):
        key = (getattr(configs, "generalize_config", None) and configs.generalize_config.llm_api_key)
        os.environ["OPENAI_API_KEY"] = key or "unused-by-baseline-backend"

    shim_name = "scripts.overall_helpers.lmp"
    if shim_name not in sys.modules:
        # Ensure the parent packages exist so the shim is reachable by dotted name.
        import importlib
        importlib.import_module("scripts.overall_helpers")
        shim = types.ModuleType(shim_name)
        shim.call_openai_api = backend.call_openai_api
        shim.__doc__ = "Shim installed by baseline_spl; upstream lmp.py is unused here."
        sys.modules[shim_name] = shim
    else:
        sys.modules[shim_name].call_openai_api = backend.call_openai_api

    import importlib
    code_gen_helper = importlib.import_module("scripts.overall_helpers.code_gen_helper")
    code_gen_helper.call_openai_api = backend.call_openai_api
    return code_gen_helper
