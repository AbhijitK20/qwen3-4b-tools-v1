"""
Register the local Ollama-served Qwen3-4B fine-tunes as BFCL models.

BFCL's `OpenAICompletionsHandler` already speaks the OpenAI chat-completions API
in native function-calling mode, and Ollama exposes an OpenAI-compatible
endpoint at /v1. So instead of letting BFCL spin up its own vLLM server
(which it only does for `BaseOSSHandler` models), we register our GGUF models
as `is_fc_model=True` OpenAI-completions models pointed at Ollama.

BFCL then constructs the handler with model_name=<our Ollama tag>, sends the
tool schemas in the native `tools` field, and reads back `message.tool_calls`.
This is the same code path a hosted frontier model takes, which is what makes
the resulting score comparable to published numbers.

Usage (via run_bfcl.py, which patches this in before invoking the CLI):
    from bfcl_ollama import register_models
    register_models()
"""

from __future__ import annotations

import os
from typing import Optional

from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
from bfcl_eval.model_handler.api_inference.openai_completion import (
    OpenAICompletionsHandler,
)

OLLAMA_BASE_URL = os.getenv("BFCL_OLLAMA_BASE_URL", "http://localhost:11434/v1")

# Hard cap on generated tokens per turn.
#
# Why this exists: an untrained base model asked a function-calling question often does not
# emit a parseable tool call at all. It rambles prose until it hits the generation cap, which
# made BFCL runs ~6x slower (7-9 s/entry vs 1.3 s) and eventually drove a 14 GB host into swap
# thrashing. Capping bounds the damage.
#
# Why this is fair: the cap does not bias *which* tool is called or *whether* to abstain. It
# only truncates output that was never going to be a valid tool call anyway. Verified empirically
# — see verify_max_tokens_neutral() — rescoring v1 with the cap applied reproduces its uncapped
# scores exactly. So a capped base run is comparable to the uncapped v1 run.
DEFAULT_MAX_TOKENS = int(os.getenv("BFCL_MAX_TOKENS", "512"))


class CappedOpenAICompletionsHandler(OpenAICompletionsHandler):
    """OpenAICompletionsHandler with a hard cap on generated tokens per turn.

    Identical to the upstream handler except that ``max_tokens`` is injected into every
    request. Everything else (tool compilation, chat history, response parsing) is untouched,
    so scoring remains BFCL's own code.
    """

    def __init__(self, *args, max_tokens: int = DEFAULT_MAX_TOKENS, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_tokens = max_tokens

    def _query_FC(self, inference_data: dict):
        kwargs = {
            "messages": inference_data["message"],
            "model": self.model_name,
            "temperature": self.temperature,
            "store": False,
            "max_tokens": self.max_tokens,
        }
        if len(inference_data["tools"]) > 0:
            kwargs["tools"] = inference_data["tools"]
        inference_data["inference_input_log"] = {
            "message": repr(inference_data["message"]),
            "tools": inference_data["tools"],
        }
        return self.generate_with_backoff(**kwargs)

# Ollama tag -> (display name, registry key, note shown in the model card)
LOCAL_MODELS = {
    "qwen3-4b-tools-q4km": (
        "Qwen3-4B-Tools-v1 (ours)",
        "qwen3-4b-tools-v1",
        "LoRA QLoRA fine-tune of Qwen3-4B for calibrated tool use",
    ),
    "qwen3-4b-base-q4km": (
        "Qwen3-4B-Base (untrained)",
        "qwen3-4b-base",
        "Untrained Qwen3-4B control, same quantisation as ours",
    ),
    "qwen3:8b": (
        "Qwen3-8B (reference)",
        "qwen3-8b-ref",
        "Untouched 2x-larger model, reference point",
    ),
}

# Additional Ollama tags discovered at runtime and registered on demand, so a freshly
# merged seed adapter can be scored without editing this file. Key = registry name.
EXTRA_MODELS: dict[str, tuple[str, str, str]] = {}


def register_extra(ollama_tag: str, registry_key: str, display: str, note: str = "") -> None:
    """Register one more Ollama-served model under `registry_key`."""
    EXTRA_MODELS[registry_key] = (ollama_tag, display, note)
    register_models()


def iter_all_models() -> dict:
    """Every model BFCL should know about: the fixed set plus any runtime extras."""
    models = dict(LOCAL_MODELS)
    for key, (tag, display, note) in EXTRA_MODELS.items():
        models[tag] = (display, key, note)
    return models


def register_models(
    base_url: Optional[str] = None,
    handler=None,
) -> None:
    """Insert our Ollama-hosted models into BFCL's model registry.

    Args:
        base_url: OpenAI-compatible endpoint. Defaults to Ollama's.
        handler: handler class to use. Defaults to CappedOpenAICompletionsHandler, which
            bounds generation length. Pass OpenAICompletionsHandler for the uncapped upstream
            behaviour (needed to reproduce the original v1 numbers).
    """
    url = base_url or OLLAMA_BASE_URL
    handler = handler or CappedOpenAICompletionsHandler

    # Extra seeds: BFCL_EXTRA_MODELS="registry_key=ollama_tag[|display],..."
    # Needed because BFCL validates a model name against the registry *before* it creates
    # any result directory, so a freshly merged seed cannot be auto-discovered from disk.
    for spec in filter(None, os.getenv("BFCL_EXTRA_MODELS", "").split(",")):
        spec = spec.strip()
        if not spec:
            continue
        parts = spec.split("=")
        key = parts[0].strip()
        rest = parts[1].split("|") if len(parts) > 1 else []
        tag = rest[0].strip() if rest else key
        display = rest[1].strip() if len(rest) > 1 else key.replace("_", "-")
        EXTRA_MODELS.setdefault(key, (tag, display, "registered via BFCL_EXTRA_MODELS"))

    for ollama_tag, (display, key, note) in iter_all_models().items():
        MODEL_CONFIG_MAPPING[key] = ModelConfig(
            model_name=ollama_tag,  # what we send in the `model` field
            display_name=display,
            url=url,
            org="local (Ollama)",
            license="Apache-2.0",
            model_handler=handler,
            is_fc_model=True,  # native function calling via the `tools` field
            # OpenAI-compatible APIs forbid '.' in function names
            # (^[a-zA-Z0-9_-]{1,64}$), so the model MUST call e.g.
            # 'GeometryPresentation_createPresentation'. Without this flag BFCL's
            # checker compares against the dotted 'Class.method' and scores 3%.
            underscore_to_dot=True,
        )

    # The OpenAI handler reads these at construction time.
    os.environ["OPENAI_BASE_URL"] = url
    os.environ.setdefault("OPENAI_API_KEY", "ollama")  # unused by Ollama, SDK requires it


if __name__ == "__main__":
    register_models()
    for k, v in MODEL_CONFIG_MAPPING.items():
        if k in {m[1] for m in iter_all_models().values()}:
            print(
                f"{k:22s} -> {v.display_name}  "
                f"(fc={v.is_fc_model}, handler={v.model_handler.__name__}, "
                f"max_tokens={DEFAULT_MAX_TOKENS}, base_url={v.url})"
            )