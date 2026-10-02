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


def register_models(base_url: Optional[str] = None) -> None:
    """Insert our Ollama-hosted models into BFCL's model registry."""
    url = base_url or OLLAMA_BASE_URL

    for ollama_tag, (display, key, note) in LOCAL_MODELS.items():
        MODEL_CONFIG_MAPPING[key] = ModelConfig(
            model_name=ollama_tag,  # what we send in the `model` field
            display_name=display,
            url=url,
            org="local (Ollama)",
            license="Apache-2.0",
            model_handler=OpenAICompletionsHandler,
            is_fc_model=True,  # native function calling via the `tools` field
        )

    # The OpenAI handler reads these at construction time.
    os.environ["OPENAI_BASE_URL"] = url
    os.environ.setdefault("OPENAI_API_KEY", "ollama")  # unused by Ollama, SDK requires it


if __name__ == "__main__":
    register_models()
    for k, v in MODEL_CONFIG_MAPPING.items():
        if k in {m[1] for m in LOCAL_MODELS.values()}:
            print(f"{k:22s} -> {v.display_name}  (fc={v.is_fc_model}, base_url={v.url})")