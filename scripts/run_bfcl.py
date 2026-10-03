"""
Run the Berkeley Function Calling Leaderboard against Ollama-served models.

This patches our models into BFCL's registry (see bfcl_ollama.py) and then
hands off to BFCL's own CLI, so generation and scoring are BFCL's code, not
ours. Results land in BFCL's standard score/ dir and are reproducible with
`bfcl evaluate`.

Examples
--------
    # smoke test - 1 category, 20 entries
    python scripts/run_bfcl.py generate --category simple --run-ids 1:20

    # main comparison: ours vs base vs 8B on the core single-turn categories
    python scripts/run_bfcl.py generate \
        --model qwen3-4b-tools-v1 --model qwen3-4b-base --model qwen3-8b-ref \
        --category simple --category parallel --category multiple

    # score whatever was generated
    python scripts/run_bfcl.py evaluate --model qwen3-4b-tools-v1
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from bfcl_eval.__main__ import cli  # noqa: E402
from bfcl_ollama import iter_all_models, register_models  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)

    register_models()

    # Extra seeds (e.g. freshly merged replicates) are registered via BFCL_EXTRA_MODELS,
    # e.g. BFCL_EXTRA_MODELS="qwen3_4b_s1337=qwen3-4b-s1337|Qwen3-4B seed 1337".
    # Disk auto-discovery cannot work: BFCL validates the model name before it creates
    # any result directory.

    # Let BFCL's own CLI parse everything after the subcommand.
    if sys.argv[1] not in {"generate", "evaluate", "scores", "models"}:
        print(f"unknown subcommand: {sys.argv[1]}")
        raise SystemExit(2)

    cli()


def _known_keys() -> set:
    return {v[1] for v in iter_all_models().values()}


if __name__ == "__main__":
    main()