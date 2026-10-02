#!/usr/bin/env python3
"""Download the fp16 Qwen3-4B base (the only step in this project that needs network).

Why fp16 and not the bnb-4bit we train on:
  the adapter is merged into the base before export. Merging into the 4-bit
  base and then quantizing to GGUF quantizes twice, which costs real accuracy
  on exact-argument JSON — exactly what tool calling is scored on. Merging
  into fp16 and quantizing once is the correct path.

Safe to re-run: snapshot_download resumes and skips files already present.
"""
import sys
import time
from pathlib import Path

BASE = Path.home() / "model training"

# train_qlora_v2.py resolves the base via
#   glob(BASE / f"hf_cache/hub/models--{slug}/snapshots/*/")[0]
# so the destination must match that layout exactly.
REPO = "unsloth/Qwen3-4B"
DEST = BASE / "hf_cache" / "hub" / "models--unsloth--Qwen3-4B" / "snapshots" / "fp16"


def main():
    from huggingface_hub import snapshot_download

    DEST.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"[dl] {REPO} -> {DEST}", flush=True)
    for attempt in range(1, 4):
        try:
            snapshot_download(
                repo_id=REPO,
                local_dir=str(DEST),
                allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"],
                max_workers=4,
            )
            print(f"[dl] done in {time.time() - t0:.0f}s", flush=True)
            return 0
        except Exception as e:  # noqa: BLE001
            print(f"[dl] attempt {attempt} failed: {e}", file=sys.stderr, flush=True)
            if attempt < 3:
                time.sleep(20)
    print("[dl] FAILED after 3 attempts", file=sys.stderr, flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
