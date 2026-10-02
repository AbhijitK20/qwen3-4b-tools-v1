#!/usr/bin/env python3
"""Download additional tool-calling datasets (the network-dependent step).

Run while internet is available. Everything lands in the local HF cache so
later prep/pretokenize steps can run fully offline.

Sources:
  ToolACE        full data.json (already partial before; re-fetch to confirm)
  When2Call      train SFT + MCQ splits (previous run used only 300 val rows)
  Hermes v1      func-calling + json-mode-agentic (cached but NEVER used in
                 training - the biggest free win available)
  xlam-60k       gated behind a licence click; skipped unless HF_TOKEN is set
  SWE-Gym        agentic multi-step tool use

xlam is gated="auto": it needs an HF token that has accepted the licence.
Without HF_TOKEN we skip it and report that, rather than silently producing an
empty cache dir the way a previous run did.
"""
import os
import sys
import time
from pathlib import Path

BASE = Path.home() / "model training"
os.environ.setdefault("HF_HOME", str(BASE / "hf_cache"))

from huggingface_hub import snapshot_download  # noqa: E402

# repo_id -> (patterns, required?)
TARGETS = [
    ("Team-ACE/ToolACE", ["data.json"], True),
    ("nvidia/When2Call", ["*.jsonl"], True),
    ("NousResearch/hermes-function-calling-v1", ["*.json"], True),
    ("SWE-Gym/SWE-Gym", ["*.jsonl"], False),
    ("Salesforce/xlam-function-calling-60k",
     ["xlam_function_calling_60k.json"], True),
]


def main():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    results = {}
    for repo, patterns, required in TARGETS:
        t0 = time.time()
        try:
            p = snapshot_download(repo_id=repo, repo_type="dataset",
                                  allow_patterns=patterns, token=token,
                                  max_workers=4)
            size = sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file())
            results[repo] = (True, size, p)
            print(f"[ok]   {repo:52s} {size/1e6:7.1f} MB  ({time.time()-t0:.0f}s)")
        except Exception as e:  # noqa: BLE001
            msg = str(e)[:120].replace("\n", " ")
            gated = "gated" in msg.lower() or "403" in msg
            results[repo] = (False, 0, msg)
            tag = "GATED" if gated else "FAIL"
            print(f"[{tag}] {repo:52s} {msg}")
            if gated and not token:
                print(f"       -> set HF_TOKEN with a licence-approved account to fetch this")
    return results


if __name__ == "__main__":
    res = main()
    print()
    ok = [r for r, (s, _, _) in res.items() if s]
    bad = [r for r, (s, _, _) in res.items() if not s]
    print(f"fetched {len(ok)}/{len(res)}")
    for r in bad:
        print(f"  missing: {r}")
    sys.exit(0)