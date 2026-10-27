"""Fetch the JailbreakHub in-the-wild jailbreak prompt corpus (TrustAIRLab).

Source: "Do Anything Now": Characterizing and Evaluating In-The-Wild Jailbreak
Prompts on LLMs, ACM CCS 2024 (arXiv:2308.03825). MIT licensed. 15,140 prompts
collected Dec 2022 - Dec 2023, of which 1,405 are labelled as jailbreaks.

Why this and not the 20-prompt set in `strict_probe.py`: the current probe draws
from a static file of hand-assembled harmful requests, so it measures whether a
model refuses *that* set. These are real attacks harvested in the wild, including
the roleplay/DAN/persona framings that a plain harmful-request prompt does not
exercise. The benign subsets (regular_*) are the matched control -- they let a
refusal rate be read against a false-positive rate on the same distribution,
which the current probe cannot do at all.

Writes to data/hf/in-the-wild-jailbreak-prompts/ and prints the shape of what
landed. Deliberately does not write a training file: this corpus is an *eval*
input. Whether any slice becomes training data is a separate, explicit decision.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

from huggingface_hub import snapshot_download

REPO_ID = "TrustAIRLab/in-the-wild-jailbreak-prompts"
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DEST = os.path.join(HERE, "..", "data", "hf", REPO_ID.split("/")[-1])

# The corpus ships four subsets. Only the two `jailbreak_*` subsets are labelled
# attacks; the two `regular_*` subsets are benign prompts from the same platforms
# and same time window, used here as the false-positive control.
SUBSETS = (
    "jailbreak_2023_05_07",
    "jailbreak_2023_12_25",
    "regular_2023_05_07",
    "regular_2023_12_25",
)


def fetch(dest: str) -> str:
    # The corpus ships as parquet, one file per subset. The parquet-converted refs
    # are NOT wanted here -- they are the dataset-viewer's own rendering and would
    # give a second copy of the same rows.
    return snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=dest,
        allow_patterns=["*.parquet", "*.md"],
    )


def _read(path: str) -> list[dict]:
    """Read a data file into a list of dicts, tolerating parquet/csv/json/jsonl."""
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if path.endswith(".json"):
        payload = json.load(open(path, encoding="utf-8"))
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for value in payload.values():
                if isinstance(value, list):
                    return value
            return []
        return []
    if path.endswith(".csv"):
        import csv

        with open(path, encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    return []


def summarize(dest: str) -> dict:
    """Read whatever the snapshot produced and describe it without assuming a format."""
    found = sorted(
        os.path.join(root, name)
        for root, _, names in os.walk(dest)
        for name in names
        if name.endswith((".parquet", ".csv", ".json", ".jsonl"))
    )
    print("\n=== files ===")
    for path in found:
        size = os.path.getsize(path)
        print(f"  {os.path.relpath(path, dest):60s} {size/1024:8.1f} KB")

    summary: dict = {"files": [os.path.relpath(p, dest) for p in found]}

    # Count rows and label balance per file, tolerating csv/json/jsonl.
    for path in found:
        rel = os.path.relpath(path, dest)
        try:
            rows = _read(path)
        except Exception as error:  # noqa: BLE001 - report, do not abort the fetch
            print(f"\n  [skip {rel}: {type(error).__name__} {error}]")
            continue

        if not rows:
            continue
        keys = sorted(rows[0].keys())
        print(f"\n=== {rel} ===")
        print(f"  rows    {len(rows)}")
        print(f"  columns {keys}")

        label_key = next((k for k in ("jailbreak", "label", "is_jailbreak") if k in rows[0]), None)
        if label_key:
            counts = Counter(str(r.get(label_key)) for r in rows)
            print(f"  {label_key}: {dict(counts)}")

        for key in ("platform", "source", "community_name"):
            if key in rows[0]:
                counts = Counter(str(r.get(key)) for r in rows).most_common(6)
                print(f"  {key}: {counts}")
                break

        prompt_key = next((k for k in ("prompt", "text", "jailbreak_prompt") if k in rows[0]), None)
        if prompt_key:
            lengths = sorted(len(str(r.get(prompt_key) or "")) for r in rows)
            median = lengths[len(lengths) // 2]
            print(f"  {prompt_key} chars: median {median}, p95 {lengths[int(len(lengths)*0.95)]}, max {lengths[-1]}")

            # Exact-duplicate rate matters a lot: this corpus was scraped from
            # forums where the same persona prompt gets reposted constantly, and a
            # duplicate-heavy eval set understates its own effective sample size.
            texts = [str(r.get(prompt_key) or "").strip() for r in rows]
            unique = len({t for t in texts if t})
            print(f"  unique prompts: {unique}/{len(texts)} ({100*unique/max(1,len(texts)):.1f}%)")

        summary[rel] = {"rows": len(rows), "columns": keys, "label_key": label_key}

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default=DEFAULT_DEST)
    parser.add_argument("--summary", default=None, help="optional path to write the summary JSON")
    args = parser.parse_args()

    dest = os.path.abspath(args.dest)
    print(f"fetching {REPO_ID} -> {dest}")
    path = fetch(dest)
    print(f"ok: {path}")

    summary = summarize(dest)
    summary["source"] = REPO_ID
    summary["paper"] = "arXiv:2308.03825 (ACM CCS 2024)"
    summary["license"] = "MIT"
    summary["subsets_expected"] = list(SUBSETS)

    if args.summary:
        with open(args.summary, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"\nwrote {args.summary}")


if __name__ == "__main__":
    main()