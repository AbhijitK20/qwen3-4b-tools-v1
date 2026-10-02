#!/usr/bin/env python3
"""Render train/val jsonl into Qwen3 chat-template text with a length filter.

Usage: python3 scripts/render_dataset.py --max-tokens 2048
Output: data/train_rendered.jsonl, data/val_rendered.jsonl  (fields: text, source, n_tokens)
"""
import argparse
import glob
import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

BASE = Path.home() / "model training"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=2048)
    args = ap.parse_args()

    tok_path = glob.glob(str(BASE / "hf_cache/hub/models--unsloth--Qwen3-8B-unsloth-bnb-4bit/snapshots/*/"))[0]
    tok = AutoTokenizer.from_pretrained(tok_path)
    print(f"tokenizer loaded from {tok_path}")

    for split in ("train", "val"):
        src = BASE / "data" / f"{split}.jsonl"
        dst = BASE / "data" / f"{split}_rendered.jsonl"
        kept = dropped = 0
        kept_sources = Counter()
        lengths = []
        with open(src) as fin, open(dst, "w") as fout:
            for line in fin:
                rec = json.loads(line)
                try:
                    text = tok.apply_chat_template(
                        rec["messages"], tools=rec.get("tools"), tokenize=False)
                except Exception:
                    dropped += 1
                    continue
                n = len(tok(text, return_tensors=None)["input_ids"])
                if n > args.max_tokens:
                    dropped += 1
                    continue
                fout.write(json.dumps({"text": text,
                                       "source": rec.get("source", "?"),
                                       "n_tokens": n}) + "\n")
                kept += 1
                kept_sources[rec.get("source", "?")] += 1
                lengths.append(n)
        import statistics
        print(f"{split}: kept={kept} dropped={dropped} "
              f"(median {statistics.median(lengths):.0f} tok, max {max(lengths)})")
        print(f"  sources kept: {dict(kept_sources)}")


if __name__ == "__main__":
    main()
