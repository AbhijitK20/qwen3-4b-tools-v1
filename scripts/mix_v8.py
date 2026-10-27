"""Tokenize the injection-hardening slice and mix it with tok_train_v3.

Separate from pretokenize.py because that script reads data/train_<v>.jsonl,
and the v3 raw jsonl has been cleaned up (it is regenerable). tok_train_v3
survives on disk already tokenized with the same template, so re-tokenizing
21,905 rows to add 600 would be wasted GPU time.

To stay consistent it must use the *same* three things pretokenize.py used,
otherwise the two halves of the mixture are not in the same format and the
model learns two different renderings of a tool call:

  - the TRL qwen3_training.jinja template (the one with {% generation %}, so
    assistant masks come back and labels can be -100 everywhere else)
  - the same tokenizer snapshot (unsloth--Qwen3-4B-unsloth-bnb-4bit)
  - the same MAX_LEN

Mix ratio is the other learned lesson. The repo's README records that v5 added
prose-heavy compliance data at 22% and lost tool-calling. This slice is
tool-call-shaped throughout, so a different ratio applies, but the conservative
default is still small: --hardening-frac 0.03. If tool-call accuracy drops,
the fix is fewer steps, not a bigger slice.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_from_disk
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
BASE = HERE.parent

_TPL_CANDIDATES = [
    BASE / "repos/trl/trl/chat_templates/qwen3_training.jinja",
    BASE / ".venv/lib/python3.12/site-packages/trl/chat_templates/qwen3_training.jinja",
]
TEMPLATE = next(p for p in _TPL_CANDIDATES if p.exists())
SLUG = "unsloth--Qwen3-4B-unsloth-bnb-4bit"
MAX_LEN = 2048  # matches pretokenize.py's TOK_MAX_LEN default


def snapshot_dir() -> Path:
    snaps = sorted((BASE / f"hf_cache/hub/models--{SLUG}/snapshots").glob("*"))
    if not snaps:
        raise SystemExit(f"tokenizer snapshot not found under hf_cache/hub/models--{SLUG}")
    return snaps[-1]


def normalise(rec: dict) -> dict:
    """Flatten tool_calls into a JSON-string column, exactly as pretokenize.py does."""
    blob = []
    for message in rec.get("messages") or []:
        calls = message.get("tool_calls")
        if isinstance(calls, list) and calls:
            norm = []
            for call in calls:
                fn = dict(call.get("function") or {})
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                fn["arguments"] = args if isinstance(args, dict) else {}
                norm.append({"function": fn})
            blob.append(json.dumps(norm))
        else:
            blob.append("")
    rec = dict(rec)
    rec["_tc"] = json.dumps(blob)
    rec["tools"] = json.dumps(rec["tools"]) if rec.get("tools") else None
    return rec


def tokenize(tok, rec: dict) -> dict | None:
    blob = json.loads(rec.get("_tc") or "[]")
    iterator = iter(blob)
    messages = []
    for message in rec["messages"]:
        message = dict(message)
        raw = next(iterator, "")
        if raw:
            message["tool_calls"] = json.loads(raw)
        messages.append(message)

    tools = rec.get("tools")
    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
        except json.JSONDecodeError:
            tools = None

    out = tok.apply_chat_template(
        messages,
        tools=tools or None,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    input_ids = out["input_ids"]
    mask = out["assistant_masks"]
    if len(input_ids) > MAX_LEN:
        return None
    labels = [token if flag else -100 for token, flag in zip(input_ids, mask)]
    if all(label == -100 for label in labels):
        return None
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "source": rec.get("source"),
        "n_calls": rec.get("n_calls"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default=str(BASE / "data" / "train_v8_injection.jsonl"))
    parser.add_argument("--tool-dataset", default=str(BASE / "data" / "tok_train_v3"))
    parser.add_argument("--hardening-frac", type=float, default=0.03,
                        help="hardening rows as a fraction of the tool slice")
    parser.add_argument("--out", default=str(BASE / "data" / "tok_train_v8"))
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(str(snapshot_dir()))
    tok.chat_template = TEMPLATE.read_text()
    print("template has {% generation %}:",
          "{% generation %}" in tok.chat_template or "{%- generation %}" in tok.chat_template)

    records = [json.loads(l) for l in Path(args.rows).read_text(encoding="utf-8").splitlines() if l.strip()]
    records = [normalise(r) for r in records]
    tokenized = [t for t in (tokenize(tok, r) for r in records) if t]
    dropped = len(records) - len(tokenized)
    print(f"hardening rows {len(records)} -> tokenized {len(tokenized)} (dropped {dropped}: over length or no assistant tokens)")

    hard_ds = Dataset.from_list(tokenized)

    tool_ds = load_from_disk(args.tool_dataset)
    target = max(1, int(len(hard_ds) / max(args.hardening_frac, 1e-6)))
    if target < len(tool_ds):
        tool_ds = tool_ds.shuffle(seed=1234).select(range(target))
    print(f"tool slice     {len(tool_ds)} rows sampled from {len(load_from_disk(args.tool_dataset))} "
          f"(target {target} at frac {args.hardening_frac})")

    mixed = concatenate_datasets([tool_ds, hard_ds]).shuffle(seed=1234)
    mixed.save_to_disk(args.out)
    print(f"\nwrote {len(mixed)} rows -> {args.out}")
    print(f"  tool {len(tool_ds)} ({100*len(tool_ds)/len(mixed):.1f}%)  hardening {len(hard_ds)} ({100*len(hard_ds)/len(mixed):.1f}%)")


if __name__ == "__main__":
    main()