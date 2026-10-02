#!/usr/bin/env python3
"""Normalize v2 data to a strict schema for the HF dataset loader:

- tools: ALWAYS a list; every tool = {type, function:{name, description, parameters}}
  (parameters always a JSON-Schema object with properties dict)
- messages: content always str; tool_calls always [{type, function:{name, args dict}}]
- verifies with datasets.Dataset.from_list

Output: data/train_final.jsonl, data/val_final.jsonl
"""
import json
from pathlib import Path
from collections import Counter

from datasets import Dataset

BASE = Path.home() / "model training"


def norm_tool(t):
    fn = t.get("function", t) if isinstance(t, dict) else {}
    params = fn.get("parameters")
    if not isinstance(params, dict):
        params = {"type": "object", "properties": {}}
    if not isinstance(params.get("properties"), dict):
        params["properties"] = {}
    return {"type": "function", "function": {
        "name": str(fn.get("name") or "unknown_function"),
        "description": str(fn.get("description") or ""),
        "parameters": params,
    }}


def norm_record(r):
    tools = r.get("tools") or []
    tools = [norm_tool(t) for t in tools if isinstance(t, dict) and (t.get("name") or t.get("function"))]
    # dedupe by name (keep first)
    seen, uniq = set(), []
    for t in tools:
        n = t["function"]["name"]
        if n not in seen:
            seen.add(n)
            uniq.append(t)
    messages = []
    for m in r.get("messages", []):
        role = m.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            continue
        msg = {"role": role, "content": str(m.get("content") or "")}
        calls = m.get("tool_calls") or []
        if calls:
            norm_calls = []
            for c in calls:
                fn = c.get("function", {}) if isinstance(c, dict) else {}
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        parsed = json.loads(args)
                        args = parsed if isinstance(parsed, dict) else {"value": parsed}
                    except json.JSONDecodeError:
                        args = {"raw": args}
                if not isinstance(args, dict):
                    args = {}
                # arguments stored as a JSON STRING: pyarrow-safe and the TRL
                # Qwen3 template inserts string arguments verbatim (exact JSON)
                norm_calls.append({"type": "function", "function": {
                    "name": str(fn.get("name") or "unknown_function"),
                    "arguments": json.dumps(
                        {str(k): v for k, v in args.items()}, ensure_ascii=False),
                }})
            msg["tool_calls"] = norm_calls
        messages.append(msg)
    return {"messages": messages,
            # tools as a JSON string: TRL json.loads() it; avoids pyarrow
            # struct-mixing across differently-shaped tool schemas
            "tools": json.dumps(uniq, ensure_ascii=False),
            "source": str(r.get("source") or "?"),
            "n_calls": int(sum(len(m.get("tool_calls") or []) for m in messages))}


def main():
    for split in ("train", "val"):
        src = BASE / "data" / f"{split}_v2.jsonl"
        dst = BASE / "data" / f"{split}_final.jsonl"
        out = []
        with open(src) as f:
            for line in f:
                out.append(norm_record(json.loads(line)))
        with open(dst, "w") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        # verify with from_list (the trainer reads the jsonl via datasets)
        ds = Dataset.from_list(out)
        print(f"{split}: {len(out)} rows written + from_list OK | "
              f"sources={dict(Counter(r['source'] for r in out))}")
        print(f"  with tools: {sum(1 for r in out if r['tools'] != '[]')} | "
              f"with calls: {sum(1 for r in out if r['n_calls']>0)}")


if __name__ == "__main__":
    main()
