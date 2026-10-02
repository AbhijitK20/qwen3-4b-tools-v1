#!/usr/bin/env python3
"""Pre-tokenize the tool-calling dataset for the Unsloth pre-tokenized path.

Why: Unsloth 2026.9.12's compiled SFTTrainer (2026.9) does not implement TRL
0.24's conversational (messages + tools) tokenization path, so we tokenize here
and hand the trainer input_ids/attention_mask/labels. It also lets us control
assistant-only masking precisely.

- Uses TRL's qwen3_training.jinja (contains {% generation %}) so
  apply_chat_template can return assistant token masks.
- labels = token id where the assistant mask is 1, else -100.
- Examples with no assistant tokens are dropped.
- Output: data/tok_train, data/tok_val (arrow datasets on disk).
"""
import json
import os
from pathlib import Path

from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer

BASE = Path.home() / "model training"
# TRL's Qwen3 *training* template (the one that adds {% generation %} so
# assistant token masks can be returned). The installed wheel ships no
# chat_templates dir, so use the cloned TRL repo copy.
_TPL_CANDIDATES = [
    BASE / "repos/trl/trl/chat_templates/qwen3_training.jinja",
    BASE / ".venv/lib/python3.12/site-packages/trl/chat_templates/qwen3_training.jinja",
]
TEMPLATE = next(p for p in _TPL_CANDIDATES if p.exists())
SLUG = "unsloth--Qwen3-4B-unsloth-bnb-4bit"
MAX_LEN = int(os.environ.get("TOK_MAX_LEN", "2048"))

snap = sorted((BASE / f"hf_cache/hub/models--{SLUG}/snapshots").glob("*"))[-1]


def tools_of(rec):
    """Return a record's tools as a list, tolerating the v1/v2 JSON-string form."""
    t = rec.get("tools")
    if isinstance(t, str):
        try:
            t = json.loads(t)
        except json.JSONDecodeError:
            return None
    return t or None


def main():
    tok = AutoTokenizer.from_pretrained(str(snap))
    tok.chat_template = TEMPLATE.read_text()
    print("template has {% generation %}:", "{% generation %}" in tok.chat_template
          or "{%- generation %}" in tok.chat_template)

    # SRC selects the corpus version. v3 is the augmented mix (abstention-heavy);
    # set TOK_SPLIT_PREFIX=final to reproduce the v1 run.
    prefix = os.environ.get("TOK_SPLIT_PREFIX", "v3")
    for split, src in ((f"train_{prefix}", f"train_{prefix}.jsonl"),
                       (f"val_{prefix}", f"val_{prefix}.jsonl")):
        out_dir = BASE / f"data/tok_{split}"
        if out_dir.exists():
            print(f"skip {split} (exists)")
            continue
        recs = [json.loads(l) for l in (BASE / "data" / src).read_text().splitlines() if l.strip()]
        # Normalize message shape BEFORE handing to Arrow. Mixed keys across
        # rows (some messages carry tool_calls, some do not) make pyarrow infer
        # a list column then hit "cannot mix list and non-list, non-null".
        # An explicit null keeps the column typed; the chat template ignores it.
        # pyarrow cannot type a nested column whose first element lacks a key
        # that later elements have ("cannot mix list and non-struct, non-null"
        # / "cannot mix list and non-list, non-null values"). A conversation
        # normally starts with a user turn (no tool_calls) and hits an
        # assistant turn later (with tool_calls), so this fires constantly on
        # the v3 corpus. Fix: flatten tool_calls into a JSON STRING column
        # before Arrow sees the record, then re-attach inside the mapper.
        for r in recs:
            blob = []
            for m in r.get("messages") or []:
                tc = m.pop("tool_calls", None)
                if isinstance(tc, list) and tc:
                    norm = []
                    for c in tc:
                        fn = dict(c.get("function") or {})
                        a = fn.get("arguments")
                        if isinstance(a, str):
                            try:
                                a = json.loads(a)
                            except json.JSONDecodeError:
                                a = {}
                        if not isinstance(a, dict):
                            a = {}
                        fn["arguments"] = a
                        norm.append({"function": fn})
                    blob.append(json.dumps(norm))
                else:
                    blob.append("")
            r["_tc"] = json.dumps(blob)
            r["tools"] = json.dumps(tools_of(r)) if tools_of(r) else None
        ds = Dataset.from_list(recs)

        def tok_one(rec):
            # tools was stringified above; restore it for the chat template.
            tools = rec.get("tools")
            if isinstance(tools, str):
                try:
                    tools = json.loads(tools)
                except json.JSONDecodeError:
                    tools = None
            blob = json.loads(rec.get("_tc") or "[]")
            it = iter(blob)
            full = []
            for m in rec["messages"]:
                m = dict(m)
                raw = next(it, "")
                if raw:
                    m["tool_calls"] = json.loads(raw)
                full.append(m)
            out = tok.apply_chat_template(
                full,
                tools=tools or None,
                tokenize=True,
                return_dict=True,
                return_assistant_tokens_mask=True,
                truncation=True,
                max_length=MAX_LEN,
            )
            ids = out["input_ids"]
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            am = out.get("assistant_masks")
            if am is None:
                return None
            if am and isinstance(am[0], list):
                am = am[0]
            if 1 not in (am or []):
                return None
            return {
                "input_ids": ids,
                "attention_mask": [1] * len(ids),
                "labels": [i if a else -100 for i, a in zip(ids, am)],
            }

        ds = ds.map(tok_one, remove_columns=ds.column_names,
                    desc=f"pretokenize {split}", num_proc=4)
        # tok_one returns None for examples with no supervised assistant
        # tokens; datasets keeps them as null rows. Drop them before computing
        # stats, otherwise len(None) raises.
        n_before = len(ds)
        ds = ds.filter(lambda r: r["input_ids"] is not None,
                       num_proc=4, desc="drop empty")
        n = len(ds)
        lens = [len(x) for x in ds["input_ids"]]
        sup_labs = [lab for lab in ds["labels"][:2000] if lab is not None]
        train_tok = (sum(sum(1 for v in lab if v != -100) for lab in sup_labs)
                     / max(len(sup_labs), 1))
        print(f"{split}: {n} examples (dropped {n_before - n} with no assistant "
              f"tokens) | mean len {sum(lens)/max(n,1):.0f} | "
              f"max {max(lens) if lens else 0} | mean supervised tokens {train_tok:.0f}")
        ds.save_to_disk(str(out_dir))
        print(f"  -> {out_dir}")


if __name__ == "__main__":
    main()
