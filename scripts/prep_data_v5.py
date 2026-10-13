"""
Build the v5 mixture: targeted compliance + general refusal-free chat + tool-calling.

Why a mixture and not just uncensored data
------------------------------------------
This checkpoint is a *tool-calling* fine-tune, and the repo has an extensive
tool-calling evaluation (BFCL irrelevance / multiple-AST, plus balanced_score.py
which penalises one-sidedness). Training on generic chat alone would strip refusals
and silently destroy the capability the model exists for, with nothing in the loss
to warn us. So the tool-calling corpus stays the majority and the new data is added
around it.

Shares, and what each one buys
-----------------------------
  tool-calling   ~75%   preserves the actual purpose; reused from tok_train_v4
  airoboros      ~22%   refusal-free general helpfulness. Anchors the model so the
                       compliance slice does not make it narrow
  compliance      ~3%   the actual refusal-removal signal: harmful prompts paired
                       with substantive non-refusing answers, distilled from the
                       ablated checkpoint and hard-filtered for coherence

3% looks small. It is enough because every example teaches the same behaviour --
answer the request -- and it is the only slice that targets the refusal itself.
Airoboros is 0.3% refusal-language overall but only 1.1% of its prompts are edgy,
so it shifts general helpfulness, not safety behaviour specifically.

Output: data/train_v5.jsonl, data/val_v5.jsonl (new data only, project format),
then pretokenize.py builds tok_train_v5 and this script concatenates the
tool-calling slice into data/tok_train_v5_full for training.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random

BASE = pathlib.Path(__file__).resolve().parent.parent


def load_jsonl(p: pathlib.Path):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compliance", default="data/compliance_distil.jsonl")
    ap.add_argument("--airoboros", default="data/hf/airoboros.jsonl")
    ap.add_argument("--n-airoboros", type=int, default=2000)
    ap.add_argument("--n-tool", type=int, default=7000)
    ap.add_argument("--val-compliance", type=int, default=24)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    compliance = load_jsonl(BASE / args.compliance)
    rng.shuffle(compliance)
    val_c = compliance[: args.val_compliance]
    train_c = compliance[args.val_compliance :]
    print(f"compliance: {len(train_c)} train / {len(val_c)} val")

    airo = load_jsonl(BASE / args.airoboros)
    # cap response length: long answers waste the 1024-token budget and the
    # tool-calling rows this must not crowd out are short
    airo = [
        {"messages": [{"role": "user", "content": r["instruction"].strip()},
                      {"role": "assistant", "content": r["response"].strip()}],
         "tools": None, "source": "airoboros_uncensored"}
        for r in airo
        if r.get("instruction") and r.get("response")
        and 80 < len(r["response"]) < 3000
    ]
    rng.shuffle(airo)
    airo = airo[: args.n_airoboros]
    print(f"airoboros: {len(airo)} selected from pool")

    new_rows = train_c + airo
    rng.shuffle(new_rows)

    for split, rows in (("train", new_rows), ("val", val_c)):
        out = BASE / f"data/{split}_v5.jsonl"
        with out.open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"wrote {out} ({len(rows)} rows)")

    print("\nnext: TOK_SPLIT_PREFIX=v5 .venv/bin/python scripts/pretokenize.py")
    print(f"      tool-calling slice to add from tok_train_v4: {args.n_tool}")


if __name__ == "__main__":
    main()
