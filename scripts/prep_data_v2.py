#!/usr/bin/env python3
"""V2 data pipeline — applies repo-researched techniques:

1. call VERIFICATION (ToolACE DLV / Hermes validator style):
   name must be one of the sample's tools; required params present; types match
2. dedupe by normalized conversation hash
3. per-tool frequency cap (glaive is templated and repetitive)
4. source rebalancing: glaive capped at ~2.5x the verified sources
   (measured: raw glaive = 80% of tokens, crowds out ToolACE/When2Call)
5. call/no-call balance: keep no-call samples at ~12% of the mix
   (Hammer ablation: ~10% irrelevance augmentation is optimal)

Output: data/train_v2.jsonl, data/val_v2.jsonl
Each line: {"messages": [...], "tools": [...]|null, "source": "...", "n_calls": k}
"""
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path.home() / "model training"
random.seed(42)

GLAIVE_CAP_MULT = 2.5          # glaive kept at most 2.5x the other sources combined
TOOL_FREQ_CAP = 2500           # max samples containing a given tool name
NOCALL_FRACTION = 0.12         # target share of no-call samples in the final mix


def norm_args(a):
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except json.JSONDecodeError:
            return None
    return a if isinstance(a, dict) else None


def verify_sample(rec):
    """Return (ok, reason). A sample is valid when every tool call names a
    declared tool, all required params are present, and arg types agree."""
    msgs = rec["messages"]
    tools = rec.get("tools") or []
    schema = {}
    for t in tools:
        fn = t.get("function", {})
        schema[fn.get("name")] = fn.get("parameters", {}) or {}

    calls_found = 0
    for m in msgs:
        for c in m.get("tool_calls", []) or []:
            calls_found += 1
            fn = c.get("function", {})
            name = fn.get("name")
            args = norm_args(fn.get("arguments"))
            if args is None:
                return False, "bad_arguments_json"
            if tools and name not in schema:
                return False, "name_not_in_tools"
            params = schema.get(name, {}) or {}
            if not isinstance(params, dict):
                params = {}
            required = params.get("required", [])
            if not isinstance(required, list):
                required = []
            for req in required:
                if req not in args:
                    return False, "missing_required_param"
            props = params.get("properties", {})
            if not isinstance(props, dict):
                props = {}
            for k, v in args.items():
                ptype = (props.get(k) or {}).get("type")
                if ptype == "integer" and not isinstance(v, int):
                    return False, "arg_type_mismatch"
                if ptype == "number" and not isinstance(v, (int, float)):
                    return False, "arg_type_mismatch"
                if ptype == "boolean" and not isinstance(v, bool):
                    return False, "arg_type_mismatch"
    # a "tool" role message must follow at least one call somewhere
    if any(m["role"] == "tool" for m in msgs) and calls_found == 0:
        return False, "tool_response_without_call"
    return True, "ok"


def conv_hash(rec):
    parts = []
    for m in rec["messages"]:
        parts.append(m["role"] + ":" + (m.get("content") or "")[:200])
        for c in m.get("tool_calls", []) or []:
            fn = c.get("function", {})
            parts.append(f"{fn.get('name')}:{json.dumps(norm_args(fn.get('arguments')) or {}, sort_keys=True)}")
    return hashlib.md5("|".join(parts).encode()).hexdigest()


def main():
    records = []
    for split in ("train", "val"):
        with open(BASE / "data" / f"{split}.jsonl") as f:
            for line in f:
                r = json.loads(line)
                r["_split"] = split
                records.append(r)
    print(f"raw: {len(records)}")

    # --- 1. verification ---
    verified, drop_reasons = [], Counter()
    for r in records:
        ok, reason = verify_sample(r)
        if ok:
            verified.append(r)
        else:
            drop_reasons[reason] += 1
    print(f"verified: {len(verified)}  (dropped {sum(drop_reasons.values())})")
    for reason, n in drop_reasons.most_common():
        print(f"  dropped {reason}: {n}")

    # --- 2. dedupe ---
    seen, deduped = set(), []
    for r in verified:
        h = conv_hash(r)
        if h in seen:
            continue
        seen.add(h)
        deduped.append(r)
    print(f"deduped: {len(deduped)} (removed {len(verified)-len(deduped)})")

    # --- 3. per-tool frequency cap ---
    tool_count = Counter()
    capped = []
    skipped_tool_cap = 0
    order = list(range(len(deduped)))
    random.shuffle(order)
    for i in order:
        r = deduped[i]
        names = {c["function"]["name"]
                 for m in r["messages"] for c in (m.get("tool_calls") or [])}
        if names and any(tool_count[n] >= TOOL_FREQ_CAP for n in names):
            skipped_tool_cap += 1
            continue
        for n in names:
            tool_count[n] += 1
        capped.append(r)
    print(f"tool-cap: kept {len(capped)} (skipped {skipped_tool_cap}); "
          f"top tools: {tool_count.most_common(5)}")

    # --- 4/5. source rebalance + call/no-call balance ---
    by_source = defaultdict(list)
    for r in capped:
        by_source[r["source"]].append(r)
    n_non_glaive = sum(len(v) for k, v in by_source.items() if k != "glaive")
    glaive_budget = int(n_non_glaive * GLAIVE_CAP_MULT)
    glaive = by_source.pop("glaive", [])
    have_calls = [r for r in glaive if any(m.get("tool_calls") for m in r["messages"])]
    no_calls = [r for r in glaive if not any(m.get("tool_calls") for m in r["messages"])]
    random.shuffle(have_calls)
    random.shuffle(no_calls)
    # keep glaive calls first, fill the rest of the budget with no-call samples
    glaive_kept = have_calls[:glaive_budget]
    room = glaive_budget - len(glaive_kept)
    if room > 0:
        glaive_kept += no_calls[:room]
    print(f"glaive: {len(glaive)} -> {len(glaive_kept)} (budget {glaive_budget}, "
          f"others {n_non_glaive})")

    mix = glaive_kept + [r for v in by_source.values() for r in v]

    # cap no-call share of the final mix
    calls = [r for r in mix if any(m.get("tool_calls") for m in r["messages"])]
    nocalls = [r for r in mix if not any(m.get("tool_calls") for m in r["messages"])]
    max_nocall = int(len(calls) * NOCALL_FRACTION / (1 - NOCALL_FRACTION))
    random.shuffle(nocalls)
    nocalls_kept = nocalls[:max_nocall]
    final = calls + nocalls_kept
    random.shuffle(final)
    print(f"mix: calls={len(calls)} no-calls={len(nocalls_kept)} "
          f"(dropped {len(nocalls)-len(nocalls_kept)} extra no-call)")

    for r in final:
        r["n_calls"] = sum(len(m.get("tool_calls") or []) for m in r["messages"])

    n_val = max(500, int(len(final) * 0.02))
    val, train = final[:n_val], final[n_val:]
    for split, data in (("train_v2", train), ("val_v2", val)):
        with open(BASE / "data" / f"{split}.jsonl", "w") as f:
            for r in data:
                r.pop("_split", None)
                f.write(json.dumps(r) + "\n")
    print(f"\nsaved train_v2={len(train)} val_v2={len(val)}")
    print("sources:", Counter(r["source"] for r in train))
    print("with tool schemas:", sum(1 for r in train if r.get("tools")))
    print("with tool calls:", sum(1 for r in train if r["n_calls"] > 0))


if __name__ == "__main__":
    main()
