#!/usr/bin/env python3
"""Tool-call accuracy eval — compares models through the exact agent path (Ollama).

Builds test cases from the val set (first tool call per conversation) plus the
synthetic agent cases, calls /api/chat with tools, and scores:
  - format:    did the model emit a STRUCTURED tool call?
  - name:      correct function name?
  - args:      arguments exactly match?
  - no_call:   for cases where no tool should be called, did it abstain?

Usage:
  python3 scripts/eval_toolcalls.py --model qwen3:8b --n 120
  python3 scripts/eval_toolcalls.py --model qwen3-8b-toolcall --n 120
"""
import argparse
import json
import random
import time
import urllib.request
from pathlib import Path

BASE = Path.home() / "model training"
OLLAMA = "http://127.0.0.1:11434"


def chat(model, messages, tools, timeout=180):
    body = {"model": model, "messages": messages, "stream": False, "think": False}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(
        f"{OLLAMA}/api/chat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


ABSTAIN_TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a text file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "run_command", "description": "Run a shell command",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                       "required": ["command"]}}},
]

# Prompts that are answerable directly and need no tool. Tools are still
# supplied on purpose: abstention is only meaningful when the model has
# something it *could* reach for and correctly declines to.
ABSTAIN_PROMPTS = [
    "What is an assertion error in testing?",
    "Explain what a git rebase does.",
    "Summarize the difference between TCP and UDP.",
    "What does the word refactoring mean in software?",
    "What is the difference between a list and a tuple in Python?",
    "Why does dividing by zero raise an exception?",
    "What time zone is UTC and how many hours ahead is IST?",
    "Explain the difference between RAM and disk storage.",
    "What does the acronym API stand for?",
    "Describe what a hash function does.",
    "What is the capital of France, and why is it famous?",
    "Explain what a for loop does in simple terms.",
]


_PY_TO_JSON = {
    "str": "string", "string": "string", "int": "integer",
    "integer": "integer", "float": "number", "number": "number",
    "bool": "boolean", "boolean": "boolean", "dict": "object",
    "object": "object", "list": "array", "none": "null", "null": "null",
}


def _json_type(t):
    """Map a when2call Python type annotation to a JSON Schema type.

    when2call ships annotations like 'str', 'int', 'List[int]',
    'str, optional' rather than JSON Schema. Every such schema makes Ollama
    reject the request outright (HTTP 400), which silently dropped all 258
    hard abstention cases. Normalizing recovers them.
    """
    if not isinstance(t, str):
        return None
    raw = t.strip().lower()
    optional = "optional" in raw
    raw = raw.replace("optional", "").replace("[", " ").replace("]", " ").strip()
    parts = [p for p in raw.replace("|", " ").split() if p]
    if not parts:
        return "string"
    base = parts[0]
    if base in ("list", "tuple", "set"):
        return "array"
    if base in ("dict",):
        return "object"
    if base in _PY_TO_JSON:
        return _PY_TO_JSON[base]
    # 'Any', 'object', unknown -> string is the safest widening.
    return "string"


def _normalize_schema(node, parent_optional=False):
    """Recursively rewrite Python annotations into valid JSON Schema."""
    if isinstance(node, list):
        out = []
        for item in node:
            conv = _normalize_schema(item, parent_optional)
            if conv is not None:
                out.append(conv)
        return out

    if not isinstance(node, dict):
        return None

    out = {}
    raw_t = node.get("type")
    is_opt = parent_optional or (
        isinstance(raw_t, str) and "optional" in raw_t.lower())

    if raw_t is not None:
        out["type"] = _json_type(raw_t)
    else:
        out["type"] = "object"

    if out["type"] == "array":
        items = node.get("items")
        if isinstance(items, list) and items:
            conv = _normalize_schema(items[0], False)
            if conv is not None:
                out["items"] = conv
        elif isinstance(items, dict):
            conv = _normalize_schema(items, False)
            if conv is not None:
                out["items"] = conv
        else:
            out["items"] = {"type": "string"}

    if out["type"] == "object":
        props = node.get("properties")
        if isinstance(props, dict):
            new_props, required = {}, []
            for pname, pschema in props.items():
                conv = _normalize_schema(pschema, False)
                if conv is None:
                    continue
                new_props[pname] = conv
                if not conv.get("_optional"):
                    required.append(pname)
            out["properties"] = new_props
            if required and not is_opt:
                out["required"] = required

    for key in ("description", "default", "enum"):
        if key in node and not isinstance(node[key], (dict, list)):
            out[key] = node[key]

    # strip internal marker
    if out.get("_optional"):
        out.pop("_optional", None)
    return out


def _clean_optional(node):
    """Mark nodes whose Python annotation said 'optional'."""
    if not isinstance(node, dict):
        return
    raw_t = node.get("type")
    if isinstance(raw_t, str) and "optional" in raw_t.lower():
        node["_optional"] = True
    for v in node.get("properties", {}).values() if isinstance(
            node.get("properties"), dict) else []:
        _clean_optional(v)
    items = node.get("items")
    if isinstance(items, dict):
        _clean_optional(items)


def _usable(tools):
    """Return True if every tool has a name and a JSON-Schema-typed parameters."""
    known = {"object", "array", "string", "integer", "number", "boolean", "null"}

    def walk(n):
        if isinstance(n, dict):
            t = n.get("type")
            if t is not None:
                if isinstance(t, str) and t not in known:
                    return False
                if isinstance(t, list) and any(x not in known for x in t):
                    return False
            for v in n.values():
                if not walk(v):
                    return False
        elif isinstance(n, list):
            return all(walk(v) for v in n)
        return True

    for t in tools or []:
        fn = t.get("function") or {}
        if not fn.get("name"):
            return False
        p = fn.get("parameters")
        if p is not None and not walk(p):
            return False
    return True


def _schema_ok(tools):
    """Back-compat alias: usable after normalization."""
    return _usable(tools)


def _normalize_tools(tools):
    """Return JSON-Schema-valid tools, or None if nothing survives."""
    if not tools:
        return None
    out = []
    for t in tools:
        fn = dict(t.get("function") or {})
        if not fn.get("name"):
            continue
        params = fn.get("parameters")
        if isinstance(params, dict):
            _clean_optional(params)
            conv = _normalize_schema(params, False)
            if conv is None:
                continue
            fn["parameters"] = conv
        out.append({"type": "function", "function": fn})
    return out if out else None


def build_cases(n, n_abstain=None, n_hard_abstain=None):
    tool_cases = []
    with open(BASE / "data" / "val.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            msgs = rec["messages"]
            # find first assistant tool call
            for i, m in enumerate(msgs):
                if m.get("role") == "assistant" and m.get("tool_calls"):
                    prompt = [x for x in msgs[:i]]
                    exp = m["tool_calls"][0]["function"]
                    if not prompt or prompt[-1]["role"] not in ("user", "tool"):
                        continue
                    tool_cases.append({
                        "messages": prompt,
                        "tools": rec.get("tools"),
                        "expected": {"name": exp["name"], "arguments": exp.get("arguments", {})},
                        "source": rec.get("source", "?"),
                    })
                    break

    # when2call examples are the real abstention set: the user asks for
    # something tool-shaped but underspecified, and the correct behaviour is to
    # ask a clarifying question instead of calling. 258/300 carry a tool list;
    # the rest are unusable. These are genuinely hard — a naive model calls the
    # tool — and they were previously discarded because build_cases only looked
    # for assistant messages containing tool_calls.
    w2c_cases = []
    w2c_path = BASE / "data" / "val.jsonl"
    if w2c_path.exists():
        with open(w2c_path) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("source") != "when2call":
                    continue
                tools = rec.get("tools")
                msgs = rec.get("messages") or []
                if not tools or len(msgs) < 3:
                    continue
                if msgs[0].get("role") != "system" or msgs[1].get("role") != "user":
                    continue
                # when2call ships Python type annotations, not JSON Schema, so
                # Ollama rejects every request with HTTP 400. Normalize first;
                # only keep what is still valid afterwards.
                norm = _normalize_tools(tools)
                if not norm or not _usable(norm):
                    continue
                w2c_cases.append({
                    "messages": msgs[:2],
                    "tools": norm,
                    "expected": None,
                    "source": "abstain_hard",
                })

    random.seed(7)
    random.shuffle(tool_cases)
    random.seed(7)
    random.shuffle(w2c_cases)

    # Abstention cases are appended AFTER the slice. Shuffling them into the
    # same pool meant they were almost always sliced away, so every run
    # reported abstain_total=0 and the metric was never actually measured.
    if n_abstain is None:
        n_abstain = len(ABSTAIN_PROMPTS)
    random.seed(7)
    random.shuffle(ABSTAIN_PROMPTS)
    easy_abstain = [
        {"messages": [{"role": "user", "content": q}],
         "tools": ABSTAIN_TOOLS, "expected": None, "source": "abstain_easy"}
        for q in ABSTAIN_PROMPTS[:n_abstain]
    ]
    # Hard abstention (real when2call prompts) is a separate axis: scored under
    # its own source label so easy-vs-hard can be read independently.
    # None means "all usable"; 0 means "skip"; a positive int caps it.
    hard = [] if n_hard_abstain == 0 else w2c_cases[:n_hard_abstain]
    return tool_cases[:n] + easy_abstain + hard


def norm_args(a):
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except json.JSONDecodeError:
            return a
    if isinstance(a, dict):
        return {k: ("" if v is None else v) for k, v in sorted(a.items())}
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--n-abstain", type=int, default=None,
                    help="easy abstention cases to append (default: all built-in)")
    ap.add_argument("--n-hard-abstain", type=int, default=None,
                    help="when2call abstention cases (default: all usable)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cases = build_cases(args.n, args.n_abstain, args.n_hard_abstain)
    n_tool = sum(1 for c in cases if c["expected"] is not None)
    n_abs = len(cases) - n_tool
    print(f"eval cases: {len(cases)} = {n_tool} tool + {n_abs} abstention "
          f"(model={args.model})")

    stats = {"format": 0, "name": 0, "args": 0, "no_call": 0,
             "total": 0, "abstain_total": 0, "abstain_ok": 0,
             "abstain_hard_total": 0, "abstain_hard_ok": 0, "errors": 0}
    per_source = {}
    errs = {}
    t0 = time.time()
    for i, case in enumerate(cases):
        resp = chat(args.model, case["messages"], case.get("tools"))
        if "error" in resp:
            stats["errors"] += 1
            # Surface the cause: a whole batch silently scoring as errors is
            # indistinguishable from a bad model unless the reason is shown.
            errs[src] = errs.get(src, 0) + 1
            if stats["errors"] <= 5:
                print(f"  error on {src}: {str(resp['error'])[:120]}")
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(cases)} ...")
            continue
        calls = resp.get("message", {}).get("tool_calls") or []
        src = case["source"]

        if case["expected"] is None:  # abstention case
            # Scored on its own denominators. Counting these into stats["total"]
            # would silently drag the format/name/args rates down, since an
            # abstention correctly emits no structured call at all.
            stats["abstain_total"] += 1
            hard = src == "abstain_hard"
            if hard:
                stats["abstain_hard_total"] += 1
            if not calls:
                stats["abstain_ok"] += 1
                if hard:
                    stats["abstain_hard_ok"] += 1
                per_source.setdefault(src, [0, 0])[0] += 1
            per_source.setdefault(src, [0, 0])[1] += 1
            continue

        per_source.setdefault(src, [0, 0])
        per_source[src][1] += 1
        stats["total"] += 1
        if calls:
            stats["format"] += 1
            fn = calls[0].get("function", {})
            if fn.get("name") == case["expected"]["name"]:
                stats["name"] += 1
                per_source[src][0] += 1
                if norm_args(fn.get("arguments")) == norm_args(case["expected"]["arguments"]):
                    stats["args"] += 1
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(cases)} ...")

    n = max(stats["total"], 1)
    print(f"\n=== RESULTS — {args.model} ===")
    print(f"cases scored     : {stats['total']}  (errors: {stats['errors']})")
    print(f"structured calls : {stats['format']}/{n} = {stats['format']/n:.1%}")
    print(f"tool-name acc    : {stats['name']}/{n} = {stats['name']/n:.1%}")
    print(f"args-exact acc   : {stats['args']}/{n} = {stats['args']/n:.1%}")
    ab = max(stats["abstain_total"], 1)
    print(f"abstention acc   : {stats['abstain_ok']}/{stats['abstain_total']} = "
          f"{stats['abstain_ok']/ab:.1%}")
    if stats["abstain_hard_total"]:
        hb = max(stats["abstain_hard_total"], 1)
        print(f"  hard (when2call): {stats['abstain_hard_ok']}/{stats['abstain_hard_total']} = "
              f"{stats['abstain_hard_ok']/hb:.1%}")
    else:
        print("  hard (when2call): NOT MEASURED (0 cases)")
    if errs:
        print("errors by source: " + ", ".join(f"{k}={v}" for k, v in errs.items()))
    print(f"elapsed: {(time.time()-t0)/60:.1f} min")
    print("per-source (name acc on tool cases; 'abstain' = correctly declined):")
    for src, (ok, tot) in sorted(per_source.items()):
        print(f"  {src:12s}: {ok}/{tot} = {ok/max(tot,1):.1%}")
    if stats["abstain_total"] == 0:
        print("WARNING: zero abstention cases scored - metric is not measured")

    if args.out:
        Path(args.out).write_text(json.dumps({"model": args.model, **stats,
                                              "per_source": per_source}, indent=2))
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
