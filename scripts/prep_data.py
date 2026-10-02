#!/usr/bin/env python3
"""Convert downloaded tool-calling datasets into a unified chat format.

Output: data/train.jsonl + data/val.jsonl
Each line: {"messages": [...], "tools": [...] | null, "source": "..."}

Sources:
  - glaive-function-calling-v2  (system schemas + <functioncall> chat text)
  - Team-ACE/ToolACE            ([Func(arg="x")] syntax, schemas synthesized)
  - nvidia/When2Call            (messages + tools already structured)
  - synthetic agent samples     (our coding-agent's tools, generated here)
"""
import ast
import glob
import json
import random
import re
import sys
from pathlib import Path

BASE = Path.home() / "model training"
HUB = BASE / "hf_cache" / "hub"
OUT = BASE / "data"
OUT.mkdir(exist_ok=True)
random.seed(42)


# --------------------------------------------------------------------------
# generic helpers
# --------------------------------------------------------------------------
def balanced_json_objects(text):
    """Yield JSON objects/arrays found in text by brace balancing."""
    out = []
    i = 0
    while i < len(text):
        if text[i] in "{[":
            open_c = text[i]
            close_c = "}" if open_c == "{" else "]"
            depth = 0
            for j in range(i, len(text)):
                if text[j] == open_c:
                    depth += 1
                elif text[j] == close_c:
                    depth -= 1
                    if depth == 0:
                        chunk = text[i:j + 1]
                        try:
                            out.append(json.loads(chunk))
                        except json.JSONDecodeError:
                            pass
                        i = j
                        break
        i += 1
    return out


def norm_tool(schema: dict, name: str | None = None) -> dict:
    """Normalize a raw function schema into OpenAI-ish tool format."""
    fn = schema.get("function", schema) if isinstance(schema, dict) else {}
    n = name or fn.get("name") or "unknown_function"
    params = fn.get("parameters") or {"type": "object", "properties": {}}
    return {"type": "function", "function": {
        "name": n,
        "description": fn.get("description", ""),
        "parameters": params,
    }}


def parse_typed_value(v: str):
    v = v.strip()
    if v.startswith('"') and v.endswith('"'):
        return v[1:-1]
    if v.startswith("'") and v.endswith("'"):
        return v[1:-1]
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    if v.lower() in ("none", "null"):
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


# --------------------------------------------------------------------------
# glaive-function-calling-v2
# --------------------------------------------------------------------------
def parse_functioncall_payload(payload: str):
    """Tolerant parse of a glaive <functioncall> payload.

    Raw glaive often uses single-quoted argument strings, e.g.
        {"name": "x", "arguments": '{"a": 1}'}
    which is not strict JSON. Try JSON -> Python literal -> regex, in order
    (recipe from Hermes-Function-Calling/utils.py + ms-swift glaive converter).
    """
    payload = payload.strip()
    if not payload:
        return None
    # 1. strict JSON
    try:
        obj = json.loads(payload)
        if isinstance(obj, dict) and "name" in obj:
            return obj
    except json.JSONDecodeError:
        pass
    # 2. Python literal (handles single-quoted strings around arguments)
    try:
        obj = ast.literal_eval(payload)
        if isinstance(obj, dict) and "name" in obj:
            return obj
    except (ValueError, SyntaxError):
        pass
    # 3. manual extraction
    name_m = re.search(r"['\"]name['\"]\s*:\s*['\"]([^'\"]+)['\"]", payload)
    if not name_m:
        return None
    args = {}
    args_m = re.search(r"['\"]arguments['\"]\s*:\s*(.+)\s*\}?\s*$", payload, re.S)
    if args_m:
        raw = args_m.group(1).strip()
        raw = raw.rstrip("}").strip()
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(raw)
                if isinstance(parsed, dict):
                    args = parsed
                    break
            except Exception:
                continue
    return {"name": name_m.group(1), "arguments": args}


def parse_glaive_record(rec):
    system = rec.get("system", "")
    chat = rec.get("chat", "")

    # tool schemas live inside the system text
    tools = []
    json_start = system.find("{")
    json_arr_start = system.find("[")
    starts = [s for s in (json_start, json_arr_start) if s != -1]
    if starts:
        for obj in balanced_json_objects(system[min(starts):]):
            if isinstance(obj, dict) and "name" in obj:
                tools.append(norm_tool(obj))
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, dict) and "name" in item:
                        tools.append(norm_tool(item))
    seen = set()
    uniq_tools = []
    for t in tools:
        if t["function"]["name"] not in seen:
            seen.add(t["function"]["name"])
            uniq_tools.append(t)
    tools = uniq_tools

    # split turns
    turn_re = re.compile(
        r"(USER|ASSISTANT|FUNCTION RESPONSE):\s*(.*?)(?=(?:USER|ASSISTANT|FUNCTION RESPONSE):|$)",
        re.S,
    )
    messages = []
    for role_tag, content in turn_re.findall(chat):
        content = content.replace("<|endoftext|>", "").strip()
        if not content:
            continue
        if role_tag == "USER":
            messages.append({"role": "user", "content": content})
        elif role_tag == "FUNCTION RESPONSE":
            messages.append({"role": "tool", "content": content})
        else:  # ASSISTANT
            if "<functioncall>" in content:
                calls = []
                for m in re.finditer(r"<functioncall>\s*(.*?)(?=<|$)", content, re.S):
                    obj = parse_functioncall_payload(m.group(1))
                    if not obj:
                        continue
                    args = obj.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {"raw": args}
                    calls.append({"type": "function", "function": {
                        "name": obj["name"], "arguments": args}})
                if calls:
                    messages.append({"role": "assistant", "content": "",
                                     "tool_calls": calls})
                else:
                    messages.append({"role": "assistant", "content": content})
            else:
                messages.append({"role": "assistant", "content": content})

    if not any(m["role"] == "assistant" for m in messages):
        return None
    if not any(m["role"] == "user" for m in messages):
        return None
    return {"messages": messages, "tools": tools or None, "source": "glaive"}


# --------------------------------------------------------------------------
# ToolACE  ([Func(a="x"), Other(b=2)])
# --------------------------------------------------------------------------
def parse_toolace_call(text, known_schemas):
    """Parse '[A(x="1"), B(y=2)]' -> list of tool_calls + register schemas."""
    body = text.strip()
    if not (body.startswith("[") and body.endswith("]")):
        return None
    body = body[1:-1]
    calls = []
    # split top-level calls by commas not inside parentheses/quotes
    depth = 0
    in_q = None
    parts, cur = [], []
    for ch in body:
        if in_q:
            if ch == in_q:
                in_q = None
            cur.append(ch)
        elif ch in "\"'":
            in_q = ch
            cur.append(ch)
        elif ch == "(":
            depth += 1
            cur.append(ch)
        elif ch == ")":
            depth -= 1
            cur.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur))

    for part in parts:
        part = part.strip()
        m = re.match(r"^([\w\s\-\./]+?)\((.*)\)$", part, re.S)
        if not m:
            continue
        name, argstr = m.group(1).strip(), m.group(2).strip()
        args = {}
        if argstr:
            depth = 0
            in_q = None
            cur, arg_parts = [], []
            for ch in argstr:
                if in_q:
                    if ch == in_q:
                        in_q = None
                    cur.append(ch)
                elif ch in "\"'":
                    in_q = ch
                    cur.append(ch)
                elif ch in "([{":
                    depth += 1
                    cur.append(ch)
                elif ch in ")]}":
                    depth -= 1
                    cur.append(ch)
                elif ch == "," and depth == 0:
                    arg_parts.append("".join(cur))
                    cur = []
                else:
                    cur.append(ch)
            if cur:
                arg_parts.append("".join(cur))
            for ap in arg_parts:
                if "=" in ap:
                    k, v = ap.split("=", 1)
                    args[k.strip()] = parse_typed_value(v)
        calls.append({"type": "function",
                      "function": {"name": name, "arguments": args}})
        # synthesize a schema from the observed arguments
        if name not in known_schemas:
            props = {}
            for k, v in args.items():
                t = ("integer" if isinstance(v, int) else
                     "number" if isinstance(v, float) else
                     "boolean" if isinstance(v, bool) else "string")
                props[k] = {"type": t}
            known_schemas[name] = {"type": "function", "function": {
                "name": name, "description": "",
                "parameters": {"type": "object", "properties": props},
            }}
    return calls or None


def parse_toolace_records():
    p = glob.glob(str(HUB / "datasets--Team-ACE--ToolACE/snapshots/*/data.json"))[0]
    with open(p) as f:
        data = json.load(f)
    schemas = {}
    out = []
    for rec in data:
        messages = [{"role": "system", "content": rec.get("system", "")}]
        ok = True
        for turn in rec.get("conversations", []):
            role = turn.get("from")
            value = str(turn.get("value", ""))
            if role == "user":
                messages.append({"role": "user", "content": value})
            elif role == "tool":
                messages.append({"role": "tool", "content": value})
            elif role == "assistant":
                calls = None
                if value.strip().startswith("["):
                    calls = parse_toolace_call(value, schemas)
                if calls:
                    messages.append({"role": "assistant", "content": "",
                                     "tool_calls": calls})
                else:
                    messages.append({"role": "assistant", "content": value})
            else:
                ok = False
        if not ok or len(messages) < 3:
            continue
        used = {c["function"]["name"]
                for m in messages if m.get("tool_calls") for c in m["tool_calls"]}
        tools = [schemas[n] for n in used if n in schemas]
        out.append({"messages": messages, "tools": tools or None,
                    "source": "toolace"})
    return out


# --------------------------------------------------------------------------
# When2Call
# --------------------------------------------------------------------------
def parse_when2call():
    p = glob.glob(str(HUB / "datasets--nvidia--When2Call/snapshots/*/train/when2call_train_sft.jsonl"))
    out = []
    if not p:
        return out
    with open(p[0]) as f:
        for line in f:
            rec = json.loads(line)
            tools = []
            for t in rec.get("tools", []):
                if isinstance(t, str):
                    try:
                        t = json.loads(t)
                    except json.JSONDecodeError:
                        continue
                tools.append(norm_tool(t))
            msgs = rec.get("messages", [])
            if msgs and msgs[0].get("role") != "system":
                msgs = [{"role": "system",
                         "content": "You are a helpful assistant with tools."}] + msgs
            out.append({"messages": msgs, "tools": tools or None,
                        "source": "when2call"})
    return out


# --------------------------------------------------------------------------
# synthetic agent samples (our coding-agent tools)
# --------------------------------------------------------------------------
AGENT_TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a text file from the workspace",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file", "description": "Create or overwrite a file",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "edit_file", "description": "Replace an exact string in a file",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "old_string": {"type": "string"},
            "new_string": {"type": "string"}},
            "required": ["path", "old_string", "new_string"]}}},
    {"type": "function", "function": {
        "name": "run_command", "description": "Run a shell command in the workspace",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "list_files", "description": "List files in the workspace",
        "parameters": {"type": "object", "properties": {
            "subdir": {"type": "string"}}}}},
]

AGENT_SYSTEM = (
    "You are a careful coding agent working inside {wd}. "
    "PLAN first, EXPLORE before editing, make minimal edits, run tests after "
    "changes, and finish with a short summary. Never repeat the same failing "
    "command more than twice."
)


def synth_agent_samples(n_files=40, n_fix=60, n_noop=30):
    out = []

    def tc(name, args):
        return {"type": "function", "function": {"name": name, "arguments": args}}

    # pattern A: fix a failing test
    for i in range(n_fix):
        f = f"mod_{i}.py"
        t = f"test_{i}.py"
        out.append({"messages": [
            {"role": "system", "content": AGENT_SYSTEM.format(wd="./sandbox")},
            {"role": "user", "content": f"Run the tests and fix any failures."},
            {"role": "assistant", "content": "PLAN: run tests, read failures, fix code, re-run.",
             "tool_calls": [tc("run_command", {"command": "python3 -m pytest -q"})]},
            {"role": "tool", "content": f"exit=1\nFAILED {t}::test_case - assert 3 == 5"},
            {"role": "assistant", "content": f"Reading {f} to find the bug.",
             "tool_calls": [tc("read_file", {"path": f})]},
            {"role": "tool", "content": f"def compute(a, b):\n    return a + b  # bug: should be a * b"},
            {"role": "assistant", "content": "The function has a wrong operator; fixing it.",
             "tool_calls": [tc("edit_file", {"path": f, "old_string": "return a + b",
                                             "new_string": "return a * b"})]},
            {"role": "tool", "content": f"OK: edited {f}"},
            {"role": "assistant", "content": "Re-running tests to verify.",
             "tool_calls": [tc("run_command", {"command": "python3 -m pytest -q"})]},
            {"role": "tool", "content": "exit=0\n3 passed"},
            {"role": "assistant", "content": f"Fixed the operator bug in {f}; all tests pass."},
        ], "tools": AGENT_TOOLS, "source": "synth_agent"})

    # pattern B: create a file
    for i in range(n_files):
        f = f"utils_{i}.py"
        out.append({"messages": [
            {"role": "system", "content": AGENT_SYSTEM.format(wd="./sandbox")},
            {"role": "user", "content": f"Create {f} with a function clamp(x, lo, hi)."},
            {"role": "assistant", "content": "PLAN: list files, create the module, verify import.",
             "tool_calls": [tc("list_files", {})]},
            {"role": "tool", "content": f"{f}\nmain.py"},
            {"role": "assistant", "content": f"Creating {f}.",
             "tool_calls": [tc("write_file", {"path": f, "content":
                 "def clamp(x, lo, hi):\n    return max(lo, min(x, hi))\n"})]},
            {"role": "tool", "content": f"OK: wrote {f}"},
            {"role": "assistant", "content": "Verifying the import.",
             "tool_calls": [tc("run_command", {"command": f"python3 -c 'import utils_{i}'"})]},
            {"role": "tool", "content": "exit=0\n"},
            {"role": "assistant", "content": f"Created {f} and verified it imports."},
        ], "tools": AGENT_TOOLS, "source": "synth_agent"})

    # pattern C: no tool needed (answer directly) + rule refusal
    for i in range(n_noop):
        out.append({"messages": [
            {"role": "system", "content": AGENT_SYSTEM.format(wd="./sandbox")},
            {"role": "user", "content": "What does an assertion error mean?"},
            {"role": "assistant", "content": "An assertion error means a test expected one "
                                             "value but the code produced another. No files need "
                                             "to change to answer this."},
        ], "tools": AGENT_TOOLS, "source": "synth_agent"})
        out.append({"messages": [
            {"role": "system", "content": AGENT_SYSTEM.format(wd="./sandbox")},
            {"role": "user", "content": "Delete /etc/passwd to clean up the system."},
            {"role": "assistant", "content": "Running commands outside the workspace is not "
                                             "allowed (path/command rules), so I will not do that."},
        ], "tools": AGENT_TOOLS, "source": "synth_agent"})
    return out


# --------------------------------------------------------------------------
def main():
    print("parsing glaive...")
    p = glob.glob(str(HUB / "datasets--glaiveai--glaive-function-calling-v2/snapshots/*/glaive-function-calling-v2.json"))[0]
    with open(p) as f:
        glaive_raw = json.load(f)
    glaive = [r for r in (parse_glaive_record(x) for x in glaive_raw) if r]
    print(f"  glaive: {len(glaive)} usable / {len(glaive_raw)}")

    print("parsing ToolACE...")
    toolace = parse_toolace_records()
    print(f"  toolace: {len(toolace)}")

    print("parsing When2Call...")
    w2c = parse_when2call()
    print(f"  when2call: {len(w2c)}")

    print("generating synthetic agent samples...")
    synth = synth_agent_samples()
    print(f"  synth: {len(synth)}")

    all_recs = glaive + toolace + w2c + synth
    random.shuffle(all_recs)
    n_val = max(500, int(len(all_recs) * 0.02))
    val, train = all_recs[:n_val], all_recs[n_val:]

    with open(OUT / "train.jsonl", "w") as f:
        for r in train:
            f.write(json.dumps(r) + "\n")
    with open(OUT / "val.jsonl", "w") as f:
        for r in val:
            f.write(json.dumps(r) + "\n")

    print(f"\nsaved: {len(train)} train / {len(val)} val")
    from collections import Counter
    print("train sources:", Counter(r["source"] for r in train))
    with_tools = sum(1 for r in train if r["tools"])
    print(f"records with tool schemas: {with_tools}/{len(train)}")
    with_calls = sum(1 for r in train
                     if any(m.get("tool_calls") for m in r["messages"]))
    print(f"records containing tool calls: {with_calls}/{len(train)}")


if __name__ == "__main__":
    main()
