#!/usr/bin/env python3
"""V3 data pipeline — augments the training mix with three newly-available sources.

Motivation, from the v1 eval:
  * hard abstention (when2call) scored 90.7% after v1 training, vs 53.5% for the
    8B base. It is the tuned model's single biggest differentiator, yet only
    3.8% of the training mix was when2call. More clarify-style data should push
    it higher.
  * glaive was 77.5% of the mix and is templated/repetitive.
  * hermes-function-calling was fully downloaded the whole time and NEVER used.

New sources:
  when2call_train_sft   15000 rows, ALL of them "ask a clarifying question"
                        (no tool call) -> directly targets abstention
  when2call_train_pref   9000 rows with chosen/rejected clarify responses;
                        we keep the CHOSEN one as the target
  hermes func-calling    1893 rows, inline <tool_call> XML -> parsed to
                        structured tool_calls (multi-call conversations)
  glaive-function-calling-5k  5209 rows (already-local, same schema as glaive)

Output: data/train_v3.jsonl, data/val_v3.jsonl
Schema matches v2 exactly: {"messages", "tools", "source", "n_calls"}

Note on when2call tool schemas: they use Python annotations ('str', 'int',
'List[int]'), not JSON Schema. They are NORMALIZED here so the model is
trained on the same schema dialect it will be served with. (The eval hit this
exact bug: all 258 hard cases 400'd at Ollama.)
"""
import glob
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path.home() / "model training"
random.seed(42)

CACHE = BASE / "hf_cache" / "hub"
W2C_DIR = glob.glob(str(CACHE / "datasets--nvidia--When2Call/snapshots/*"))[0]
HERMES_DIR = glob.glob(str(CACHE / "datasets--NousResearch--hermes-function-calling-v1/snapshots/*"))[0]

# Target mix. Weighted by evidence, not by availability.
GLAIVE_TARGET = 0.34      # down from 77.5%
W2C_TARGET = 0.34         # abstention/clarify — the differentiator
HERMES_TARGET = 0.22      # structured multi-call, was 0%
TOOLACE_TARGET = 0.10     # keep, it is the hardest verified source

MAX_PER_SOURCE = 60000



# ---------------------------------------------------------------- utilities
def norm_args(a):
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except json.JSONDecodeError:
            return None
    return a if isinstance(a, dict) else None


_PY_TO_JSON = {
    "str": "string", "string": "string", "int": "integer", "integer": "integer",
    "float": "number", "number": "number", "bool": "boolean",
    "boolean": "boolean", "dict": "object", "object": "object",
    "list": "array", "none": "null", "null": "null",
}


def _json_type(t):
    if not isinstance(t, str):
        return None
    raw = t.strip().lower().replace("optional", "")
    raw = raw.replace("[", " ").replace("]", " ").strip()
    parts = [p for p in raw.replace("|", " ").split() if p]
    if not parts:
        return "string"
    base = parts[0]
    if base in ("list", "tuple", "set"):
        return "array"
    return _PY_TO_JSON.get(base, "string")


def _mark_optional(node):
    if not isinstance(node, dict):
        return
    if isinstance(node.get("type"), str) and "optional" in node["type"].lower():
        node["_opt"] = True
    props = node.get("properties")
    if isinstance(props, dict):
        for v in props.values():
            _mark_optional(v)
    if isinstance(node.get("items"), dict):
        _mark_optional(node["items"])


def _norm_node(node, parent_opt=False):
    if not isinstance(node, dict):
        return None
    is_opt = parent_opt or (isinstance(node.get("type"), str)
                            and "optional" in node["type"].lower())
    out = {"type": _json_type(node.get("type")) if node.get("type") is not None
           else "object"}
    if out["type"] == "array":
        items = node.get("items")
        first = None
        if isinstance(items, list) and items:
            first = items[0]
        elif isinstance(items, dict):
            first = items
        out["items"] = _norm_node(first, False) if first else {"type": "string"}
    elif out["type"] == "object":
        props = node.get("properties")
        new_props, required = {}, []
        if isinstance(props, dict):
            for name, sch in props.items():
                conv = _norm_node(sch, False)
                if conv is None:
                    continue
                new_props[name] = conv
                if not conv.get("_opt"):
                    required.append(name)
        out["properties"] = new_props
        if required and not is_opt:
            out["required"] = required
    for k in ("description", "default", "enum"):
        if k in node and not isinstance(node[k], (dict, list)):
            out[k] = node[k]
    if out.get("_opt"):
        out.pop("_opt")
    return out


def normalize_tools(tools):
    """when2call ships Python annotations; rewrite to JSON Schema.

    Its tools are a MIXED list: some entries are already
    {"type": "function", "function": {...}}, others are BARE function dicts,
    and some are JSON *strings*. All three are accepted here.
    """
    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
        except json.JSONDecodeError:
            return None
    out = []
    for t in tools or []:
        if isinstance(t, str):
            try:
                t = json.loads(t)
            except json.JSONDecodeError:
                continue
        if not isinstance(t, dict):
            continue
        if "function" in t and isinstance(t["function"], dict):
            fn = dict(t["function"])
        else:
            fn = {k: v for k, v in t.items() if k != "type"}
        if not fn.get("name"):
            continue
        params = fn.get("parameters")
        if isinstance(params, dict):
            _mark_optional(params)
            conv = _norm_node(params, False)
            if conv is not None:
                fn["parameters"] = conv
        out.append({"type": "function", "function": fn})
    return out or None


def tools_usable(tools):
    known = {"object", "array", "string", "integer", "number", "boolean", "null"}

    def walk(n):
        if isinstance(n, dict):
            t = n.get("type")
            if isinstance(t, str) and t not in known:
                return False
            if isinstance(t, list) and any(x not in known for x in t):
                return False
            return all(walk(v) for v in n.values())
        if isinstance(n, list):
            return all(walk(v) for v in n)
        return True

    return all((t.get("function") or {}).get("name") and
               walk(((t.get("function") or {}).get("parameters") or {}))
               for t in tools or [])


def verify(rec):
    """Same rules as prep_data_v2 plus a tool-count sanity check."""
    msgs = rec["messages"]
    tools = rec.get("tools") or []
    schema = {(t.get("function") or {}).get("name"): (t.get("function") or {}).get("parameters") or {}
              for t in tools}
    calls = 0
    for m in msgs:
        for c in m.get("tool_calls") or []:
            calls += 1
            fn = c.get("function") or {}
            name = fn.get("name")
            args = norm_args(fn.get("arguments"))
            if args is None:
                return False, "bad_args"
            if tools and name not in schema:
                return False, "name_not_declared"
            params = schema.get(name) or {}
            if isinstance(params, dict):
                reqd = params.get("required")
                # Some when2call schemas carry required as a bool or a bare
                # string instead of a list; only a real list is meaningful.
                if isinstance(reqd, str):
                    reqd = [reqd]
                elif not isinstance(reqd, list):
                    reqd = []
                for req in reqd:
                    if req not in args:
                        return False, "missing_required"
                props = params.get("properties") or {}
                if isinstance(props, dict):
                    for k, v in args.items():
                        pt = (props.get(k) or {}).get("type")
                        if pt == "integer" and not isinstance(v, int):
                            return False, "type_mismatch"
                        if pt == "number" and not isinstance(v, (int, float)):
                            return False, "type_mismatch"
                        if pt == "boolean" and not isinstance(v, bool):
                            return False, "type_mismatch"
    if any(m["role"] == "tool" for m in msgs) and calls == 0:
        return False, "tool_response_without_call"
    if len(msgs) < 2:
        return False, "too_short"
    return True, "ok"


def conv_hash(rec):
    parts = []
    for m in rec["messages"]:
        parts.append(m["role"] + ":" + (m.get("content") or "")[:200])
        for c in m.get("tool_calls") or []:
            fn = c.get("function") or {}
            parts.append(f"{fn.get('name')}:{json.dumps(norm_args(fn.get('arguments')) or {}, sort_keys=True)}")
    return hashlib.md5("|".join(parts).encode()).hexdigest()


# ---------------------------------------------------------------- loaders
def load_when2call_sft():
    """15k rows, every one a 'should not call, ask instead' case."""
    p = Path(W2C_DIR) / "train" / "when2call_train_sft.jsonl"
    if not p.exists():
        return []
    kept, drop = [], Counter()
    for line in open(p):
        r = json.loads(line)
        raw = r.get("tools")
        if isinstance(raw, list) and not raw:
            # 2,223/15,000 rows ship an EMPTY tool list. With no tools on
            # offer there is nothing to decline, so these cannot teach tool
            # calling. Reported explicitly rather than dropped silently.
            drop["empty_tool_list"] += 1
            continue
        tools = normalize_tools(raw)
        if not tools or not tools_usable(tools):
            drop["bad_schema"] += 1
            continue
        msgs = r.get("messages") or []
        # when2call rows are exactly (user, assistant) -- there is NO system
        # turn. The tool list carries the capability, so no system prompt is
        # synthesized; pretokenize.py renders tools into the system turn.
        if len(msgs) < 2 or msgs[0]["role"] != "user" or msgs[1]["role"] != "assistant":
            drop["bad_shape"] += 1
            continue
        kept.append({
            "messages": [
                {"role": "user", "content": msgs[0].get("content") or ""},
                {"role": "assistant", "content": msgs[1].get("content") or ""},
            ],
            "tools": tools,
            "source": "w2c_clarify",
        })
    print(f"  when2call sft : {len(kept):6d} kept | dropped {dict(drop)}")
    return kept


PREF_TC_RE = re.compile(r"<TOOLCALL>\s*(.*?)\s*</TOOLCALL>", re.S)


def load_when2call_pref():
    """9k preference rows.

    chosen_response is a DICT, not a string, and is NOT purely clarify:
      6,000 rows -> plain clarifying question  (abstention)
      3,000 rows -> <TOOLCALL>[...]</TOOLCALL>  (a real tool call)
    Both are usable, so both are kept and labelled by what they actually are
    rather than dumping everything into the clarify bucket.
    """
    p = Path(W2C_DIR) / "train" / "when2call_train_pref.jsonl"
    if not p.exists():
        return []
    kept, drop, kinds = [], Counter(), Counter()
    for line in open(p):
        r = json.loads(line)
        tools = normalize_tools(r.get("tools"))
        if not tools or not tools_usable(tools):
            drop["bad_schema"] += 1
            continue
        msgs = r.get("messages") or []
        chosen = r.get("chosen_response")
        if not (isinstance(chosen, dict) and msgs and msgs[0].get("role") == "user"):
            drop["bad_shape"] += 1
            continue
        text = chosen.get("content") or ""

        m2 = [{"role": "user", "content": msgs[0].get("content") or ""}]
        seg = PREF_TC_RE.search(text)
        if seg:
            # tool-call target: parse the JSON array inside <TOOLCALL>
            calls = []
            try:
                raw = json.loads(seg.group(1))
                raw = raw if isinstance(raw, list) else [raw]
            except json.JSONDecodeError:
                raw = []
            for c in raw:
                if isinstance(c, dict) and c.get("name"):
                    calls.append({"function": {"name": c["name"],
                                              "arguments": c.get("arguments", {})}})
            if not calls:
                drop["bad_toolcall"] += 1
                continue
            lead = PREF_TC_RE.sub("", text).strip()
            m2.append({"role": "assistant", "content": lead,
                       "tool_calls": calls})
            kinds["w2c_call"] += 1
            src = "w2c_call"
        else:
            body = text.strip()
            if not body:
                drop["empty_chosen"] += 1
                continue
            m2.append({"role": "assistant", "content": body})
            kinds["w2c_clarify"] += 1
            src = "w2c_clarify"
        kept.append({"messages": m2, "tools": tools, "source": src})
    print(f"  when2call pref: {len(kept):6d} kept | {dict(kinds)} | dropped {dict(drop)}")
    return kept


ZERO_W = ""
TC_RE = re.compile(
    r"<" + ZERO_W + r"?tool_call>\s*(.*?)\s*</" + ZERO_W + r"?tool_call>",
    re.S,
)


def load_hermes(fname="func-calling.json", source="hermes"):
    """Hermes stores tool calls as inline XML; parse into structured calls."""
    p = Path(HERMES_DIR) / fname
    if not p.exists():
        return []
    try:
        rows = json.load(open(p))
    except Exception:  # noqa: BLE001
        return []

    role_map = {"system": "system", "human": "user", "user": "user",
                "gpt": "assistant", "assistant": "assistant", "tool": "tool",
                "observation": "tool"}
    out, dropped = [], Counter()
    for r in rows:
        try:
            tools = normalize_tools(r.get("tools"))
            if not tools:
                dropped["no_tools"] += 1
                continue
        except Exception:  # noqa: BLE001
            dropped["bad_tools"] += 1
            continue

        msgs = []
        for m in r.get("conversations") or []:
            role = role_map.get(m.get("from") or m.get("role"))
            if not role:
                continue
            val = m.get("value") or m.get("content") or ""
            calls = []
            for seg in TC_RE.findall(val):
                try:
                    obj = json.loads(seg)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict) and obj.get("name"):
                    calls.append({"function": {
                        "name": obj["name"],
                        "arguments": obj.get("arguments", {}),
                    }})
            text = TC_RE.sub("", val).strip()
            # Hermes tags the tool_response with the function name.
            if role == "tool" and text.startswith("{"):
                try:
                    o = json.loads(text)
                    if "name" in o:
                        text = str(o.get("content", text))
                except json.JSONDecodeError:
                    pass
            if not text and not calls:
                continue
            m2 = {"role": role, "content": text}
            if calls:
                m2["tool_calls"] = calls
            msgs.append(m2)

        if len(msgs) >= 2:
            out.append({"messages": msgs, "tools": tools, "source": source})
        else:
            dropped["too_short"] += 1
    print(f"  hermes/{fname}: {len(out)} kept, {dict(dropped)}")
    return out


def load_jsonl_source(path, source):
    rows = []
    if not Path(path).exists():
        return rows
    for line in open(path):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def main():
    print("[1] loading new sources")
    new = []
    new += load_when2call_sft()
    new += load_when2call_pref()
    new += load_hermes("func-calling.json", "hermes")
    new += load_hermes("json-mode-agentic.json", "hermes_agentic")

    # glaive-function-calling-5k ships in Hermes' conversations+inline-XML
    # shape, NOT the plain glaive {"messages"} shape. It must go through the
    # Hermes parser or every tool call is silently lost.
    glaive5k = load_hermes("glaive-function-calling-5k.json", "glaive5k")

    print("[2] loading the v2 corpus (toolace + glaive + existing)")
    base_rows = []
    for line in open(BASE / "data" / "train_v2.jsonl"):
        base_rows.append(json.loads(line))
    for line in open(BASE / "data" / "val_v2.jsonl"):
        base_rows.append(json.loads(line))
    print(f"  v2 corpus: {len(base_rows)}")
    print("  v2 sources:", dict(Counter(r["source"] for r in base_rows)))

    everything = base_rows + new + glaive5k
    print(f"[3] combined raw: {len(everything)}")

    print("[4] verifying")
    ok, reasons = [], Counter()
    for r in everything:
        good, why = verify(r)
        if good:
            ok.append(r)
        else:
            reasons[why] += 1
    print(f"  verified {len(ok)} (dropped {sum(reasons.values())})")
    for w, n in reasons.most_common(8):
        print(f"    dropped {w}: {n}")

    print("[5] dedupe")
    seen, ded = set(), []
    for r in ok:
        h = conv_hash(r)
        if h in seen:
            continue
        seen.add(h)
        ded.append(r)
    print(f"  deduped -> {len(ded)}")

    print("[6] balancing to target mix")
    by_source = defaultdict(list)
    for r in ded:
        by_source[r["source"]].append(r)
    for v in by_source.values():
        random.shuffle(v)

    # Group sources into buckets so glaive-family stays bounded.
    glaive_like = ("glaive", "glaive5k")
    glaive_pool = [r for s in glaive_like for r in by_source.get(s, [])]
    toolace_pool = by_source.get("toolace", [])
    w2c_pool = by_source.get("w2c_clarify", [])
    w2c_call_pool = by_source.get("w2c_call", [])
    hermes_pool = (by_source.get("hermes", []) + by_source.get("hermes_agentic", [])
                   + by_source.get("when2call", []) + by_source.get("synth_agent", [])
                   + by_source.get("glaive5k", []))
    # 3,000 genuine tool-call rows from the pref split belong on the call side.
    w2c_call_pool = by_source.get("w2c_call", [])

    n_target = min(MAX_PER_SOURCE,
                   max(len(glaive_pool), len(toolace_pool) + len(w2c_pool)
                       + len(hermes_pool)))
    quotas = {
        "glaive_like": int(n_target * GLAIVE_TARGET),
        "toolace": int(n_target * TOOLACE_TARGET),
        "w2c_clarify": int(n_target * W2C_TARGET),
        "hermes": int(n_target * HERMES_TARGET),
    }
    pools = {
        "glaive_like": glaive_pool, "toolace": toolace_pool,
        "w2c_clarify": w2c_pool, "hermes": hermes_pool,
    }
    final = []
    for key, q in quotas.items():
        pool = pools[key]
        final += pool[:q]
        print(f"  {key:12s} pool={len(pool):6d} quota={q:6d} kept={min(len(pool), q):6d}")

    # Guarantee the clarify share is exactly what we intended, measured on
    # "no tool call in the assistant turn" rather than on the source label.
    print("[7] enforcing global clarify ratio")
    has_call = [r for r in final if any(m.get("tool_calls") for m in r["messages"])]
    no_call = [r for r in final if not any(m.get("tool_calls") for m in r["messages"])]
    want_nocall = int(len(has_call) * 0.34 / (1 - 0.34))
    random.shuffle(no_call)
    if len(no_call) > want_nocall:
        dropped = len(no_call) - want_nocall
        no_call = no_call[:want_nocall]
    else:
        dropped = 0
    final = has_call + no_call
    random.shuffle(final)
    print(f"  calls={len(has_call)} clarify={len(no_call)} "
          f"(target clarify {want_nocall}, dropped {dropped})")

    for r in final:
        r["n_calls"] = sum(len(m.get("tool_calls") or []) for m in r["messages"])

    n_val = max(600, int(len(final) * 0.02))
    val, train = final[:n_val], final[n_val:]
    for split, data in (("train_v3", train), ("val_v3", val)):
        with open(BASE / "data" / f"{split}.jsonl", "w") as f:
            for r in data:
                f.write(json.dumps(r) + "\n")

    print(f"\nsaved train_v3={len(train)}  val_v3={len(val)}")
    print("sources:", dict(Counter(r["source"] for r in train)))
    print("with tool schemas:", sum(1 for r in train if r.get("tools")))
    print("clarify (no tool call):",
          sum(1 for r in train if r["n_calls"] == 0),
          f"({sum(1 for r in train if r['n_calls']==0)/len(train):.1%})")
    print("multi-call samples:",
          sum(1 for r in train if r["n_calls"] > 1))


if __name__ == "__main__":
    main()