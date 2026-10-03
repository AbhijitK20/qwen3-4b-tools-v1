"""
Balanced score for tool-calling models, computed from BFCL v4 results.

Why this exists
---------------
BFCL's Irrelevance Detection category rewards *declining to call a tool*. A model can
maximise that metric by collapsing into always-refuse, which produces a large, impressive,
entirely meaningless abstention number. We measured this happening: a seed replicate scored
97.50% on irrelevance while dropping to 12.50% on Multiple (AST) -- declining 135 of 200
cases it should have answered.

So irrelevance must never be read on its own. This script pairs it with a category that
*requires* calling tools, and reports a combined number that punishes one-sidedness.

Metric
------
    harmonic_mean(irrelevance_accuracy, multiple_call_rate)

`irrelevance_accuracy` is BFCL's own Irrelevance Detection score. `multiple_call_rate` is the
fraction of Multiple (AST) entries on which the model emitted *any* tool call at all, read from
the raw generated responses.

Note the deliberate use of **call rate** rather than Multiple AST accuracy. AST accuracy
penalises wrong arguments and wrong schemas, so a healthy tool-calling model can score low on it
while still being perfectly willing to act. Refusal collapse shows up as a model *not calling*,
not as one calling badly. v1 calls 195/200 multiple entries and still scores only 37.00% AST --
calling often, not always right -- which is not the same failure as declining 135 times.

The harmonic mean (rather than the arithmetic mean) is deliberate: it collapses when either
component collapses, so a model cannot buy irrelevance points by sacrificing the ability to act.
A degenerate always-refuse model scores ~22 here; a balanced model scores ~52.

Degeneracy is flagged on call rate below DEGENERATE_CALL_RATE, not on a score gap, because a
large irrelevance-minus-AST gap is normal for a 4B model and does not imply refusal.

Usage
-----
    python scripts/balanced_score.py                 # score every model in BFCL's result dir
    python scripts/balanced_score.py --json          # machine-readable, for committing
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
from typing import Optional

# Where bfcl-eval writes generated responses / scores inside this venv.
_SITE = pathlib.Path(__file__).resolve().parent.parent / ".venv/lib/python3.12/site-packages"
RESULT_DIR = _SITE / "result"
SCORE_CSV = _SITE / "score" / "data_non_live.csv"

# A model that declines to emit a tool call on most tool-requiring entries is not a
# tool-calling model any more, whatever its abstention score says.
DEGENERATE_CALL_RATE = 50.0


def _pct(cell: str) -> Optional[float]:
    cell = (cell or "").strip()
    if not cell or cell in {"N/A", "None"}:
        return None
    try:
        return float(cell.rstrip("%"))
    except ValueError:
        return None


def emitted_call(row: dict) -> bool:
    """True if this BFCL response row contains an actual tool call.

    BFCL stores a model response as a list. A tool call is a non-empty dict at index 0, e.g.
    [{"get_weather": "{\"city\":\"Paris\"}"}]. A decline is a prose string, or an empty list.
    """
    resp = row.get("result")
    if not resp:
        return False
    try:
        first = resp[0]
    except (IndexError, TypeError):
        return False
    return isinstance(first, dict) and len(first) > 0


def call_rate(result_dir: pathlib.Path, category: str = "multiple") -> Optional[float]:
    """Percentage of entries in which the model emitted a tool call."""
    path = result_dir / "non_live" / f"BFCL_v4_{category}_result.json"
    if not path.exists():
        return None
    rows = [json.loads(l) for l in path.open() if l.strip()]
    if not rows:
        return None
    return round(100.0 * sum(emitted_call(r) for r in rows) / len(rows), 2)


def read_scores(csv_path: pathlib.Path = SCORE_CSV) -> dict[str, dict]:
    """Parse BFCL's own scorer output into {display_name: {category: pct}}."""
    out: dict[str, dict] = {}
    if not csv_path.exists():
        return out
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("Model") or "").strip()
            if not name:
                continue
            out[name] = {
                "irrelevance": _pct(row.get("Irrelevance Detection", "")),
                "multiple_ast": _pct(row.get("Multiple AST", "")),
                "non_live_overall": _pct(row.get("Non-Live Overall Acc", "")),
            }
    return out


def combined(irrelevance: Optional[float], call_pct: Optional[float]) -> Optional[float]:
    """Harmonic mean of the two components, or None if either is missing."""
    if irrelevance is None or call_pct is None:
        return None
    if irrelevance + call_pct == 0:
        return None
    return round(2 * irrelevance * call_pct / (irrelevance + call_pct), 2)


def is_degenerate(call_pct: Optional[float]) -> bool:
    if call_pct is None:
        return False
    return call_pct < DEGENERATE_CALL_RATE


# BFCL's aggregate CSV stores *display* names, which need not resemble the result-directory
# keys we registered (it also rewrites "-" as "/" in some paths). With a handful of models an
# explicit alias table is clearer and more auditable than fuzzy matching, and it removes the
# failure mode where one model's numbers get attributed to another.
RESULT_DIR_ALIASES = {
    "qwen3-4b-tools-v1 (ours)": "qwen3-4b-tools-v1",
    "qwen3-4b-base (untrained)": "qwen3-4b-base",
    "qwen3-8b (reference)": "qwen3-8b-ref",
    "seed1337": "qwen3-4b-s1337",
}


def build_table(
    csv_path: pathlib.Path = SCORE_CSV, result_dir: pathlib.Path = RESULT_DIR
) -> list[dict]:
    """Join BFCL's scores with call rates read from the raw responses.

    Matching is alias-first, then normalised name comparison. A positional fallback was tried
    and removed: it silently attributed one model's call rate to another, which is precisely
    the kind of error this script exists to prevent. An unmeasured category is reported as
    incomplete rather than borrowed.
    """
    scores = read_scores(csv_path)
    call_rates: dict[str, float] = {}
    if result_dir.exists():
        for d in sorted(result_dir.iterdir()):
            if d.is_dir():
                cr = call_rate(d)
                if cr is not None:
                    call_rates[d.name] = cr

    def match(display: str) -> Optional[str]:
        alias = RESULT_DIR_ALIASES.get(display)
        if alias and alias in call_rates:
            return alias
        norm = display.lower().replace(" ", "").replace("(", "").replace(")", "")
        norm = norm.replace("-", "").replace("_", "").replace("/", "")
        for dname in call_rates:
            dn = dname.lower().replace("-", "").replace("_", "").replace("/", "")
            if dn and (dn in norm or norm.endswith(dn)):
                return dname
        return None

    rows = []
    for name, s in scores.items():
        key = match(name)
        cr = call_rates.get(key) if key else None
        comb = combined(s["irrelevance"], cr)
        rows.append(
            {
                "model": name,
                "result_dir": key,
                "irrelevance": s["irrelevance"],
                "multiple_ast": s["multiple_ast"],
                "multiple_call_rate": cr,
                "combined_harmonic": comb,
                "degenerate": is_degenerate(cr),
            }
        )
    rows.sort(key=lambda r: (r["combined_harmonic"] is None, -(r["combined_harmonic"] or 0)))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    ap.add_argument("--out", help="write JSON to this path")
    args = ap.parse_args()

    rows = build_table()

    if args.json or args.out:
        payload = {
            "benchmark": "BFCL v4 (non-live subset)",
            "metric": "harmonic_mean(irrelevance_accuracy, multiple_call_rate)",
            "rationale": (
                "Irrelevance rewards declining to call a tool and is maximised by an "
                "always-refuse model. Pairing it with Multiple (AST), which requires calling, "
                "makes an always-decline model score ~22 instead of looking like a 97.5% model."
            ),
            "degenerate_call_rate_threshold": DEGENERATE_CALL_RATE,
            "rows": rows,
            "caveat": (
                "Non-Live Overall Acc is intentionally not used: only some categories were run, "
                "so unscored categories contribute zero and make that column meaningless."
            ),
        }
        text = json.dumps(payload, indent=2) + "\n"
        if args.out:
            pathlib.Path(args.out).write_text(text)
            print(f"wrote {args.out}")
        else:
            print(text, end="")
        return

    print(f"{'model':28s} {'irrel':>7} {'call%':>7} {'AST':>7} {'combined':>9}  verdict")
    print("-" * 80)
    for r in rows:
        irr = f"{r['irrelevance']:.2f}" if r["irrelevance"] is not None else "-"
        call = f"{r['multiple_call_rate']:.1f}" if r["multiple_call_rate"] is not None else "-"
        ast = f"{r['multiple_ast']:.2f}" if r["multiple_ast"] is not None else "-"
        comb = f"{r['combined_harmonic']:.2f}" if r["combined_harmonic"] is not None else "-"
        verdict = "DEGENERATE (refuses to call)" if r["degenerate"] else ""
        if r["combined_harmonic"] is None:
            verdict = verdict or "incomplete"
        print(f"{r['model']:28s} {irr:>7} {call:>7} {ast:>7} {comb:>9}  {verdict}")


if __name__ == "__main__":
    main()
