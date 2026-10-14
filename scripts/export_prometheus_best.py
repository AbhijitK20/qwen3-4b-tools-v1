"""Apply a completed Prometheus trial's steering parameters and export a merged model.

Prometheus's non-interactive mode finishes the Optuna search but never exports the
winning adapter, because export normally happens in an interactive menu that needs a
TTY. This reproduces that step headlessly: it reads the study journal, rebuilds the
SteeringProfile for the chosen trial, applies the steering, and saves a merged model.
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "repos", "prometheus", "src"))

from prometheus.core.engine import SteeringEngine
from prometheus.core.steering import apply_steering
from prometheus.data import load_prompt_dataset
from prometheus.settings import PrometheusConfig
from prometheus.types import SteeringProfile
from prometheus.vectors import compute_steering_vectors


def load_trial_params(journal_glob):
    rows = [json.loads(l) for l in open(glob.glob(journal_glob)[0]) if l.strip()]
    done = {r["trial_id"]: r["values"] for r in rows if r.get("op_code") == 6}
    params = {}
    for r in rows:
        if r.get("op_code") == 9:
            for key, value in r.get("system_attr", {}).items():
                if "relative_params" in key:
                    params[r["trial_id"]] = json.loads(value)
    return done, params


def pick_trial(done, requested):
    if requested is not None:
        return requested
    zero = [t for t in done if done[t][0] == 0.0]
    pool = zero or list(done)
    return min(pool, key=lambda t: done[t][1])


def build_profiles(raw, components):
    profiles = {}
    for component in components:
        max_w = raw[f"{component}.max_weight"]
        profiles[component] = SteeringProfile(
            max_weight=max_w,
            max_weight_position=raw[f"{component}.max_weight_position"],
            min_weight=raw[f"{component}.min_weight"] * max_w,
            min_weight_distance=raw[f"{component}.min_weight_distance"],
        )
    return profiles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--journal", required=True)
    parser.add_argument("--trial", type=int)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    os.environ["PM_CONFIG"] = args.config
    sys.argv = [sys.argv[0]]
    config = PrometheusConfig()
    if not config.model.max_memory:
        config.model.max_memory = {"0": "3GiB", "cpu": "9GiB"}
    if config.inference.batch_size == 0:
        config.inference.batch_size = 1

    done, params = load_trial_params(args.journal)
    trial_id = pick_trial(done, args.trial)
    raw = params[trial_id]
    print(f"* selected trial {trial_id}: objective values {done[trial_id]}")

    engine = SteeringEngine(config)
    print("* loading prompts...")
    benign_msgs = load_prompt_dataset(config, config.benign_prompts)
    target_msgs = load_prompt_dataset(config, config.target_prompts)
    print(f"* {len(benign_msgs)} benign / {len(target_msgs)} target prompts")

    print("* extracting residuals for benign prompts...")
    benign_states = engine.extract_hidden_states_batched(benign_msgs)
    print("* extracting residuals for target prompts...")
    target_states = engine.extract_hidden_states_batched(target_msgs)

    vectors = compute_steering_vectors(
        benign_states,
        target_states,
        config.steering.vector_method,
        config.steering.orthogonal_projection,
    )
    del benign_states, target_states

    components = engine.list_steerable_components()
    profiles = build_profiles(raw, components)
    vector_index = raw["vector_index"]

    print(f"* applying steering (trial {trial_id}, vector_index={vector_index:.3f})...")
    engine.restore_baseline()
    apply_steering(engine, vectors, vector_index, profiles, config)

    print("* exporting merged model...")
    merged = engine.export_merged()
    merged.save_pretrained(args.out)
    engine.tokenizer.save_pretrained(args.out)
    print(f"* saved -> {args.out}")


if __name__ == "__main__":
    main()