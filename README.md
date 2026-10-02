# Qwen3-4B Tool-Calling Fine-Tune (v1)

A LoRA fine-tune of `Qwen3-4B` that teaches the model **when not to call a tool**.

The headline result: on a held-out abstention suite, this 4B model scores **90.7%** where an
untouched Qwen3-8B scores **53.5%** — a 4B model beating a model twice its size by 37 points
on the one behaviour that matters for not shipping a broken agent.

> **Read the [Limitations](#limitations) section before you trust any number here.**
> These are in-distribution results. No public benchmark has been run.

---

## The result

Held-out suite, 390 prompts per model: 120 tool-call cases, 12 easy negatives, 258 hard negatives.

| Model | Params | Format | Args exact | **Hard abstention** |
|---|---:|---:|---:|---:|
| `unsloth/Qwen3-4B` (untrained) | 4.0B | 99/120 | 87/120 | 166/258 = **64.3%** |
| **v1 (this repo)** | 4.0B | 114/120 | 100/120 | 234/258 = **90.7%** |
| Qwen3-8B (reference) | 8.2B | 119/120 | 103/120 | 138/258 = **53.5%** |

Broken out by source, because the aggregate hides what actually moved:

| Source | 4B base | **v1** | Qwen3-8B |
|---|---:|---:|---:|
| `toolace` (calls) | 39.1% | **100.0%** | 95.7% |
| `glaive` (calls) | 92.8% | 93.8% | **100.0%** |
| `abstain_easy` | 100.0% | 100.0% | 100.0% |
| **`abstain_hard`** | 64.3% | **90.7%** | 53.5% |

The win is concentrated almost entirely in `abstain_hard`: hard negatives are prompts that
*look* tool-shaped (a tool is defined, the request sounds actionable) but the correct answer is
to decline. The base model calls the tool. v1 declines. The 8B reference also calls it.

## Why this is the metric that matters

An agent that calls a tool when it shouldn't does something irreversible with the wrong
arguments. An agent that declines when it could have acted just asks again. **The cost of a
false positive is an order of magnitude worse than a false negative.**

`toolace` alone shows the base model was never broken — it could emit valid tool calls (39.1% →
100.0% format validity). What it lacked was calibration: the decision of *whether* to call.

---

## The experiment that failed (v4)

This is the most useful thing in the repo, so it's documented rather than deleted.

**Hypothesis:** adding `APIGen` training data would improve multi-step tool-call accuracy.

**Result:** partially true, net negative.

| Model | Train data | Format | Args | **Hard abstention** |
|---|---|---:|---:|---:|
| **v1** | glaive, toolace, when2call + synthetic negatives | 114/120 | 100/120 | **90.7%** |
| v4 | v1 + APIGen (dominant) | 115/120 | 102/120 | **73.6%** |

v4 bought +1 format point and +2 arg points and **lost 44 hard-negative cases** (−17 points).
Per-source shows it was surgical — `toolace` held at 100%, `glaive` *improved* slightly, and
the entire regression is in `abstain_hard`:

| Source | v1 | v4 | Δ |
|---|---:|---:|---:|
| `toolace` | 100.0% | 100.0% | — |
| `glaive` | 93.8% | 94.8% | +1.0 |
| **`abstain_hard`** | **90.7%** | **73.6%** | **−17.1** |

**Diagnosis:** APIGen is dominated by positive call examples. Proportionally increasing it
shifted the model's prior toward "a tool is available, therefore call it." The negative examples
were diluted rather than outvoted, so the model learned to call *more* — including on the hard
negatives it had previously learned to decline.

**Status:** rejected. v1 is the production model. v4 is retained as evidence for the diagnosis.

The obvious next experiment is a rebalanced mixture (APIGen capped ~10–15%, hard negatives
loss-weighted 2–3×). **That has not been run** — treat it as a proposal, not a result.

### Training loss

Last logged value per checkpoint. The uptick at the final step in both runs is batch noise, not divergence.

| v1 | step 600 | 800 | 1000 | **1200 (end)** |
|---|---|---|---|---|
| loss | 0.1862 | 0.1779 | 0.2091 | 0.2356 |

| v4 | step 1400 | 1600 | 1800 | **1900 (end)** |
|---|---|---|---|---|
| loss | 0.1139 | 0.1069 | 0.0995 | 0.1539 |

A third run (`v3`, 31,268 samples) **died at step 200 of 1369** with CUDA OOM at a
`max_seq_length` of 2048. No checkpoint was kept.

---

## Method

LoRA QLoRA via Unsloth, merged into fp16 before a single GGUF quantization pass.

**Deliberate choice:** merging into fp16 rather than the bnb-4bit training base.
Double-quantization measurably degrades exact-argument accuracy, which is precisely what's
being measured here. `scripts/merge_export.py:6`

| Setting | Value |
|---|---|
| Base | `unsloth/Qwen3-4B-unsloth-bnb-4bit` |
| Quantization | `load_in_4bit=True` |
| LoRA rank / alpha | 16 / 16 |
| Target modules | `q,k,v,o` + MLP projections |
| Effective batch | 16 (batch 2 × grad-accum 8) |
| Learning rate | 2e-4 |
| Epochs | 1 |
| Merge dtype | fp16 → GGUF `Q4_K_M` |

Values are `scripts/train_qlora_v2.py` defaults; the exact per-run overrides were not logged.

**Training data** — public function-calling corpora plus synthetic hard negatives:

| Source | Role |
|---|---|
| `glaive-function-calling-v2` | positive call examples |
| `toolace` | positive call examples |
| `when2call` | positive/negative boundary pairs |
| `APIGen` | multi-step chains — *used in v4 only, dropped after the regression* |
| synthetic | easy + hard negatives, the core of the v1 win |

---

## Layout

```
scripts/
  prep_data.py … prep_data_v4.py   dataset construction per version
  pretokenize.py                   tokenize + pack
  train_qlora.py, train_qlora_v2.py
  merge_export.py                  fp16 merge → GGUF (Q4_K_M/Q5_K_M/Q4_K_S)
  eval_toolcalls.py                the eval harness that produced eval/*.json
eval/                              10 result JSONs, every run that was measured
logs/                              training logs (not committed)
```

**What is deliberately not committed:** model weights, GGUF files, LoRA adapters, training data,
and the `hf_cache` / `repos` checkouts. That's ~32 GB. Adapters belong on Hugging Face;
`data/` is regenerable from `prep_data_*.py`.

## Reproducing

```bash
# 1. Data
uv pip install -r requirements.txt
python scripts/prep_data_v3.py          # → data/train_v3.jsonl, val_v3.jsonl

# 2. Tokenize
python scripts/pretokenize.py --train data/train_v3.jsonl --out data/tok_train

# 3. Train  (~9 h on a 6 GB card; watch your VRAM — see below)
python scripts/train_qlora_v2.py --dataset data/tok_train --out output/full4b_v3

# 4. Merge + quantize
python scripts/merge_export.py --adapter output/full4b_v3

# 5. Serve, then evaluate
ollama create qwen3-4b-tools-q4km -f Modelfile
python scripts/eval_toolcalls.py --model qwen3-4b-tools-q4km --out eval/myrun.json
```

`merge_export.py` expects a compiled `llama.cpp` at `repos/llama.cpp` with
`build/bin/llama-quantize` present. It is not vendored here.

## Hardware reality

Everything above ran on a single **RTX 4050 Laptop, 6141 MiB VRAM, 14 GB system RAM**.

PyTorch reports **5.64 GiB** usable — the Windows compositor and display output hold the rest.
This is a hard ceiling, and it's why:

- **4B is the largest trainable model here.** 8B QLoRA needs ~8 GB; that's ~2.5 GB over budget.
- **`max_seq_length=2048` OOMs.** The header comment in `train_qlora_v2.py:5` records 1024 as
  the safe value. The v3 run died this way at step 200.
- **Training is slow** — v4 took 9 h 19 m for 1900 steps.

The AMD Radeon 780M iGPU cannot be pooled with the NVIDIA card. Integrated GPUs carve VRAM out
of system RAM, VRAM does not pool across vendors, and ROCm does not support Windows. This torch
build (`2.14.0+cu130`) has no HIP support compiled in.

## Limitations

Stated plainly, because these numbers are easy to over-read:

1. **In-distribution only.** `eval_toolcalls.py` builds its cases from the same source families
   used for training. Held-out, but not out-of-distribution. It measures *did the model learn
   this data*, not *can the model use tools in the wild*.
2. **No public benchmark.** No Terminal-Bench, SWE-bench, BFCL, or τ-bench number exists for
   this model. On Terminal-Bench 2.0 a 4B model of this class would be expected near 0–2% —
   untuned `Qwen3-8B` scores 2.47% on that benchmark, and the published work that moved 8B to
   13.0% used a purpose-built terminal corpus. Function-calling ability and terminal-agentic
   ability are different skills; this repo only addresses the first.
3. **Self-reported eval harness.** The scorer in `eval_toolcalls.py` is mine, not a third-party
   one. It was not adversarially tested for false positives.
4. **Single seed, single run per configuration.** No variance estimate. The v1-vs-v4 gap
   (+17 points, 44 cases) is almost certainly real, but the ±2 point differences are not
   distinguishable from noise.
5. **Loss curves were not tuned.** No LR sweep, no epoch sweep, no seed sweep.
6. **Rejected model shipped.** v4 is included as evidence, not as a release.

## License

Base model `Qwen3-4B` is Apache-2.0. This fine-tune is offered under the same terms.
This is an unofficial derivative and is not affiliated with or endorsed by Alibaba/Qwen.