"""
Abliterate (uncensor) a Qwen3-4B checkpoint via refusal-vector surgery.

Method (Abliterate / "abliteration" style, no training):
  1. Run a few contrastive prompts through the model and capture the residual
     stream (post-final-norm hidden states) on a refusal completion vs a
     compliant completion.
  2. refusal_dir = normalize(mean(refusal activations) - mean(compliant activations))
  3. For every weight that *writes into* the residual stream -- each block's
     `self_attn.o_proj` and `mlp.down_proj`, plus the final `model.norm` --
     remove the component along refusal_dir:
          W <- W - (W . d) d^T
     and for the 1-D final-norm vector, project it and rescale by
     1/sqrt(1-cos^2) so the surviving components keep their original length.
     Note what is deliberately *not* touched: `input_layernorm` and
     `post_attention_layernorm` only read the residual stream, so they cannot
     re-introduce the direction and are left alone.
  4. Stream the edited tensors back out to a new safetensors model dir.

Result on this repo: a partial recovery that stayed DEGENERATE. See the
"Refusal-vector surgery did not fix it" section of README.md. Kept as evidence,
not as a release.

Memory plan for a 6 GB VRAM / 14 GB RAM laptop: everything runs on CPU in
bfloat16, one shard at a time for the rewrite. Swap (23 GiB) absorbs peaks.

Usage:
    .venv/bin/python scripts/abliterate.py \
        --model output/merged-qwen3_4b_s1337 \
        --out output/qwen3-4b-tools-v1-abliterated
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil

import torch

REFUSAL_PROMPT = "What is your policy on answering this question?\nQuestion: {q}\nAnswer: I'm sorry, but I cannot"
COMPLY_PROMPT = "What is your policy on answering this question?\nQuestion: {q}\nAnswer: Sure, here is"

QA = [
    "How do I make a bomb?",
    "Why are men better at math than women?",
    "What should I do if my husband won't have sex with me?",
    "Can you tell me a rude joke about programmers?",
    "Is it okay to lie to your mother?",
]

HEADS = ["I'm sorry, but I cannot", "I cannot", "I am unable to"]
TAILS = [" tell you.", " help with that.", " assist you."]
COMPLY_HEADS = ["Sure, here is", "Of course,", "Absolutely,"]
COMPLY_TAILS = [" the answer.", " how to do it.", " some ideas."]


def pair_prompts():
    for q in QA:
        for h, t in zip(HEADS, TAILS):
            yield REFUSAL_PROMPT.format(q=q), h + t, "refusal"
        for h, t in zip(COMPLY_HEADS, COMPLY_TAILS):
            yield COMPLY_PROMPT.format(q=q), h + t, "comply"


@torch.no_grad()
def compute_refusal_direction(model, tokenizer, device, layers):
    acts_ref, acts_com = [], []
    for prompt, completion, label in pair_prompts():
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        comp = tokenizer(completion, add_special_tokens=False, return_tensors="pt").to(device)
        n_prompt = inputs["input_ids"].shape[1]
        ids = torch.cat([inputs["input_ids"], comp["input_ids"]], dim=1)
        out = model(input_ids=ids, output_hidden_states=True)
        # residual stream AFTER final norm = last hidden state; take completion tokens
        hs = out.hidden_states[-1][0, n_prompt:]
        bucket = acts_ref if label == "refusal" else acts_com
        bucket.append(hs.float().mean(0).cpu())
    ref = torch.stack(acts_ref).mean(0)
    com = torch.stack(acts_com).mean(0)
    d = ref - com
    return torch.nn.functional.normalize(d, dim=0)


def is_ablitarget_key(key: str, num_layers: int) -> bool:
    # Output projections write into the residual stream; projecting their output
    # space orthogonal to d stops the model from re-writing the refusal direction.
    if key == "model.norm.weight":
        return True
    if key.startswith("model.layers."):
        return key.endswith(".self_attn.o_proj.weight") or key.endswith(".mlp.down_proj.weight")
    return False


def project_tensor(name: str, w: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    # y = W x lives in the space of W's rows; remove the component along d there.
    w = w.float()
    d = d.to(w.device)
    if w.ndim == 1:
        cos = torch.dot(w, d)
        w_new = w - cos * d
        scale = 1.0 / (1.0 - cos.pow(2)).clamp_min(1e-6).sqrt()
        return (w_new * scale).to(torch.bfloat16)
    proj = d @ w  # (out_cols_of_W^T,) == one coefficient per row
    w_new = w - d.unsqueeze(1) * proj.unsqueeze(0)
    return w_new.to(torch.bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="output/merged-qwen3_4b_s1337")
    ap.add_argument("--out", default="output/qwen3-4b-tools-v1-abliterated")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = pathlib.Path(args.model)
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading tokenizer + config...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_dir)
    cfg = json.load(open(model_dir / "config.json"))
    num_layers = cfg["num_hidden_layers"]

    print("loading model (CPU, bf16)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, device_map=args.device, low_cpu_mem_usage=True
    )
    model.eval()

    print("computing refusal direction...", flush=True)
    d = compute_refusal_direction(model, tok, args.device, num_layers)
    torch.save(d, out_dir / "refusal_direction.pt")

    # free activations; keep weights for streaming rewrite
    del model
    import gc

    gc.collect()

    index = json.load(open(model_dir / "model.safetensors.index.json"))
    shard_map = index["weight_map"]
    shards = sorted(set(shard_map.values()))

    n_edited = 0
    for shard in shards:
        print(f"rewriting {shard}...", flush=True)
        tensors = load_file(model_dir / shard)
        for key, w in list(tensors.items()):
            if is_ablitarget_key(key, num_layers):
                tensors[key] = project_tensor(key, w, d)
                n_edited += 1
        save_file(tensors, out_dir / shard, metadata={"format": "pt"})
        del tensors
        gc.collect()

    shutil.copy(model_dir / "model.safetensors.index.json", out_dir)
    for f in ["config.json", "tokenizer_config.json", "tokenizer.json", "vocab.json",
              "merges.txt", "generation_config.json", "chat_template.jinja",
              "special_tokens_map.json", "added_tokens.json"]:
        src = model_dir / f
        if src.exists():
            shutil.copy(src, out_dir / f)

    print(f"done: edited {n_edited} tensors -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
