#!/usr/bin/env python3
"""QLoRA fine-tune Qwen3-8B for tool-calling — v2, repo-researched config.

Applied best practices (from LLaMA-Factory / Axolotl / TRL / Unsloth repos):
  - max_seq_length = 1024  (6GB VRAM envelope for 8B QLoRA; 2048 would OOM)
  - assistant_only_loss=True via TRL's patched Qwen3 training template
    (grades assistant turns incl. tool_calls; masks system/user/tool tokens)
  - structured messages+tools dataset (not pre-rendered) so the template
    applies generation markers; length filter computed WITH tools
  - unsloth GC + 4-bit NF4 double-quant; r=16, alpha=16, all 7 projections
  - LR 2e-4 cosine, warmup 5%, paged_adamw_8bit, batch 1 x accum 16, seed 3407
  - save every 200 steps (checkpoint selection matters: budget ~70% of epoch)

Usage:
    python3 scripts/train_qlora_v2.py --max-steps 2200
    python3 scripts/train_qlora_v2.py --max-steps 40   # sanity run

Output: output/qwen3-8b-tools/   (LoRA adapters)
TensorBoard: logs/tb2
"""
import argparse
import glob
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import load_from_disk
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

BASE = Path.home() / "model training"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=2200)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--model", default="unsloth/Qwen3-4B-unsloth-bnb-4bit")
    ap.add_argument("--resume", action="store_true",
                    help="resume from the newest checkpoint-* in --out")
    ap.add_argument("--no-gc", action="store_true",
                    help="disable gradient checkpointing (faster, uses more VRAM)")
    ap.add_argument("--offload-embedding", action="store_true",
                    help="keep the 151K-vocab embedding on CPU (slower; only if VRAM-bound)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--tok-prefix", default="",
                    help="dataset suffix: '' -> data/tok_train (v1), "
                         "'_v3' -> data/tok_train_v3 (augmented mix)")
    ap.add_argument("--out", default=str(BASE / "output" / "qwen3-8b-tools"))
    args = ap.parse_args()

    slug = args.model.replace("/", "--")
    model_path = glob.glob(str(BASE / f"hf_cache/hub/models--{slug}/snapshots/*/"))[0]
    print(f"[1] base model: {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=args.seq_len,
        load_in_4bit=True,
        dtype=None,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        offload_embedding=args.offload_embedding,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_r,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing=(False if args.no_gc else "unsloth"),
        random_state=3407,
    )

    print("[2] datasets (pre-tokenized, assistant-only labels)")
    # Unsloth's compiled trainer has no conversational (messages+tools) path, so
    # we tokenize offline (scripts/pretokenize.py) using TRL's Qwen3 training
    # template; labels are already masked to assistant tokens only.
    tok_train = load_from_disk(str(BASE / "data" / f"tok_train{args.tok_prefix}"))
    tok_val = load_from_disk(str(BASE / "data" / f"tok_val{args.tok_prefix}"))
    n0 = len(tok_train)
    if args.max_samples:
        tok_train = tok_train.shuffle(seed=3407).select(
            range(min(args.max_samples, len(tok_train))))
    ds = {"train": tok_train, "val": tok_val}
    sup = sum(int(sum(1 for v in lab if v != -100)) for lab in ds["train"][:1000]["labels"]) / 1000
    print(f"    train {n0} -> {len(ds['train'])} | val {len(ds['val'])} | "
          f"mean supervised tokens {sup:.0f}")

    print("[3] trainer config")
    cfg = SFTConfig(
        output_dir=args.out,
        max_length=args.seq_len,
        packing=False,               # keep sequences intact; lengths are bounded
        # assistant masking is pre-applied in the labels column by
        # scripts/pretokenize.py, so the trainer flag is not needed (and Unsloth
        # rejects it for pre-tokenized, non-conversational datasets)
        assistant_only_loss=False,
        eval_strategy="steps",
        eval_steps=250,
        per_device_train_batch_size=args.batch,
        # eval defaults to batch 8; on a 5.6GB card that OOMs at the first
        # eval step (full4b died at step 250 = eval_steps with 1.12GiB alloc fail)
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=0.05,
        num_train_epochs=1,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=10,
        logging_dir=str(BASE / "logs" / "tb2"),
        report_to="tensorboard",
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        save_steps=200,
        save_total_limit=4,
        gradient_checkpointing=not args.no_gc,
        dataset_num_proc=8,
        seed=3407,
    )

    trainer = SFTTrainer(model=model, tokenizer=tokenizer,
                         train_dataset=ds["train"], eval_dataset=ds["val"],
                         args=cfg)

    print("[4] training")
    resume = None
    if args.resume:
        cks = sorted((p for p in Path(args.out).glob("checkpoint-*")),
                     key=lambda q: int(q.name.split("-")[1]))
        if cks:
            resume = str(cks[-1])
            print(f"    resuming from {resume}")
    t0 = time.time()
    stats = trainer.train(resume_from_checkpoint=resume)
    dur = time.time() - t0
    print(f"[5] done in {dur/60:.1f} min | steps={stats.global_step} | "
          f"final loss={stats.training_loss:.4f}")

    print("[6] saving adapters")
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
