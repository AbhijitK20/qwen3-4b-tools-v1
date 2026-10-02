#!/usr/bin/env python3
"""QLoRA fine-tune Qwen3-8B for tool-calling on the rendered dataset.

Usage:
    python3 scripts/train_qlora.py                 # full run (2 epochs, time-boxable)
    python3 scripts/train_qlora.py --max-steps 200 # quick sanity run

Output: output/qwen3-8b-toolcalling/  (LoRA adapters + tokenizer)
TensorBoard: logs/tb  ->  tensorboard --logdir logs/tb
"""
import argparse
import glob
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

BASE = Path.home() / "model training"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--out", default=str(BASE / "output" / "qwen3-8b-toolcalling"))
    args = ap.parse_args()

    model_path = glob.glob(str(BASE / "hf_cache/hub/models--unsloth--Qwen3-8B-unsloth-bnb-4bit/snapshots/*/"))[0]
    print(f"[1] loading base model: {model_path}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=args.seq_len,
        load_in_4bit=True,
        dtype=None,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_r * 2,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    print("[2] loading datasets (pre-rendered)")
    data_files = {
        "train": str(BASE / "data" / "train_rendered.jsonl"),
        "val": str(BASE / "data" / "val_rendered.jsonl"),
    }
    ds = load_dataset("json", data_files=data_files)

    print("[3] configuring trainer")
    cfg = SFTConfig(
        output_dir=args.out,
        dataset_text_field="text",
        max_seq_length=args.seq_len,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=0.03,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        logging_dir=str(BASE / "logs" / "tb"),
        report_to="tensorboard",
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        save_steps=300,
        save_total_limit=3,
        eval_strategy="no",
        seed=42,
    )

    trainer = SFTTrainer(model=model, tokenizer=tokenizer,
                         train_dataset=ds["train"], eval_dataset=None,
                         args=cfg)

    # train on assistant turns only (mask system/user/tool tokens)
    try:
        from unsloth.chat_templates import train_on_responses_only
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )
        print("    train_on_responses_only: ON (assistant-token loss)")
    except Exception as e:  # noqa: BLE001
        print(f"    train_on_responses_only unavailable ({e}); training on full text")

    print("[4] training")
    t0 = time.time()
    stats = trainer.train()
    dur = time.time() - t0
    print(f"[5] done in {dur/60:.1f} min | "
          f"steps={stats.global_step} | "
          f"final loss={stats.training_loss:.4f}")

    print("[6] saving adapters")
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
