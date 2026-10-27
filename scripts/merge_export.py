#!/usr/bin/env python3
"""Merge a LoRA adapter into the fp16 base and export a GGUF for Ollama eval.

Path: fp16 base + adapter -> merged fp16 -> GGUF Q4_K_M -> ollama create

Merging into fp16 (not the bnb-4bit training base) so quantization happens
once. Double quantization measurably hurts exact-argument tool-call accuracy.

Degrades gracefully: if the GGUF writer or ollama create fails, the merged
fp16 directory is still on disk and can be served by transformers instead.
Fully offline: HF_HUB_OFFLINE=1, all inputs are local paths.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

BASE = Path.home() / "model training"
OLLAMA_BIN = Path.home() / "ollama" / "bin" / "ollama"
GGUF_NAME = "qwen3-4b-tools-q4km"
QUANT_TYPES = ["Q4_K_M", "Q4_K_S", "Q5_K_M"]


def log(msg):
    print(f"[merge] {msg}", flush=True)


def find_fp16_base():
    """Locate the fp16 base snapshot downloaded by scripts/dl_fp16_base.py."""
    cands = sorted((BASE / "hf_cache/hub/models--unsloth--Qwen3-4B/snapshots").glob("*"))
    for c in cands:
        if c.is_dir() and (c / "config.json").exists() and \
                list(c.glob("*.safetensors")):
            return c
    raise SystemExit("[merge] fp16 base not found - run scripts/dl_fp16_base.py")


def find_adapter(path):
    p = Path(path)
    if p.is_dir() and (p / "adapter_config.json").exists():
        return p
    if p.is_dir():
        cks = sorted(
            (q for q in p.glob("checkpoint-*") if (q / "adapter_config.json").exists()),
            key=lambda q: int(q.name.split("-")[1]),
        )
        if cks:
            return cks[-1]
    raise SystemExit(f"[merge] no adapter found at {path}")


def merge_fp16(base_dir, adapter_dir, out_dir, scale=1.0):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log(f"loading fp16 base: {base_dir}")
    tok = AutoTokenizer.from_pretrained(str(base_dir))
    # CPU + float16: the 4050 has 5.6GB VRAM and an fp16 4B merge needs ~8GB,
    # so this deliberately stays off the GPU and off bf16 autocast.
    model = AutoModelForCausalLM.from_pretrained(
        str(base_dir), torch_dtype=torch.float16, device_map="cpu",
        low_cpu_mem_usage=True,
    )
    log(f"applying adapter: {adapter_dir}")
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    if scale != 1.0:
        log(f"scaling adapter delta by {scale}")
        for module in model.modules():
            if hasattr(module, "lora_B") and hasattr(module, "scaling"):
                for key in module.scaling:
                    module.scaling[key] = module.scaling[key] * scale
    log("merging (this is the slow part, several minutes)")
    model = model.merge_and_unload()
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"saving merged fp16 -> {out_dir}")
    model.save_pretrained(str(out_dir), safe_serialization=True)
    tok.save_pretrained(str(out_dir))
    log("merged fp16 saved")
    return out_dir


def quantize(merged_dir, work_dir):
    """Merge fp16 -> f32 GGUF -> Q4_K_M. Returns gguf path or None."""
    merged_dir, work_dir = Path(merged_dir), Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    f32 = work_dir / "f16.gguf"

    log("fp16 safetensors -> f16 GGUF")
    try:
        import unsloth  # noqa: F401  (must precede transformers/peft)
        from unsloth import FastLanguageModel
        model, _ = FastLanguageModel.from_pretrained(
            model_name=str(merged_dir), max_seq_length=2048,
            load_in_4bit=False, dtype=None,
        )
        # save_pretrained_gguf is attached by unsloth_zoo.llama_cpp on load;
        # importing that submodule registers it on the model instances.
        import unsloth_zoo.llama_cpp  # noqa: F401
        log("unsloth gguf writer -> f16")
        model.save_pretrained_gguf(str(f32))
        del model
    except Exception as e:  # noqa: BLE001
        log(f"unsloth gguf path failed ({e}); trying llama.cpp converter")
        conv = BASE / "repos/llama.cpp/convert_hf_to_gguf.py"
        if not conv.exists():
            log("no convert_hf_to_gguf.py either - skipping GGUF")
            return None
        r = subprocess.run(
            [sys.executable, str(conv), str(merged_dir),
             "--outfile", str(f32), "--outtype", "f16"],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            log(f"convert failed: {r.stderr[-800:]}")
            return None
        log("convert_hf_to_gguf ok")

    if not f32.exists() or f32.stat().st_size < 10_000_000:
        log("f32 gguf missing/too small")
        return None

    quant_bin = BASE / "repos/llama.cpp/build/bin/llama-quantize"
    for qt in QUANT_TYPES:
        dst = merged_dir.parent / f"{merged_dir.name}-{qt.lower()}.gguf"
        log(f"quantizing {qt} -> {dst.name}")
        r = subprocess.run(
            [str(quant_bin), str(f32), str(dst), qt, "8"],
            capture_output=True, text=True,
        )
        if r.returncode == 0 and dst.exists() and dst.stat().st_size > 500_000_000:
            log(f"{qt} ok ({dst.stat().st_size / 1e9:.2f} GB)")
            f32.unlink(missing_ok=True)
            return dst
        log(f"{qt} failed: {(r.stderr or '')[-400:]}")
    return None


def ollama_create(gguf_path, name, merged_dir):
    if not OLLAMA_BIN.exists():
        log(f"ollama not at {OLLAMA_BIN}")
        return False
    modelfile = merged_dir.parent / "Modelfile"
    modelfile.write_text(
        f"FROM {gguf_path}\n"
        f"PARAMETER num_ctx 4096\n"
        f"PARAMETER temperature 0.6\n"
        f'PARAMETER stop ""\n'
        f'TEMPLATE "{{ if .Tools }}system\\n{{ . }}\\n'
        f'{{{{ range .Messages }}}}{{{{ .Role }}}}\\n{{{{ .Content }}}}\\n{{{{ end }}}}\\n'
        f'assistant\\n{{{{ end }}}}"\n'
    )
    log(f"ollama create {name}")
    env = dict(os.environ, OLLAMA_HOST="http://127.0.0.1:11434")
    r = subprocess.run([str(OLLAMA_BIN), "create", name, "-f", str(modelfile)],
                       capture_output=True, text=True, env=env, timeout=1800)
    if r.returncode != 0:
        log(f"ollama create failed: {(r.stderr or r.stdout)[-800:]}")
        return False
    log(f"ollama create ok -> {name}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=str(BASE / "output" / "full4b"))
    ap.add_argument("--out", default=str(BASE / "output" / "merged-4b"))
    ap.add_argument("--name", default=GGUF_NAME)
    ap.add_argument("--skip-gguf", action="store_true")
    ap.add_argument("--base", default=None,
                    help="explicit base model dir (e.g. a previously merged model)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="scale the LoRA delta before merging; <1.0 walks back toward the base")
    args = ap.parse_args()

    base_dir = Path(args.base) if args.base else find_fp16_base()
    adapter = find_adapter(args.adapter)
    out_dir = Path(args.out)
    log(f"base={base_dir.name} adapter={adapter} out={out_dir.name}")

    merge_fp16(base_dir, adapter, out_dir, args.scale)

    if args.skip_gguf:
        return 0
    gguf = quantize(out_dir, BASE / "output" / "gguf_work")
    if gguf is None:
        log("GGUF export unavailable; merged fp16 still usable via transformers")
        return 0
    ollama_create(gguf, args.name, out_dir)
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
