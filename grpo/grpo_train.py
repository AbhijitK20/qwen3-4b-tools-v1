import json, os, random, re, shutil, subprocess, sys, tempfile, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

MODEL_DIR = "/content/v9t-hf"
OUT_DIR = "/content/grpo-out"
os.makedirs(OUT_DIR, exist_ok=True)


def log(*a):
    m = " ".join(str(x) for x in a)
    print(m, flush=True)
    with open("/content/grpo.log", "a") as f:
        f.write(m + "\n")


class Sandbox:
    def __init__(self, task):
        self.d = tempfile.mkdtemp(prefix="rl_")
        for path, content in task.get("setup", {}).items():
            fp = os.path.join(self.d, path)
            os.makedirs(os.path.dirname(fp), exist_ok=True) if os.path.dirname(fp) else None
            open(fp, "w").write(content)
        if task.get("pre"):
            self.sh(task["pre"], 30)

    def sh(self, cmd, timeout=15):
        try:
            p = subprocess.run(["bash", "-lc", cmd], cwd=self.d, capture_output=True,
                               text=True, timeout=timeout, env={**os.environ, "HOME": self.d})
            return p.returncode, ((p.stdout or "") + (p.stderr or ""))[:800]
        except subprocess.TimeoutExpired:
            return 124, "[timeout]"
        except Exception as e:
            return 1, f"[err {e}]"

    def listing(self):
        out = []
        for root, _, files in os.walk(self.d):
            rel = os.path.relpath(root, self.d)
            for f in sorted(files)[:12]:
                p = os.path.join(root, f)
                try:
                    sz = os.path.getsize(p)
                    if sz < 400:
                        out.append(f"{rel}/{f} ({sz}B): " + open(p, errors="replace").read()[:180].replace("\n", "\\n"))
                    else:
                        out.append(f"{rel}/{f} ({sz}B)")
                except Exception:
                    pass
        return "\n".join(out[:18]) or "(empty)"

    def verify(self, script):
        rc, out = self.sh(script, 25)
        return rc == 0

    def close(self):
        shutil.rmtree(self.d, ignore_errors=True)


def parse_json(txt):
    txt = re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.M).strip()
    try:
        return json.loads(txt)
    except Exception:
        pass
    i, j = txt.find("{"), txt.rfind("}")
    if i >= 0 and j > i:
        try:
            return json.loads(txt[i:j + 1])
        except Exception:
            return None
    return None


@torch.no_grad()
def gen(model, tok, msgs, max_new=160, temp=0.9):
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt", truncation=True, max_length=1600).input_ids.to(model.device)
    out = model.generate(ids, max_new_tokens=max_new, do_sample=temp > 0, temperature=max(temp, 1e-5),
                         top_p=0.95, top_k=0, pad_token_id=tok.eos_token_id)
    return out[0].tolist(), ids.shape[1]


def rollout(policy, tok, task, max_turns=8, temp=0.9, max_new=160):
    sb = Sandbox(task)
    hist, turns = [], []
    try:
        for _ in range(max_turns):
            user = render(task, hist, sb.listing())
            msgs = [{"role": "system", "content": SYSP}, {"role": "user", "content": user}]
            toks, plen = gen(policy, tok, msgs, max_new=max_new, temp=temp)
            turns.append((toks, plen))
            raw = tok.decode(toks[plen:], skip_special_tokens=True)
            act = parse_json(raw)
            if not act or "commands" not in act:
                hist.append({"cmd": "<unparseable>", "out": raw[:200], "rc": 1})
                continue
            cmds = act.get("commands") or []
            if isinstance(cmds, str):
                cmds = [cmds]
            for c in cmds[:3]:
                if isinstance(c, str):
                    cmd = c
                elif isinstance(c, dict):
                    cmd = c.get("keystrokes") or c.get("command") or ""
                else:
                    continue
                if not isinstance(cmd, str) or not cmd.strip():
                    continue
                if any(b in cmd for b in ("sudo", "rm -rf /", "mkfs", "shutdown", ":(){")):
                    hist.append({"cmd": cmd, "out": "[blocked]", "rc": 1})
                    continue
                rc, out = sb.sh(cmd)
                hist.append({"cmd": cmd, "out": out, "rc": rc})
            if act.get("task_complete"):
                break
        ok = sb.verify(task["verify"])
        partial = False
        if not ok and task.get("partial"):
            rc, _ = sb.sh(task["partial"], 15)
            partial = (rc == 0)
        reward = 1.0 if ok else (0.5 if partial else 0.0)
        return dict(reward=reward, turns=turns, ok=ok, partial=partial, hist=hist)
    finally:
        sb.close()


def build_batch(tok, turns, device):
    ids, mask = [], []
    for toks, plen in turns:
        ids.extend(toks)
        mask.extend([False] * plen + [True] * (len(toks) - plen))
    ids = torch.tensor([ids], device=device)
    mask = torch.tensor([mask], device=device)
    return ids, mask


def seq_logprob(model, ids, mask, chunk=128):
    if ids.shape[1] < 2:
        return torch.zeros((), device=ids.device)
    out = model(input_ids=ids, use_cache=False)
    lg = out.logits[:, :-1]
    tgt = ids[:, 1:]
    m = mask[:, 1:].float()
    total = torch.zeros((), device=ids.device)
    for i in range(0, lg.shape[1], chunk):
        c = lg[:, i:i + chunk].float()
        t = tgt[:, i:i + chunk]
        lp = c.gather(-1, t.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(c, -1)
        total = total + (lp * m[:, i:i + chunk]).sum()
    del out, lg
    return total / m.sum().clamp(min=1)


def main(tasks, steps, G=4, lr=2e-5, beta=0.01, temp=0.9, max_turns=8, on_ckpt=None, ckpt_every=1):
    random.seed(0)
    torch.manual_seed(0)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    log("loading policy 4-bit")
    policy = AutoModelForCausalLM.from_pretrained(MODEL_DIR, quantization_config=bnb, device_map={"": 0})
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lcfg = LoraConfig(r=16, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    policy = get_peft_model(policy, lcfg)
    policy.gradient_checkpointing_enable()
    policy.enable_input_require_grads()
    policy.print_trainable_parameters()
    log("loading reference 4-bit")
    ref = AutoModelForCausalLM.from_pretrained(MODEL_DIR, quantization_config=bnb, device_map={"": 0})
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=lr)
    t0 = time.time()
    stats = []
    for step in range(steps):
        task = tasks[step % len(tasks)]
        policy.eval()
        rollouts = []
        for g in range(G):
            r = rollout(policy, tok, task, max_turns=max_turns, temp=temp, max_new=160)
            rollouts.append(r)
            log(f"  step{step} g{g} task={task['id']} reward={r['reward']:.1f} ok={r['ok']} partial={r['partial']} turns={len(r['turns'])}")
        rs = [r["reward"] for r in rollouts]
        if len(set(rs)) == 1:
            log(f"  step{step} no variance (all {rs[0]:.0f}) -> skip")
            stats.append(dict(step=step, task=task["id"], rewards=rs, skipped=True))
            continue
        mean = sum(rs) / len(rs)
        std = (sum((x - mean) ** 2 for x in rs) / len(rs)) ** 0.5 or 1.0
        adv = [(x - mean) / std for x in rs]
        policy.train()
        opt.zero_grad()
        loss_total = 0.0
        for r, a in zip(rollouts, adv):
            ids, mask = build_batch(tok, r["turns"], policy.device)
            if ids.shape[1] > 1536:
                ids, mask = ids[:, :1536], mask[:, :1536]
            with torch.no_grad():
                lp_ref = seq_logprob(ref, ids, mask).detach()
                torch.cuda.empty_cache()
            lp = seq_logprob(policy, ids, mask)
            ratio = (lp_ref - lp).clamp(-10, 10)
            kl = torch.exp(ratio) - ratio - 1
            loss = (-(a * lp) + beta * kl) / len(rollouts)
            loss.backward()
            loss_total += float(loss)
            del lp, ratio, kl, loss, ids, mask
            torch.cuda.empty_cache()
        gn = torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 1.0)
        nz = sum(1 for p in policy.parameters() if p.requires_grad and p.grad is not None and p.grad.norm().item() > 0)
        log(f"    grad_norm={float(gn):.4f} nonzero_param_grads={nz}")
        opt.step()
        stats.append(dict(step=step, task=task["id"], rewards=rs, adv=adv, loss=float(loss_total)))
        log(f"  step{step} UPDATE loss={float(loss_total):.4f} rewards={rs}")
        if (step + 1) % ckpt_every == 0 or step + 1 == steps:
            ck=f"{OUT_DIR}/step{step+1}"
            policy.save_pretrained(ck)
            json.dump(stats, open(f"{OUT_DIR}/stats.json", "w"), indent=1)
            log(f"  checkpoint saved {ck} ({time.time()-t0:.0f}s)")
            if on_ckpt:
                try: on_ckpt(step+1, ck)
                except Exception as e: log("  upload failed:", e)
    policy.save_pretrained(f"{OUT_DIR}/final")
    json.dump(stats, open(f"{OUT_DIR}/stats.json", "w"), indent=1)
    log("DONE", f"{time.time()-t0:.0f}s")


SYSP = """You are an AI agent controlling a bash terminal. You are given a task and the contents of the working directory.
Respond with ONLY a JSON object, no markdown fences, in exactly this schema:
{"analysis": "<what you observe>", "plan": "<what you will do next>", "commands": [{"keystrokes": "<bash command>", "duration": 0.1}], "task_complete": false}
Rules:
- Issue one or a few shell commands per turn. Each command runs in the working directory.
- When you believe the task is done, set "task_complete": true and use an empty commands list.
- Never use interactive programs, sudo, or network access."""


def render(task, history, listing):
    parts = [f"TASK: {task['prompt']}", "", "WORKING DIRECTORY:", listing, ""]
    if history:
        parts.append("TRANSCRIPT:")
        for h in history[-8:]:
            parts.append(f"$ {h['cmd']}")
            if h["out"].strip():
                parts.append(h["out"][:700])
            if h["rc"] != 0:
                parts.append(f"[exit {h['rc']}]")
        parts.append("")
    parts.append("Respond with the JSON object now.")
    return "\n".join(parts)
