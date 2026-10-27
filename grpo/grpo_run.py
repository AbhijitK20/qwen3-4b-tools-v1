"""Sandboxed rollout + verifiable reward for terminal RL."""
import json, os, re, shutil, subprocess, sys, tempfile, time
import urllib.request
from grpo_env import TASKS, SYSTEM, render

OLLAMA = "http://127.0.0.1:11434"
SANDBOX_FILES = {"passwd", "shadow", "sudoers"}


def sh(cmd, cwd, timeout=20):
    try:
        p = subprocess.run(["bash", "-lc", cmd], cwd=cwd, capture_output=True,
                           text=True, timeout=timeout, env={**os.environ, "HOME": cwd})
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "[timeout]"
    except Exception as e:
        return 1, f"[error {e}]"


def make_sandbox(task):
    d = tempfile.mkdtemp(prefix="rl_")
    for path, content in task.get("setup", {}).items():
        fp = os.path.join(d, path)
        os.makedirs(os.path.dirname(fp), exist_ok=True) if os.path.dirname(fp) else None
        with open(fp, "w") as f:
            f.write(content)
    if task.get("pre"):
        sh(task["pre"], d, 30)
    return d


def listing(d, limit=700):
    out = []
    for root, dirs, files in os.walk(d):
        rel = os.path.relpath(root, d)
        for f in sorted(files)[:15]:
            p = os.path.join(root, f)
            try:
                sz = os.path.getsize(p)
                if sz < 400:
                    body = open(p, errors="replace").read()[:200].replace("\n", "\\n")
                    out.append(f"{rel}/{f} ({sz}B): {body}")
                else:
                    out.append(f"{rel}/{f} ({sz}B)")
            except Exception:
                pass
    return "\n".join(out[:20]) or "(empty)"


def parse_json(txt):
    txt = txt.strip()
    txt = re.sub(r"^```(?:json)?|```$", "", txt, flags=re.M).strip()
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


def llm(model, messages, temperature=0.6, seed=None):
    body = {"model": model, "messages": messages, "stream": False, "format": "json",
            "options": {"temperature": temperature, "num_ctx": 4096, "num_predict": 400}}
    if seed is not None:
        body["options"]["seed"] = seed
    req = urllib.request.Request(OLLAMA + "/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())["message"]["content"]


def rollout(task, model, max_steps=12, temperature=0.6, seed=None, verbose=False):
    d = make_sandbox(task)
    hist, steps, done = [], 0, False
    try:
        for _ in range(max_steps):
            msg = render(task, hist, listing(d))
            try:
                raw = llm(model, [{"role": "system", "content": SYSTEM},
                                  {"role": "user", "content": msg}], temperature, seed)
            except Exception as e:
                return dict(ok=False, reward=0.0, steps=steps, err=f"llm:{e}", hist=hist)
            act = parse_json(raw)
            if not act or "commands" not in act:
                hist.append(dict(cmd="<unparseable>", out=raw[:300], rc=1))
                steps += 1
                continue
            cmds = act.get("commands") or []
            if isinstance(cmds, str):
                cmds = [cmds]
            for c in cmds[:4]:
                if isinstance(c, str):
                    cmd = c
                elif isinstance(c, dict):
                    cmd = c.get("keystrokes") or c.get("command") or c.get("cmd") or ""
                else:
                    continue
                if not isinstance(cmd, str) or not cmd.strip():
                    continue
                if any(b in cmd for b in ("sudo", "rm -rf /", "mkfs", ":(){", "shutdown")):
                    hist.append(dict(cmd=cmd, out="[blocked]", rc=1))
                    continue
                rc, out = sh(cmd, d)
                hist.append(dict(cmd=cmd, out=out[:800], rc=rc))
                steps += 1
                if verbose:
                    print(f"    $ {cmd[:90]}  -> rc={rc}")
            if act.get("task_complete"):
                done = True
                break
        rc, out = sh(task["verify"], d, 30)
        return dict(ok=(rc == 0), reward=1.0 if rc == 0 else 0.0, steps=steps,
                    done=done, verify_out=out[-300:], hist=hist)
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "v9-terminal:latest"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else len(TASKS)
    res = []
    t0 = time.time()
    for task in TASKS[:n]:
        r = rollout(task, model, verbose=True)
        res.append((task["id"], r))
        print(f"  {task['id']:20s} reward={r['reward']:.0f} steps={r['steps']:2d} done={r.get('done')}")
    good = sum(r["reward"] for _, r in res)
    print(f"\n{model}: {good:.0f}/{len(res)} = {good/len(res)*100:.0f}%  ({time.time()-t0:.0f}s)")
