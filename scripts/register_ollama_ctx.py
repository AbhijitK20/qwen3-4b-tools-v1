"""Re-register the serving models with a chosen num_ctx.

Ollama ignores the per-request options.num_ctx override for its context-limit
check, so the value has to be baked into the Modelfile at create time.
"""

import os
import subprocess
import sys

BASE = "/home/abhijitk20/model training"
OLLAMA = os.path.expanduser("~/ollama/bin/ollama")

MODELS = {
    "qwen3-4b-v6": f"{BASE}/output/v6-ctx-q4km.gguf",
    "qwen3-4b-v7": f"{BASE}/output/v7-ctx-q4km.gguf",
    "qwen3-4b-v7h": f"{BASE}/output/merged-v7h-q4_k_m.gguf",
}

TEMPLATE = (
    'TEMPLATE "{{ if .Tools }}system\n{{ . }}\n{{ range .Messages }}{{ .Role }}\n'
    '{{ .Content }}\n{{ end }}\nassistant\n{{ end }}"\n'
)


def main():
    ctx = int(sys.argv[1]) if len(sys.argv) > 1 else 32768
    modelfile = f"{BASE}/output/Modelfile.ctx"
    env = dict(os.environ, OLLAMA_HOST="http://127.0.0.1:11434")

    for name, gguf in MODELS.items():
        if not os.path.exists(gguf):
            print(f"  skip {name}: missing {gguf}")
            continue
        with open(modelfile, "w") as handle:
            handle.write(
                f"FROM {gguf}\nPARAMETER num_ctx {ctx}\nPARAMETER temperature 0.6\n" + TEMPLATE
            )
        result = subprocess.run(
            [OLLAMA, "create", name, "-f", modelfile],
            capture_output=True, text=True, env=env, timeout=1800,
        )
        ok = result.returncode == 0
        print(f"  {'OK  ' if ok else 'FAIL'} {name} (num_ctx={ctx})")
        if not ok:
            print(f"       {result.stderr[-200:] or result.stdout[-200:]}")


if __name__ == "__main__":
    main()