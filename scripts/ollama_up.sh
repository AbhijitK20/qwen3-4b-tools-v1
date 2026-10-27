#!/usr/bin/env bash
# Start ollama detached so it survives the calling shell, then wait until ready.
set -u
export OLLAMA_HOST=127.0.0.1:11434
BIN="$HOME/ollama/bin/ollama"
LOG="/home/abhijitk20/model training/logs/ollama_serve.log"

if pgrep -f "ollama serve" >/dev/null 2>&1; then
  echo "ollama already running"
else
  mkdir -p "$(dirname "$LOG")"
  setsid nohup "$BIN" serve >"$LOG" 2>&1 < /dev/null &
  disown || true
fi

for _ in $(seq 1 40); do
  if curl -s --max-time 3 http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
    echo "ollama ready"
    exit 0
  fi
  sleep 1
done

echo "ollama failed to become ready"
tail -5 "$LOG"
exit 1