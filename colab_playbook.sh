#!/usr/bin/env bash
# Colab GPU efficiency playbook
#
# Rules that cost us ~3h of a 4h/day budget on 2026-10-20 (mostly redundant
# merging + long runs that died on the last step).
#
# Usage:
#   ./colab_playbook.sh preflight <script.py>   # 3-row smoke, 2 min, no waste
#   ./colab_playbook.sh train    <script.py>   # detached, Drive-checkpointed
#   ./colab_playbook.sh stop                    # release GPU immediately
#   ./colab_playbook.sh status
set -uo pipefail
COLAB="$HOME/.local/bin/colab"
SESSION="${COLAB_SESSION:-v9run}"
OUTDIR="${COLAB_OUT:-$HOME/eval_out}"          # big artifacts -> user storage
MODELS="${COLAB_MODELS:-/content}"

# Drive artefacts already prepared (do NOT re-merge):
#   v9-merged.tar              1Wpo8g2syqJHCcGchyfs_fjedh79DzA6X  (8.1GB, merged v9 fp16)
#   v9-heretic-model.tar       1L1tf2XG8HQb3ZZlz1mQqJ2QgqcGsCad_  (8.1GB, abliterated fp16)
#   v9-heretic-q4km.gguf      159_CGsdDQxaft_7W66ydI_phHwHJTWH-  (2.5GB)
#   v9-heretic-distil-q4km.gguf 16EdpA4iGKkYjOheZEHAyoqu6K2ckxR3a (2.5GB)
#   distil_adapter.tar.gz     1XBiKOl_gY8N0l28PyJJFyZi-yUedrxcO  (126MB)
#   distil_minimax.jsonl      1JiNKtF-c0rporEAA1_XAnh9xFrRRocrw  (2.9MB)

case "${1:-help}" in
  preflight)
    # 3 rows, 2 minutes. Catches broken deps, wrong paths, wrong dtype, OOM.
    SCRIPT="$2"
    echo "[preflight] uploading + running 3-row smoke on $SESSION"
    "$COLAB" exec -s "$SESSION" -f "$SCRIPT" --timeout 600
    ;;
  train)
    # Detached so the tool-call timeout can never kill the run.
    SCRIPT="$2"
    echo "[train] launching detached on $SESSION -> logs in $OUTDIR"
    "$COLAB" exec -s "$SESSION" --file "$SCRIPT" --timeout 30 2>/dev/null || true
    echo "[train] check progress:  $COLAB exec -s $SESSION -f <poller>"
    echo "[train] when done:       $COLAB stop -s $SESSION"
    ;;
  stop)
    # ALWAYS release the GPU when finished - quota burns while allocated.
    "$COLAB" stop -s "$SESSION"
    echo "[stop] session $SESSION terminated"
    ;;
  status)
    "$COLAB" ls
    "$COLAB" usage 2>/dev/null | grep -viE "update|silence"
    ;;
  *)
    sed -n '2,30p' "$0"
    ;;
esac
