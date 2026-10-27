# Colab GPU rules (non-negotiable)

1. NEVER merge in a GPU session. Merge = 8-18 min. Use the prebuilt tars on Drive.
2. One session = one job. No parallel experiments.
3. Preflight with 3 rows BEFORE any long run. Fail in 2 min, not 40.
4. Checkpoint to Drive every N steps. Session death must cost minutes.
5. Merge + quantize on CPU sessions (free). GPU = training/inference only.
6. On session death: check Drive for the last artifact before re-running anything.
7. `colab stop -s <name>` the moment work is done. Quota burns while allocated.
8. If Colab throttles (GPU *and* CPU refused), fall back to Kaggle - 30h GPU/week
   and the kernel can pull Drive files directly (no dataset upload needed).
