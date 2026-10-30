#!/usr/bin/env bash
# Queue universality sweep after heatmap_polarized.py (PID 82193) finishes.

WATCH_PID=82193
LOG="$(dirname "$0")/sweep_P_rerun.log"
CONDA_SH="/Users/ricardoperez/miniconda3/etc/profile.d/conda.sh"

echo "[queue] $(date)  Watching PID $WATCH_PID (heatmap_polarized.py) — will start sweep when it exits." | tee "$LOG"

while kill -0 "$WATCH_PID" 2>/dev/null; do
    sleep 30
done
echo "[queue] $(date)  PID $WATCH_PID done. Starting sweep now." | tee -a "$LOG"

source "$CONDA_SH"
conda activate ode
cd "$(dirname "$LOG")"
python3 universality_test.py 2>&1 | tee -a "$LOG"
echo "[done] $(date)" | tee -a "$LOG"
