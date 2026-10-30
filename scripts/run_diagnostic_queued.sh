#!/usr/bin/env bash
# Queue diagnostic_analysis.py after the running heatmap chain (PID 88781) finishes.

WATCH_PID=89510
DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$DIR/diagnostic_queued.log"
CONDA_SH="/Users/ricardoperez/miniconda3/etc/profile.d/conda.sh"

echo "[queue] $(date)  Watching PID $WATCH_PID (heatmap chain) — will run diagnostic when it exits." | tee "$LOG"

while kill -0 "$WATCH_PID" 2>/dev/null; do
    sleep 30
done

echo "[queue] $(date)  PID $WATCH_PID done. Running diagnostic_analysis.py now." | tee -a "$LOG"

# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate ode
cd "$DIR"
python3 diagnostic_analysis.py 2>&1 | tee -a "$LOG"
echo "[done] $(date)" | tee -a "$LOG"
