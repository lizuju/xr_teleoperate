#!/usr/bin/env bash
# Bring the trained model back from the rented machine to Ubuntu ~/unitree_r1_dev/training-runs/NAME/.
# Usage: ./download.sh -p PORT USER@HOST [NAME]   (NAME default: icecream_act24_52_320x384)
# Takes best.pt (about 200 MB) and the small records; latest.pt (only for resuming) stays remote.
set -euo pipefail
if [[ $# -lt 3 || "$1" != "-p" ]]; then
  echo "Usage: $0 -p PORT USER@HOST [NAME]" >&2
  exit 2
fi
port="$2"; target="$3"; name="${4:-icecream_act24_52_320x384}"
remote="${REMOTE_DIR:-icecream52}/runs/$name"
dest="$HOME/unitree_r1_dev/training-runs/$name"
if [[ -e "$dest/best.pt" ]]; then
  echo "STOP: $dest/best.pt already exists; choose another NAME or move it first." >&2
  exit 3
fi
mkdir -p "$dest"
rsync -av --partial -e "ssh -p $port -o ServerAliveInterval=30" \
  --include='best.pt' --include='run.json' --include='metrics.jsonl' --include='eval_*.json' --exclude='*' \
  "$target:$remote/" "$dest/"
ls -la "$dest"
echo "Shadow run on the robot (reads only, sends nothing):"
echo "  cd ~/unitree_r1_dev/xr_teleoperate && ./teleop/run_r1_a7_policy.sh $dest/best.pt --shadow-seconds 60 --img-server-ip 192.168.50.165"
