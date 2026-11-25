#!/usr/bin/env bash
# Upload this package and its raw episodes (episodes.txt) from Ubuntu to the rented GPU machine.
# Usage, with the port, user and host from the platform's SSH login command:
#   ./upload.sh -p PORT USER@HOST
# Files go to ~/icecream52 on the remote machine (REMOTE_DIR=... to change, e.g. /root/autodl-tmp/icecream52 on AutoDL).
# Interrupted uploads resume when run again (rsync --partial).
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -ne 3 || "$1" != "-p" ]]; then
  echo "Usage: $0 -p PORT USER@HOST" >&2
  exit 2
fi
port="$2"; target="$3"
remote="${REMOTE_DIR:-icecream52}"   # relative paths are under the remote user's home
recordings="$HOME/unitree_r1_dev/teleop-recordings/icecream_dex1"
weights="$HOME/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
ssh_cmd="ssh -p $port -o ServerAliveInterval=30"

missing=0
while read -r name; do
  [[ -f "$recordings/$name/frames.jsonl" ]] || { echo "missing on Ubuntu: $name" >&2; missing=1; }
done < "$here/episodes.txt"
[[ $missing == 0 ]] || exit 3

$ssh_cmd "$target" "mkdir -p $remote/raw/icecream_dex1 && df -h $remote | tail -1"
echo "== package files"
rsync -av --partial -e "$ssh_cmd" \
  "$here/split.json" "$here/episodes.txt" "$here/CODE_COMMIT" "$here/xr_teleoperate-main.tar.gz" \
  "$here/run_training.sh" "$here/README.md" "$weights" "$target:$remote/"
echo "== $(wc -l < "$here/episodes.txt") episodes"
rsync -a -r --partial --info=progress2 -e "$ssh_cmd" \
  --files-from="$here/episodes.txt" "$recordings/" "$target:$remote/raw/icecream_dex1/"
echo "== check on the remote machine"
$ssh_cmd "$target" "cd $remote/raw/icecream_dex1 && echo \$(ls -d episode_* | wc -l) episodes, \$(du -sh . | cut -f1)"
echo "Uploaded. On the remote machine run: cd $remote && nohup bash run_training.sh > run.log 2>&1 &"
