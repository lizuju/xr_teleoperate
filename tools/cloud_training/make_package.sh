#!/usr/bin/env bash
# Build a cloud training package on Ubuntu from a split file.
# Usage (from anywhere inside the xr_teleoperate checkout):
#   tools/cloud_training/make_package.sh SPLIT.json OUT_DIR
# OUT_DIR gets: split.json (a copy of SPLIT.json), episodes.txt (every train/val/test episode),
# CODE_COMMIT and xr_teleoperate-main.tar.gz (the committed code at HEAD), and the scripts and README.
# Then upload with OUT_DIR/upload.sh. See README.md in this folder.
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "Usage: $0 SPLIT.json OUT_DIR" >&2
  exit 2
fi
split="$1"; out="$2"
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo="$(git -C "$here" rev-parse --show-toplevel)"
if [[ -e "$out" ]]; then
  echo "STOP: $out already exists; choose a new folder." >&2
  exit 3
fi
python3 - "$split" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
for key in ("train", "val", "test", "exportable_frames"):
    if key not in s:
        sys.exit(f"STOP: {sys.argv[1]} has no '{key}' (see split.example.json)")
names = s["train"] + s["val"] + s["test"]
if len(set(names)) != len(names):
    sys.exit("STOP: an episode is listed twice (an episode belongs to exactly one of train/val/test)")
missing = [n for n in names if n not in s["exportable_frames"]]
if missing:
    sys.exit(f"STOP: no exportable_frames for {missing[:3]}")
print(f"split: train {len(s['train'])}, val {len(s['val'])}, test {len(s['test'])}")
PY
if ! git -C "$repo" diff --quiet HEAD -- . ':(exclude)teleop/teleimager' ':(exclude)teleop/televuer'; then
  echo "WARNING: the checkout has uncommitted changes; the package holds the committed HEAD only." >&2
fi
mkdir -p "$out"
cp "$split" "$out/split.json"
python3 -c "import json,sys; s=json.load(open(sys.argv[1])); print('\n'.join(s['train']+s['val']+s['test']))" \
  "$split" > "$out/episodes.txt"
git -C "$repo" rev-parse HEAD > "$out/CODE_COMMIT"
git -C "$repo" archive --format=tar.gz --prefix=xr_teleoperate/ -o "$out/xr_teleoperate-main.tar.gz" HEAD
cp "$here/run_training.sh" "$here/upload.sh" "$here/download.sh" "$here/make_filtered_dataset.py" \
  "$here/README.md" "$out/"
chmod +x "$out"/*.sh
echo "package: $out ($(wc -l < "$out/episodes.txt") episodes, code $(cut -c1-7 "$out/CODE_COMMIT"))"
echo "next: cd $out && ./upload.sh -p PORT USER@HOST"
