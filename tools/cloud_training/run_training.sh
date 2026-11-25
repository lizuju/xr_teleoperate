#!/usr/bin/env bash
# One run on the rented GPU machine (AutoDL, 优云智算, ...): check the environment, unpack the
# code, export, train, evaluate on val. Start it so it keeps running after SSH disconnects:
#   cd ~/icecream52 && nohup bash run_training.sh > run.log 2>&1 &
#   tail -f run.log
# Running it again resumes: a finished export is reused, and training continues from latest.pt.
# Settings (environment variables):
#   IMAGE_HW=320x384   export image size (240x288 needs about half the disk)
#   STEPS=50000 BATCH=2 ACCUM=4 CHUNK=40   training (same as the guide in docs/icecream_il24.md)
#   NUM_WORKERS=4 DEVICE=auto PY=python    data loader processes, cuda/cpu, Python to use
#   MATCH_UBUNTU_TORCH=1   install torch 2.5.1 / torchvision 0.20.1 (as on Ubuntu) if the image has another version
#   SPLIT=split.json TAG=52   which split file to export, and the tag in the dataset and run names
#   WAIT_PID=1234   after exporting, wait until that process (e.g. an earlier run) ends before training
set -euo pipefail

ROOT="${ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
IMAGE_HW="${IMAGE_HW:-320x384}"
STEPS="${STEPS:-50000}"; BATCH="${BATCH:-2}"; ACCUM="${ACCUM:-4}"; CHUNK="${CHUNK:-40}"
NUM_WORKERS="${NUM_WORKERS:-4}"; DEVICE="${DEVICE:-auto}"; PY="${PY:-python}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
SPLIT="${SPLIT:-split.json}"; TAG="${TAG:-52}"
NAME="${NAME:-icecream_act24_${TAG}_${IMAGE_HW}}"
CODE="$ROOT/xr_teleoperate"
RAW="$ROOT/raw/icecream_dex1"
DATA="$ROOT/datasets/icecream_24d_${TAG}_${IMAGE_HW}"
RUN="$ROOT/runs/$NAME"
cd "$ROOT"
stamp() { echo; echo "[$(date '+%F %T')] $*"; }

stamp "1/5 environment"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "no NVIDIA GPU visible"
if [[ "${MATCH_UBUNTU_TORCH:-1}" == "1" ]] && \
   [[ "$("$PY" -c 'import torch, torchvision; print(torch.__version__.split("+")[0], torchvision.__version__.split("+")[0])' 2>/dev/null)" != "2.5.1 0.20.1" ]]; then
  # Ubuntu loads the trained model with torch 2.5.1; train with the same version.
  # The PyPI Linux wheels of 2.5.1 are CUDA 12.4 builds; a newer NVIDIA driver runs them too. About 2.5 GB.
  echo "installing torch 2.5.1 / torchvision 0.20.1 to match Ubuntu"
  "$PY" -m pip install -q "torch==2.5.1" "torchvision==0.20.1"
fi
if ! "$PY" -c "import h5py, cv2" 2>/dev/null; then
  "$PY" -m pip install -q "h5py>=3.10" "opencv-python-headless==4.10.0.84"
fi
if [[ "$("$PY" -c 'import numpy; print(numpy.__version__.split(".")[0])')" != "1" ]]; then
  # The code was tested with numpy 1.26 (Ubuntu .venv-train); keep the same major version.
  "$PY" -m pip install -q "numpy==1.26.4"
fi
"$PY" - <<'PY'
import sys, torch, torchvision, h5py, cv2, numpy
print("python", sys.version.split()[0], "| torch", torch.__version__, "| torchvision", torchvision.__version__,
      "| cuda", torch.version.cuda, torch.cuda.is_available(), "| h5py", h5py.__version__, "| cv2", cv2.__version__,
      "| numpy", numpy.__version__)
major_minor = tuple(int(v) for v in torch.__version__.split("+")[0].split(".")[:2])
if major_minor < (2, 5):
    sys.exit("torch 2.5 or newer is required (Ubuntu runs the model with torch 2.5.1)")
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    print("gpu", torch.cuda.get_device_name(0), arch, "| this torch has kernels for", torch.cuda.get_arch_list())
    # GPU code built for sm_XY also runs on sm_XZ with Z >= Y (same major), e.g. sm_86 on an sm_89 RTX 4090.
    usable = [a for a in torch.cuda.get_arch_list() if a.startswith("sm_") and len(a) >= 5
              and int(a[3:-1]) == major and int(a[-1]) <= minor]
    if not usable:
        sys.exit(f"STOP: this PyTorch build has no kernels for {arch}. RTX 50-series and RTX PRO 6000 need a "
                 "PyTorch 2.7+ / CUDA 12.8 image; for torch 2.5.1 (as on Ubuntu) rent an RTX 4090 / 4090D / 3090 class GPU.")
else:
    print("WARNING: no CUDA GPU visible; training would run on the CPU")
PY

stamp "2/5 code and weights"
if [[ ! -d "$CODE" ]]; then tar xzf xr_teleoperate-main.tar.gz; fi
echo "code commit: $(cat CODE_COMMIT)"
mkdir -p ~/.cache/torch/hub/checkpoints
if [[ -f resnet18-f37072fd.pth ]]; then cp -n resnet18-f37072fd.pth ~/.cache/torch/hub/checkpoints/; fi

stamp "3/5 export ($IMAGE_HW)"
if [[ -f "$DATA/dataset.json" ]] && grep -q '"status": "complete"' "$DATA/dataset.json"; then
  echo "export already complete: $DATA"
else
  if [[ -e "$DATA" ]]; then
    echo "STOP: $DATA exists but is not a complete export (an earlier export stopped). Delete it and run again." >&2
    exit 3
  fi
  "$PY" - "$ROOT" "$RAW" "$DATA" "$IMAGE_HW" "$SPLIT" <<'PY'
import json, shutil, sys
from pathlib import Path
root, raw, data, hw = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
h, w = (int(v) for v in hw.lower().split("x"))
split = json.loads((root / sys.argv[5]).read_text())
print("split file:", sys.argv[5], "| train", len(split["train"]), "val", len(split["val"]), "test", len(split["test"]))
names = split["train"] + split["val"] + split["test"]
missing = [n for n in names if not (raw / n / "frames.jsonl").is_file()]
if missing:
    sys.exit(f"STOP: {len(missing)} episodes missing under {raw}: {missing[:3]}")
# About 0.97 MB per exported frame at 320x384 (gzip level 1), scaled by image area, plus 15%.
need = sum(split["exportable_frames"][n] for n in names) * 0.97e6 * (h * w) / (320 * 384) * 1.15
data.parent.mkdir(parents=True, exist_ok=True)
free = shutil.disk_usage(data.parent).free
print(f"{len(names)} episodes; export needs about {need / 1e9:.0f} GB, {free / 1e9:.0f} GB free")
if free < need:
    sys.exit("STOP: not enough disk. Expand the AutoDL data disk, or run with IMAGE_HW=240x288.")
code = root / "xr_teleoperate"
sys.path[:0] = [str(code), str(code / "tools")]
from export_r1_training_dataset import export_dataset
export_dataset(raw, data, include_sources=names, val_sources=split["val"], test_sources=split["test"],
               image_hw=(h, w))
print("export complete:", data)
PY
fi

if [[ -n "${WAIT_PID:-}" ]] && kill -0 "$WAIT_PID" 2>/dev/null; then
  stamp "export done; waiting for process $WAIT_PID to finish before training"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
fi

stamp "4/5 train -> $RUN"
cd "$CODE"
resume=()
if [[ -f "$RUN/latest.pt" ]]; then resume=(--resume); echo "resuming from $RUN/latest.pt"; fi
# shellcheck disable=SC2086
"$PY" tools/train_r1_act.py "$DATA" --output "$RUN" "${resume[@]}" \
  --steps "$STEPS" --batch-size "$BATCH" --accumulate "$ACCUM" --chunk-size "$CHUNK" \
  --num-workers "$NUM_WORKERS" --device "$DEVICE" $EXTRA_TRAIN_ARGS

stamp "5/5 evaluate best.pt on the validation episodes"
"$PY" tools/eval_r1_act.py "$RUN/best.pt" --split val --batch-size 16 --num-workers "$NUM_WORKERS" --device "$DEVICE"
# The test split stays untouched until a final model is chosen (see docs/icecream_il24.md).

stamp "done"
ls -la "$RUN"
echo "Download to Ubuntu with download.sh; the files to take are best.pt, run.json, metrics.jsonl, eval_val.json."
echo "Shut the rented instance down when finished."
