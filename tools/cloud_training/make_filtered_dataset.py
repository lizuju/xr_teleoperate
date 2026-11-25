"""Derive a training dataset that leaves some train episodes out of an existing export, without exporting again.

Runs on the GPU machine. Usage: python make_filtered_dataset.py SRC_DATASET DST_DATASET SPLIT_JSON
Every file of SRC except dataset.json is symlinked into DST. DST/dataset.json is SRC's, with splits.train
reduced to the exported episodes whose recording is in SPLIT_JSON's train list. val and test must be
unchanged, and SPLIT_JSON's train list must be a subset of SRC's. normalization stays SRC's (its full
train split), so the model's inputs are scaled exactly as round 2's were.
"""
import hashlib
import json
import sys
from pathlib import Path

src, dst, split_path = Path(sys.argv[1]).resolve(), Path(sys.argv[2]), Path(sys.argv[3])
meta = json.loads((src / "dataset.json").read_text())
if meta.get("status") != "complete":
    sys.exit(f"STOP: {src} is not a complete export")
split = json.loads(split_path.read_text())
name_of = {s["source_id"]: Path(s["episode"]).name for s in meta["sources"]}
source_of = {e["episode_id"]: e["source_id"] for e in meta["episodes"]}


def recordings(ids):
    return {name_of[source_of[i]] for i in ids}


for part in ("val", "test"):
    if recordings(meta["splits"][part]) != set(split[part]):
        sys.exit(f"STOP: the {part} split differs from {src}")
old_train = recordings(meta["splits"]["train"])
keep = set(split["train"])
if not keep <= old_train:
    sys.exit(f"STOP: train recordings not in {src}: {sorted(keep - old_train)}")
dropped = sorted(old_train - keep)
old_episodes = len(meta["splits"]["train"])
new_train = [i for i in meta["splits"]["train"] if name_of[source_of[i]] in keep]
if recordings(new_train) != keep:
    sys.exit("STOP: some kept recordings have no exported train episodes")
if dst.exists():
    sys.exit(f"STOP: {dst} already exists")
dst.mkdir(parents=True)
for path in sorted(src.iterdir()):
    if path.name != "dataset.json":
        (dst / path.name).symlink_to(path)
meta["splits"]["train"] = new_train
meta["derived_from"] = {
    "dataset": str(src),
    "manifest_sha256": hashlib.sha256((src / "dataset.json").read_bytes()).hexdigest(),
    "operation": (f"Leave {len(dropped)} train recordings out ({split_path.name}). Episode files are symlinks to "
                  "the source export; val, test and normalization are the source's."),
    "dropped_train_recordings": dropped,
    "previous": meta.get("derived_from"),
}
(dst / "dataset.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n")
print(f"{dst}: train {len(old_train)} -> {len(keep)} recordings, {old_episodes} -> {len(new_train)} exported "
      f"episodes; val {len(meta['splits']['val'])}, test {len(meta['splits']['test'])} episodes unchanged")
print("left out:", ", ".join(name[8:12] for name in dropped))
