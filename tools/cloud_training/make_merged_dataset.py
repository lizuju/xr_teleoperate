"""Run 6 dataset: run 5's training set plus a separate export of new recordings, without exporting again.

Runs on the GPU machine. Usage:
  python make_merged_dataset.py OLD HR NEW DST CORRECTION_COPIES CORRECTIONS_TXT
OLD  the complete round 2 export (71 recordings). Its val/test splits and its normalization are kept, so
     validation numbers compare directly with run 5 and the model's inputs are scaled exactly as run 5's.
HR   run 5's dataset (OLD minus the 14 low-reach train recordings); its train split is the base.
NEW  a complete export of the new recordings; every segment it exported goes to train.
Each segment of a recording named in CORRECTIONS_TXT (the DAgger corrections) appears CORRECTION_COPIES
times in train: the extra episode entries point at the same HDF5 file, so it is sampled that much more
often. Every data file in DST is a symlink; nothing is copied or exported again.
"""
import json
import sys
import time
from pathlib import Path

if len(sys.argv) != 7:
    sys.exit(__doc__)
old, hr, new = (Path(a).resolve() for a in sys.argv[1:4])
dst = Path(sys.argv[4]).absolute()
copies = int(sys.argv[5])
corrections = set(Path(sys.argv[6]).read_text().split())
if copies < 1:
    sys.exit("STOP: CORRECTION_COPIES must be at least 1")
if dst.exists():
    sys.exit(f"STOP: {dst} already exists")


def load(path):
    meta = json.loads((path / "dataset.json").read_text())
    if meta.get("status") != "complete" or meta.get("schema") != "r1_act_hdf5_v4":
        sys.exit(f"STOP: {path} is not a complete r1_act_hdf5_v4 export")
    return meta


O, H, N = load(old), load(hr), load(new)
same = ["state_dim", "action_dim", "joint_groups", "camera_names", "image_size_hw", "image_transform_kind",
        "image_transform", "image_color_order", "image_color_profile", "training_inputs", "imu_used", "units",
        "fps", "joint_names", "task", "end_effectors", "time_limits_ms", "min_segment_frames",
        "controller_config", "gripper_position_limits", "phase_annotations", "action_semantics",
        "observation_semantics"]
differ = [key for key in same if O.get(key) != N.get(key)]
if differ:
    sys.exit(f"STOP: the new export differs from the old one in {differ}")
for part in ("val", "test"):
    if sorted(H["splits"][part]) != sorted(O["splits"][part]):
        sys.exit(f"STOP: run 5's {part} split differs from the round 2 export")
if H["normalization"] != O["normalization"]:
    sys.exit("STOP: run 5's normalization differs from the round 2 export")
new_names = {s["source_id"]: Path(s["episode"]).name for s in N["sources"]}
exported = {e["source_id"] for e in N["episodes"]}
missing = sorted(corrections - {new_names[i] for i in exported})
if missing:
    sys.exit(f"STOP: corrections with no exported segment: {missing}")
old_names = {Path(s["episode"]).name for s in O["sources"]}
overlap = sorted(old_names & set(new_names.values()))
if overlap:
    sys.exit(f"STOP: recordings in both exports: {overlap}")

dst.mkdir(parents=True)
for item in sorted(old.iterdir()):
    if item.name == "dataset.json":
        continue
    if item.is_dir():
        (dst / item.name).mkdir()
        for child in sorted(item.iterdir()):
            (dst / item.name / child.name).symlink_to(child.resolve())
    else:
        (dst / item.name).symlink_to(item.resolve())

meta = json.loads(json.dumps(O))
source_offset = max(s["source_id"] for s in O["sources"]) + 1
next_id = max(e["episode_id"] for e in O["episodes"]) + 1
for source in N["sources"]:
    entry = dict(source)
    entry["source_id"] = source["source_id"] + source_offset
    report = source.get("quality_report")
    if report:
        link = dst / "quality" / f"source_{entry['source_id']:04d}.json"
        link.parent.mkdir(exist_ok=True)
        if link.exists() or link.is_symlink():
            sys.exit(f"STOP: {link} already exists")
        link.symlink_to((new / report).resolve())
        entry["quality_report"] = str(link.relative_to(dst))
    entry["merged_from"] = str(new)
    meta["sources"].append(entry)

added = {"demos": [], "corrections": [], "correction_copies": []}
frames = {"demos": 0, "corrections": 0}
for episode in N["episodes"]:
    entry = dict(episode)
    entry["episode_id"] = next_id
    entry["source_id"] = episode["source_id"] + source_offset
    name = f"episode_{next_id}.hdf5"
    if (dst / name).exists() or (dst / name).is_symlink():
        sys.exit(f"STOP: {name} already exists in {dst}")
    (dst / name).symlink_to((new / episode["file"]).resolve())
    entry["file"] = name
    meta["episodes"].append(entry)
    next_id += 1
    kind = "corrections" if new_names[episode["source_id"]] in corrections else "demos"
    added[kind].append(entry["episode_id"])
    frames[kind] += entry["anchors"]
    if kind == "corrections":
        for _ in range(copies - 1):
            copy = dict(entry, episode_id=next_id, duplicate_of=entry["episode_id"])
            meta["episodes"].append(copy)
            added["correction_copies"].append(next_id)
            next_id += 1

base = list(H["splits"]["train"])
meta["splits"] = {"train": base + added["demos"] + added["corrections"] + added["correction_copies"],
                  "val": list(O["splits"]["val"]), "test": list(O["splits"]["test"])}
base_frames = sum(e["anchors"] for e in O["episodes"] if e["episode_id"] in set(base))
meta["merge"] = {
    "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    "old_export": str(old), "train_base": str(hr), "new_export": str(new),
    "normalization": "round 2 export (unchanged, as in run 5)",
    "correction_copies": copies, "corrections": sorted(corrections),
    "train_anchors": {"run5_base": base_frames, "new_demos": frames["demos"],
                      "corrections_x1": frames["corrections"],
                      "corrections_weighted": frames["corrections"] * copies},
}
(dst / "dataset.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
total = base_frames + frames["demos"] + frames["corrections"] * copies
print(f"{dst}: train {len(meta['splits']['train'])} episodes "
      f"(run 5 base {len(base)}, new demos {len(added['demos'])}, corrections {len(added['corrections'])} x{copies}); "
      f"val {len(meta['splits']['val'])}, test {len(meta['splits']['test'])}")
print(f"train samples: run 5 base {base_frames}, new demos {frames['demos']}, "
      f"corrections {frames['corrections']} x{copies} = {frames['corrections'] * copies} "
      f"({100 * frames['corrections'] * copies / total:.1f}% of {total})")
