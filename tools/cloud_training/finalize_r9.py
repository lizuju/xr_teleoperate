"""Run 9/10 dataset: run 6/7's 72 recordings exported again with the segment bridge (segment-bridge-20261125).

Runs on the GPU machine. Usage: python finalize_r9.py EXPORT R6 DST CORRECTION_COPIES CORRECTIONS_TXT
EXPORT  a complete export of exactly run 6's recordings, with run 6's val/test sources named.
R6      run 6/7's dataset. Its source splits must match EXPORT's, and its normalization is copied, so the only
        difference from run 7/8 is how recordings are cut into segments.
Each segment of a correction recording appears CORRECTION_COPIES times in train (extra entries point at the
same HDF5 file), as in run 6. Every data file in DST is a symlink into EXPORT.
"""
import json
import sys
import time
from pathlib import Path

if len(sys.argv) != 6:
    sys.exit(__doc__)
export, r6 = (Path(a).resolve() for a in sys.argv[1:3])
dst = Path(sys.argv[3]).absolute()
copies = int(sys.argv[4])
corrections = set(Path(sys.argv[5]).read_text().split())
if dst.exists():
    sys.exit(f"STOP: {dst} already exists")
E, R = (json.loads((p / "dataset.json").read_text()) for p in (export, r6))
for name, meta in (("export", E), ("r6", R)):
    if meta.get("status") != "complete" or meta.get("schema") != "r1_act_hdf5_v4":
        sys.exit(f"STOP: {name} is not a complete r1_act_hdf5_v4 export")
same = ["state_dim", "action_dim", "joint_groups", "camera_names", "image_size_hw", "image_transform_kind",
        "image_transform", "image_color_order", "image_color_profile", "training_inputs", "imu_used", "units",
        "fps", "joint_names", "task", "end_effectors", "time_limits_ms", "min_segment_frames",
        "controller_config", "gripper_position_limits", "phase_annotations", "action_semantics",
        "observation_semantics"]
differ = [k for k in same if E.get(k) != R.get(k)]
if differ:
    sys.exit(f"STOP: the export differs from run 6's dataset in {differ}")


def split_names(meta):
    src = {s["source_id"]: Path(s["episode"]).name for s in meta["sources"]}
    ep = {e["episode_id"]: e["source_id"] for e in meta["episodes"]}
    return {k: sorted({src[ep[i]] for i in v}) for k, v in meta["splits"].items()}


e_names, r_names = split_names(E), split_names(R)
for part in ("train", "val", "test"):
    if e_names[part] != r_names[part]:
        sys.exit(f"STOP: {part} sources differ from run 6: {sorted(set(e_names[part]) ^ set(r_names[part]))}")
src = {s["source_id"]: Path(s["episode"]).name for s in E["sources"]}
missing = sorted(corrections - {src[e["source_id"]] for e in E["episodes"]})
if missing:
    sys.exit(f"STOP: corrections with no exported segment: {missing}")

dst.mkdir(parents=True)
for item in sorted(export.iterdir()):
    if item.name == "dataset.json":
        continue
    if item.is_dir():
        (dst / item.name).mkdir()
        for child in sorted(item.iterdir()):
            (dst / item.name / child.name).symlink_to(child.resolve())
    else:
        (dst / item.name).symlink_to(item.resolve())

meta = json.loads(json.dumps(E))
meta["normalization"] = R["normalization"]
next_id = max(e["episode_id"] for e in E["episodes"]) + 1
train_ids = set(E["splits"]["train"])
extra = []
for episode in E["episodes"]:
    if src[episode["source_id"]] in corrections and episode["episode_id"] in train_ids:
        for _ in range(copies - 1):
            meta["episodes"].append(dict(episode, episode_id=next_id, duplicate_of=episode["episode_id"]))
            extra.append(next_id)
            next_id += 1
meta["splits"] = {"train": list(E["splits"]["train"]) + extra, "val": list(E["splits"]["val"]),
                  "test": list(E["splits"]["test"])}
lengths = sorted(e["frames"] for e in E["episodes"])
meta["merge"] = {
    "created": time.strftime("%Y-%m-%d %H:%M:%S"), "export": str(export), "splits_and_normalization_from": str(r6),
    "segment_rule": "segment-bridge-20261125", "correction_copies": copies, "corrections": sorted(corrections),
    "segments": len(lengths), "run6_segments": sum(1 for e in R["episodes"] if "duplicate_of" not in e),
}
(dst / "dataset.json").write_text(json.dumps(meta, indent=2))
print(f"done: {dst} | {len(E['sources'])} recordings, {len(lengths)} segments (run 6: {meta['merge']['run6_segments']}), "
      f"median {lengths[len(lengths) // 2]} frames | train {len(meta['splits']['train'])} entries "
      f"({len(extra)} correction copies), val {len(meta['splits']['val'])}, test {len(meta['splits']['test'])}")
