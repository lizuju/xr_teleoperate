"""Stage labels for segment-by-segment ice-cream recordings, on synthetic episodes."""
import contextlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import label_icecream_stages as labels

HZ = 40.0
OPEN, CUP = 5.4, 4.3  # gripper rad: open, closed on the cup


def ramp(start, end, count, begin=0.3, finish=0.7):
    """Hold start, move linearly between the fractions begin..finish, hold end."""
    values = []
    for index in range(count):
        fraction = min(1.0, max(0.0, (index / (count - 1) - begin) / (finish - begin)))
        values.append(start + (end - start) * fraction)
    return values


def write_episode(root, name, waist, gripper, left, outcome="success", gripper_device="dex1_1"):
    directory = Path(root) / name
    directory.mkdir(parents=True)
    count = len(waist)
    info = {"frequency": HZ, "end_effectors": {"left": "linker_o6", "right": gripper_device}}
    manifest = {"schema": "xr_teleop_episode_v2", "status": "complete", "outcome": outcome,
                "episode_id": int(name.split("_")[1]), "frame_count": count, "frames": "frames.jsonl",
                "info": info, "text": {}}
    (directory / "episode.json").write_text(json.dumps(manifest))
    with (directory / "frames.jsonl").open("w") as stream:
        for index in range(count):
            states = {"body": {"qpos": [waist[index], 0.0, 0.0]},
                      "left_ee": {"qpos": [left[index]] * 6},
                      "right_ee": {"qpos": [gripper[index]] if gripper_device == "dex1_1" else [0.5] * 6}}
            frame = {"idx": index, "states": states,
                     "sample": {"monotonic_ns": int(1e9 * (1000.0 + index / HZ)), "mode": "following"}}
            stream.write(json.dumps(frame) + "\n")
    return directory


RIGHT = -math.pi / 2


def full_session(root, start_id=0, outcome="success"):
    n = 400  # 10 s each
    names = []
    specs = [
        (ramp(0.0, 0.0, n), ramp(OPEN, CUP, n), ramp(0.1, 0.1, n)),
        (ramp(0.0, RIGHT, n), ramp(CUP, CUP, n), ramp(0.1, 0.1, n)),
        (ramp(RIGHT, RIGHT, n), ramp(CUP, CUP, n), [0.1 + 0.7 * math.sin(math.pi * i / n) for i in range(n)]),
        (ramp(RIGHT, 0.0, n), ramp(CUP, OPEN, n, 0.8, 0.95), ramp(0.1, 0.1, n)),
    ]
    for offset, (waist, gripper, left) in enumerate(specs):
        name = f"episode_{start_id + offset:04d}_20261111_1000{offset:02d}"
        write_episode(root, name, waist, gripper, left, outcome)
        names.append(name)
    return names


class StageLabelTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "icecream_dex1"
        self.root.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_four_segments_get_their_stages_without_warnings(self):
        names = full_session(self.root)
        results = [labels.label_episode(self.root / name) for name in names]
        self.assertEqual([r["stage"] for r in results], [1, 2, 3, 4])
        self.assertEqual([r["name"] for r in results],
                         ["grasp_cup", "turn_right_deliver", "dispense", "turn_left_place"])
        for result in results:
            self.assertEqual(result["warnings"], [], result)
        self.assertEqual(results[1]["waist_deg"], [0.0, -90.0])
        self.assertEqual(results[1]["headings"], ["front", "right"])

    def test_front_episode_without_closing_the_gripper_is_not_a_grasp(self):
        n = 200
        write_episode(self.root, "episode_0000_x", ramp(0.0, 0.0, n), ramp(OPEN, OPEN, n), ramp(0.1, 0.1, n))
        result = labels.label_episode(self.root / "episode_0000_x")
        self.assertIsNone(result["stage"])
        self.assertIn("did not close", result["reason"])

    def test_a_left_turn_or_a_waist_between_stops_is_no_stage(self):
        n = 200
        write_episode(self.root, "episode_0000_x", ramp(0.0, math.pi / 2, n), ramp(CUP, CUP, n), ramp(0.1, 0.1, n))
        write_episode(self.root, "episode_0001_x", ramp(0.0, math.radians(-45), n), ramp(CUP, CUP, n), ramp(0.1, 0.1, n))
        self.assertIn("front -> left", labels.label_episode(self.root / "episode_0000_x")["reason"])
        self.assertIn("front -> between", labels.label_episode(self.root / "episode_0001_x")["reason"])

    def test_cross_checks_flag_a_missing_lever_pull_and_a_kept_cup(self):
        n = 200
        write_episode(self.root, "episode_0000_x", ramp(RIGHT, RIGHT, n), ramp(CUP, CUP, n), ramp(0.1, 0.1, n))
        write_episode(self.root, "episode_0001_x", ramp(RIGHT, 0.0, n), ramp(CUP, CUP, n), ramp(0.1, 0.1, n))
        dispense = labels.label_episode(self.root / "episode_0000_x")
        place = labels.label_episode(self.root / "episode_0001_x")
        self.assertEqual((dispense["stage"], place["stage"]), (3, 4))
        self.assertIn("lever", dispense["warnings"][0])
        self.assertIn("did not open", place["warnings"][0])

    def test_start_heading_can_be_offset(self):
        n = 200
        front = math.radians(40.0)
        write_episode(self.root, "episode_0000_x", ramp(front, front + RIGHT, n), ramp(CUP, CUP, n), ramp(0.1, 0.1, n))
        # Read against 0 deg, +40 -> -50 deg sits between stops; against the real start it is stage 2.
        self.assertIsNone(labels.label_episode(self.root / "episode_0000_x", 0.0)["stage"])
        self.assertEqual(labels.label_episode(self.root / "episode_0000_x", front)["stage"], 2)

    def test_two_o6_episode_is_labelled_by_waist_with_a_warning(self):
        n = 200
        write_episode(self.root, "episode_0000_x", ramp(0.0, RIGHT, n), ramp(CUP, CUP, n), ramp(0.1, 0.1, n),
                      gripper_device="linker_o6")
        result = labels.label_episode(self.root / "episode_0000_x")
        self.assertEqual(result["stage"], 2)
        self.assertIn("no Dex1-1 gripper angle", result["warnings"][0])
        # Staying at the front says nothing without the gripper, so it is not called a grasp.
        write_episode(self.root, "episode_0001_x", ramp(0.0, 0.0, n), ramp(CUP, CUP, n), ramp(0.1, 0.1, n),
                      gripper_device="linker_o6")
        front = labels.label_episode(self.root / "episode_0001_x")
        self.assertIsNone(front["stage"])
        self.assertIn("no Dex1-1 gripper angle", front["reason"])

    def test_cli_writes_labels_and_links_successful_episodes_per_stage(self):
        good = full_session(self.root)
        failed = full_session(self.root, start_id=4, outcome="failure")
        link_dir = Path(self.tmp.name) / "stages"
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(labels.main([str(self.root), "--write", "--link-dir", str(link_dir)]), 0)
        self.assertIn("grasp_cup: 1", out.getvalue())
        saved = json.loads((self.root / "stage_labels.json").read_text(encoding="utf-8"))
        self.assertEqual([e["stage"] for e in saved["episodes"]], [1, 2, 3, 4] * 2)
        folders = sorted(path.name for path in link_dir.iterdir())
        self.assertEqual(folders, ["stage1_grasp_cup", "stage2_turn_right_deliver",
                                   "stage3_dispense", "stage4_turn_left_place"])
        for folder, name in zip(folders, good):
            linked = sorted(link_dir.joinpath(folder).glob("episode_*/episode.json"))
            self.assertEqual([path.parent.name for path in linked], [name])
            self.assertEqual(linked[0].parent.resolve(), (self.root / name).resolve())
        self.assertFalse(any((link_dir / f).joinpath(n).exists() for f in folders for n in failed))
        # Running again is idempotent.
        with contextlib.redirect_stdout(io.StringIO()):
            labels.main([str(self.root), "--link-dir", str(link_dir)])

    def test_link_dir_must_not_be_the_task_directory(self):
        full_session(self.root)
        with self.assertRaises(ValueError), contextlib.redirect_stdout(io.StringIO()):
            labels.main([str(self.root), "--link-dir", str(self.root)])

    def test_unreadable_episode_is_reported_not_fatal(self):
        full_session(self.root)
        broken = self.root / "episode_0009_broken"
        broken.mkdir()
        (broken / "episode.json").write_text("{not json")
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(labels.main([str(self.root)]), 0)
        self.assertIn("unreadable", out.getvalue())


if __name__ == "__main__":
    unittest.main()
