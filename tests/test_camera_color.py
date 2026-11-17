import ast
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools"), str(ROOT / "tests")]
from teleop.utils.camera_color import IMAGE_COLOR_PROFILE, LEGACY_DEX1_SWAP_HASHES, recorded_red_blue_swaps
from teleop.utils.act_runtime import prepare_training_image
from teleop.utils.act_contract import validate_contract
from teleop.utils.icecream_task import source_digest


class CameraColorTests(unittest.TestCase):
    def test_live_o6_swaps_but_dex1_preserves_pixels_and_timing(self):
        tree = ast.parse((ROOT / "teleop/teleop_hand_and_arm.py").read_text())
        nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
                 and n.name in ("PalmFrame", "correct_palm_frame")]
        namespace = {"np": np}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "<colors>", "exec"), namespace)
        raw = np.full((4, 5, 3), [20, 50, 190], np.uint8)
        frame = SimpleNamespace(bgr=raw, sequence=12, received_monotonic_ns=123, timing={"source": 100})
        correct = namespace["correct_palm_frame"]
        o6 = correct(frame, "linker_o6")
        np.testing.assert_array_equal(o6.bgr[0, 0], [190, 50, 20])
        self.assertIs(o6.timing, frame.timing)
        self.assertEqual(o6.received_monotonic_ns, 123)
        for device in ("dex1", "dex1_1"):
            self.assertIs(correct(frame, device), frame)
            self.assertIsNone(correct(None, device))
        np.testing.assert_array_equal(frame.bgr[0, 0], [20, 50, 190])
        np.testing.assert_array_equal(prepare_training_image(frame.bgr, (32, 32))[16, 16], [190, 50, 20])

    def test_only_verified_legacy_dex1_is_corrected(self):
        for digest in LEGACY_DEX1_SWAP_HASHES:
            info = {"end_effectors": {"right": "dex1_1"},
                    "code_sha256": {"teleop/teleop_hand_and_arm.py": digest}}
            self.assertEqual(recorded_red_blue_swaps(info), ("color_3",))
            self.assertEqual(recorded_red_blue_swaps(dict(info, image_color_profile=IMAGE_COLOR_PROFILE)), ())
        self.assertEqual(recorded_red_blue_swaps({"end_effector": "linker_o6"}), ())
        with self.assertRaises(ValueError):
            recorded_red_blue_swaps({"end_effectors": {"right": "dex1_1"}})
        with self.assertRaises(ValueError):
            recorded_red_blue_swaps({"image_color_profile": "unknown"})

    def test_old_checkpoint_is_rejected_by_new_runtime(self):
        from test_icecream_il24 import metadata
        info = metadata()
        validate_contract(info)
        del info["image_color_profile"]
        with self.assertRaisesRegex(ValueError, "color profile"):
            validate_contract(info)

    def test_dataset_correction_preserves_every_other_array_and_sources(self):
        from correct_r1_dataset_colors import correct_dataset
        from test_icecream_il24 import metadata
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); episode = root / "episode"; episode.mkdir()
            info = {"end_effectors": {"right": "dex1_1"},
                    "code_sha256": {"teleop/teleop_hand_and_arm.py": sorted(LEGACY_DEX1_SWAP_HASHES)[0]}}
            (episode / "episode.json").write_text(json.dumps({"info": info}))
            (episode / "frames.jsonl").write_text('{}\n')
            source = root / "source"; source.mkdir(); (source / "quality").mkdir()
            (source / "quality/0.json").write_text('{"valid":true}')
            document = metadata(); del document["image_color_profile"]
            document.update(schema="r1_act_hdf5_v4", status="complete", sources=[{
                "source_id": 0, "episode": str(episode), "manifest_sha256": source_digest(episode / "episode.json"),
                "frames_sha256": source_digest(episode / "frames.jsonl")}],
                episodes=[{"episode_id": 0, "source_id": 0, "file": "episode_0.hdf5"}],
                splits={"train": [0], "val": [], "test": []}, normalization={"action_mean": list(range(24))})
            (source / "dataset.json").write_text(json.dumps(document))
            with h5py.File(source / "episode_0.hdf5", "w") as h:
                h.attrs['color_order'] = 'RGB'
                h['action'] = np.arange(48).reshape(2, 24)
                h['observations/qpos'] = np.arange(48).reshape(2, 24)
                h['source/frame_index'] = [3, 4]
                for camera in document['camera_names']:
                    h.create_dataset('observations/images/' + camera,
                        data=np.full((2, 32, 32, 3), [30, 60, 200], np.uint8), chunks=(1, 32, 32, 3),
                        compression='gzip', compression_opts=1)
            frozen = {p: source_digest(p) for p in [episode/'episode.json',episode/'frames.jsonl',source/'dataset.json',source/'episode_0.hdf5']}
            result = correct_dataset(source,root/'fixed')
            self.assertEqual(result['splits'],document['splits'])
            self.assertEqual(result['normalization'],document['normalization'])
            validate_contract(result)
            with h5py.File(source/'episode_0.hdf5') as old, h5py.File(root/'fixed/episode_0.hdf5') as new:
                def check(name,obj):
                    if isinstance(obj,h5py.Dataset):
                        expected = obj[:][...,::-1] if name=='observations/images/right_wrist' else obj[:]
                        np.testing.assert_array_equal(new[name][:],expected)
                old.visititems(check)
            self.assertEqual({p:source_digest(p) for p in frozen},frozen)
            with self.assertRaisesRegex(ValueError,'second correction'):
                correct_dataset(root/'fixed',root/'twice')

    def test_old_and_new_recordings_export_and_preview_the_same_correct_red(self):
        import base64
        import cv2
        import re
        from test_training_export import TrainingExportTests
        from test_icecream_il24 import metadata
        from export_r1_training_dataset import export_dataset
        from annotate_r1_icecream import review
        fixture = TrainingExportTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        original_hashes = {}
        for number in (0, 1):
            directory, manifest, rows = fixture.episode(number=number,count=50)
            spec = metadata()
            info = manifest['info']
            info.update(end_effectors=spec['end_effectors'],gripper_position_limits=spec['gripper_position_limits'],
                        launch_arguments=spec['controller_config'],
                        joint_names={g:[n.split('/')[1] for n in spec['joint_names'] if n.startswith(g+'/')]
                                     for g in spec['joint_groups']})
            if number == 0:
                info['code_sha256'] = {'teleop/teleop_hand_and_arm.py': sorted(LEGACY_DEX1_SWAP_HASHES)[0]}
            else:
                info['image_color_profile'] = IMAGE_COLOR_PROFILE
            for row in rows:
                row['states']['right_ee']['qpos'] = [4.9]
                row['actions']['right_ee']['qpos'] = [4.8]
                sample = row['sample']
                sample['aligned_states']['hands']['right']['q'] = [4.9]
                sample['commands']['hands']['requested']['right_q'] = [4.8]
                sample['commands']['hands']['published']['right']['q'] = [4.8]
                path = directory / row['colors']['color_3']
                cv2.imwrite(str(path),np.full((12,16,3),[200,60,20] if number==0 else [20,60,200],np.uint8))
                original_hashes[path] = source_digest(path)
            fixture.write(directory,manifest,rows)
        exported = export_dataset(fixture.source,fixture.output,min_frames=2,image_hw=(48,64))
        self.assertEqual([s['red_blue_swapped_on_export'] for s in exported['sources']],[['color_3'],[]])
        for episode in exported['episodes']:
            with h5py.File(fixture.output/episode['file']) as h:
                for camera in ('head_left','head_right','left_wrist','right_wrist'):
                    np.testing.assert_array_equal(h['observations/images/'+camera][0,24,32],[200,60,20])
        review(fixture.source,fixture.root/'review')
        page=(fixture.root/'review/review.html').read_text()
        for payload in re.findall(r'data:image/jpeg;base64,([^" ]+)',page):
            pixels=cv2.imdecode(np.frombuffer(base64.b64decode(payload),np.uint8),cv2.IMREAD_COLOR)
            blue,green,red=map(int,pixels[90,600])
            self.assertGreater(red-blue,150)
        self.assertEqual({p:source_digest(p) for p in original_hashes},original_hashes)
