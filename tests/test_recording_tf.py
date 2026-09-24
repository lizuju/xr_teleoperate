import ast
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np

from teleop.utils.camera_calibration import load_camera_calibration
from teleop.utils.episode_writer import EpisodeWriter
from teleop.utils.recording_tf import R1RecordingTF, ARM_JOINTS, rigid_matrix, URDF_PATH
from teleop.utils.r1_capture import R1Capture
from tools.check_teleop_episode import validate_episode
from test_r1_capture import build

ROOT = Path(__file__).resolve().parents[1]


def robot_state():
    return {'q': [0.0]*14, 'waist_q': 0.0, 'head_q': [0.0, 0.0],
            'monotonic_ns': 1_000_000_000, 'sequence': 17}


class RecordingTFTests(unittest.TestCase):
    def setUp(self):
        self.tf = R1RecordingTF(frame_calibration_path=None)

    def test_turning_waist_rotates_both_arms_and_head_in_pelvis_frame(self):
        state = robot_state()
        zero = self.tf.sample(state, state['monotonic_ns'], True)
        state['waist_q'] = np.pi/2
        turned = self.tf.sample(state, state['monotonic_ns'], True)
        rotation = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        for frame in ('left_wrist_yaw_link', 'right_wrist_yaw_link', 'head_yaw_link'):
            np.testing.assert_allclose(turned['poses_in_root'][frame], rotation @ zero['poses_in_root'][frame], atol=1e-8)
        np.testing.assert_allclose(turned['poses_in_root']['pelvis_link'], np.eye(4))

    def test_actual_head_and_each_arm_move_independently(self):
        state = robot_state()
        before = self.tf.sample(state, state['monotonic_ns'], True)['poses_in_root']
        state['head_q'] = [.2, -.3]
        state['q'][0] = -.5
        after = self.tf.sample(state, state['monotonic_ns'], True)['poses_in_root']
        np.testing.assert_allclose(after['right_wrist_yaw_link'], before['right_wrist_yaw_link'])
        self.assertFalse(np.allclose(after['left_wrist_yaw_link'], before['left_wrist_yaw_link']))
        self.assertFalse(np.allclose(after['head_yaw_link'], before['head_yaw_link']))

    def test_camera_transform_direction_and_mirror_are_preserved(self):
        calibration = load_camera_calibration(ROOT/'assets/r1/camera_calibration.json')
        engine = R1RecordingTF(calibration, frame_calibration_path=None)
        sample = engine.sample(robot_state(), 1_000_000_000, True)
        for entry in calibration['hand_eye'].values():
            camera = entry['camera']+'_optical'
            expected = np.asarray(sample['poses_in_root'][entry['frame']]) @ rigid_matrix(entry, camera)
            np.testing.assert_allclose(sample['poses_in_root'][camera], expected)
        self.assertEqual(engine.metadata['static_transforms']['left_wrist_optical']['image_mirror'], 'horizontal')
        self.assertNotIn('world', sample['poses_in_root'])
        self.assertNotIn('left_tcp', sample['poses_in_root'])
        self.assertNotIn('table', sample['poses_in_root'])
        self.assertIn('left_tcp', engine.metadata['unavailable'])

    def test_measured_tcp_and_stationary_task_composition(self):
        document = {'schema': 'r1_recording_frames_v1', 'root_frame': 'pelvis_link',
                    'stationary_pelvis_confirmed': True,
                    'tcp': {'left': {'rotation': np.eye(3).tolist(), 'translation_m': [.08, .01, -.02]}},
                    'task_frames': {'table': {'rotation': np.eye(3).tolist(), 'translation_m': [.4, 0, -.2]}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'frames.json';path.write_text(json.dumps(document))
            engine = R1RecordingTF(frame_calibration_path=path)
            sample = engine.sample(robot_state(), 1_000_000_000, True)
            tcp = np.asarray(sample['poses_in_root']['left_tcp'])
            table = np.asarray(sample['poses_in_root']['table'])
            np.testing.assert_allclose(table @ sample['tcp_in_task']['table']['left'], tcp)
            document['stationary_pelvis_confirmed'] = False
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ValueError, 'stationary_pelvis'):
                R1RecordingTF(frame_calibration_path=path)

    def test_invalid_rotation_and_missing_or_stale_feedback(self):
        with self.assertRaises(ValueError):
            rigid_matrix({'rotation': np.diag([-1, 1, 1]), 'translation_m': [0, 0, 0]}, 'bad')
        self.assertFalse(self.tf.sample(None, 1_000_000_000, True)['valid'])
        state = robot_state()
        stale = self.tf.sample(state, 1_100_000_000, True)
        self.assertTrue(stale['valid']);self.assertFalse(stale['aligned_to_camera'])
        self.assertFalse(self.tf.sample(state, 1_000_000_000, False)['aligned_to_camera'])
        state['q'][0] = float('nan')
        result = self.tf.sample(state, 1_000_000_000, True)
        self.assertFalse(result['valid']);self.assertEqual(result['poses_in_root'], {})
        json.dumps(result, allow_nan=False)

    def test_capture_uses_aligned_feedback_without_changing_actions(self):
        now = time.monotonic_ns()
        arm, hand, loop, xr, image = build(now)
        aligned = copy.deepcopy(arm.data['state'])
        aligned['waist_q'] = .75
        aligned['sequence'] = 5
        aligned['monotonic_ns'] = now - 20_000_000
        original = arm.get_recording_samples
        arm.get_recording_samples = lambda *args: {**original(*args), 'nearest': aligned}
        image.timing = {'clock_valid': True, 'mapped_monotonic_ns': aligned['monotonic_ns'],
                        'clock_measured_monotonic_ns': now-100_000_000,
                        'clock_uncertainty_ns': 1000, 'stereo_skew_ns': 1000}
        capture = R1Capture(arm, hand, loop, .25, (16, 32), recording_tf=self.tf)
        frame = capture.frame(xr, image, 'following')
        self.assertEqual(frame['sample']['tf']['state_sequence'], 5)
        self.assertTrue(frame['sample']['tf']['aligned_to_camera'])
        self.assertEqual(frame['states']['body']['qpos'][0], .5)
        self.assertEqual(frame['actions']['left_arm']['qpos'], [.6]*7)
        np.testing.assert_allclose(frame['sample']['tf']['poses_in_root']['waist_yaw_link'],
                                   self.tf.sample(aligned,aligned['monotonic_ns'],True)['poses_in_root']['waist_yaw_link'])

    def test_writer_checker_round_trip_and_corrupt_transform_rejection(self):
        now = time.monotonic_ns()
        arm, hand, loop, xr, image = build(now)
        engine = R1RecordingTF(load_camera_calibration(ROOT/'assets/r1/camera_calibration.json'),
                                frame_calibration_path=None)
        capture = R1Capture(arm, hand, loop, .25, (16, 32), recording_tf=engine)
        with tempfile.TemporaryDirectory() as directory:
            writer = EpisodeWriter(directory, image_size=(16,16), rerun_log=False,
                                   metadata={'tf': engine.metadata})
            writer.create_episode();writer.add_item(**capture.frame(xr,image,'following'))
            writer.save_episode(outcome='success');writer.close()
            episode = writer.episode_dir
            report = validate_episode(episode)
            self.assertTrue(report['valid'], report['errors'])
            self.assertEqual(report['tf_valid_frames'], 1)
            self.assertEqual(report['tf_aligned_frames'], 0)
            path = episode/'frames.jsonl'
            frame = json.loads(path.read_text())
            frame['sample']['tf']['poses_in_root']['head_left_optical'][0][3] += .1
            path.write_text(json.dumps(frame)+'\n')
            report = validate_episode(episode)
            self.assertFalse(report['valid'])
            self.assertTrue(any('composition mismatch' in e for e in report['errors']),report['errors'])
            del frame['sample']['tf']
            path.write_text(json.dumps(frame)+'\n')
            self.assertFalse(validate_episode(episode)['valid'])

    def test_production_capture_receives_the_same_tf_object_as_metadata(self):
        tree = ast.parse((ROOT/'teleop/teleop_hand_and_arm.py').read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
        for name in ('capture_metadata', 'R1Capture'):
            call = next(c for c in calls if c.func.id == name)
            value = next(k.value for k in call.keywords if k.arg == 'recording_tf')
            self.assertIsInstance(value,ast.Name);self.assertEqual(value.id,'recording_tf')


if __name__ == '__main__':
    unittest.main()
