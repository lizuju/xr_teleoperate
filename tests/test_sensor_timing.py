import ast
from collections import deque
import importlib.util
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from teleop.utils.frame_sync import CameraSynchronizer
from teleop.utils.r1_capture import R1Capture
from test_r1_capture import build

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('sensor_timing', ROOT / 'teleop/teleimager/src/teleimager/timing.py')
timing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timing)


class ClockTests(unittest.TestCase):
    def test_asymmetric_network_delay_within_reported_uncertainty(self):
        clock = timing.ClockMapping()
        # Source is 10 ms ahead; outbound 1 ms, server processing 2 ms, return 5 ms.
        self.assertTrue(clock.observe(100_000_000, 108_000_000,
                                     {'clock_id': 'pc2', 'receive_ns': 111_000_000, 'send_ns': 113_000_000}))
        result = clock.map({'clock_id': 'pc2', 'source_monotonic_ns': 110_000_000}, 109_000_000)
        self.assertTrue(result['clock_valid'])
        self.assertLessEqual(abs(result['mapped_monotonic_ns'] - 100_000_000), result['clock_uncertainty_ns'])
        self.assertFalse(clock.map({'clock_id': 'rebooted'}, 109_000_000)['clock_valid'])
        self.assertFalse(clock.map({'clock_id': 'pc2'}, 11_000_000_000)['clock_valid'])
        self.assertFalse(clock.observe(100, 110, {'camera': {}}))

    def test_timestamped_jpeg_remains_decodable_and_metadata_is_atomic(self):
        pixels = np.full((16, 16, 3), 93, np.uint8)
        jpeg = cv2.imencode('.jpg', pixels)[1].tobytes()
        metadata = {'source_monotonic_ns': 123, 'source_sequence': 7, 'clock_id': 'pc2',
                    'timestamp_kind': 'pc2_v4l2_dequeue'}
        stamped = timing.timestamp_jpeg(jpeg, metadata)
        self.assertEqual(timing.jpeg_timestamp(stamped), metadata)
        self.assertTrue(np.array_equal(cv2.imdecode(np.frombuffer(stamped, np.uint8), 1), pixels))
        self.assertIsNone(timing.jpeg_timestamp(jpeg))
        self.assertIsNone(timing.jpeg_timestamp(stamped[:30]))


class RecordingHistoryTests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse((ROOT / 'teleop/robot_control/robot_arm.py').read_text())
        namespace = {'np': np, 'threading': threading, 'time': time,
                     'R1_A7_Num_Motors': 35, 'R1_A7_JointArmIndex': list(range(14)),
                     'R1_A7_JointHeadIndex': [14, 15],
                     'R1_A7_JointIndex': SimpleNamespace(kWaistYaw=16)}
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name in ('MotorState', 'R1_A7_LowState', 'DataBuffer')]
        controller = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                          and node.name == 'R1_A7_ArmController')
        controller.body = [node for node in controller.body if isinstance(node, ast.FunctionDef)
                           and node.name in ('_ingest_motor_state', '_recording_state', 'get_recording_samples')]
        exec(compile(ast.Module(body=classes + [controller], type_ignores=[]), '<controller>', 'exec'), namespace)
        self.arm = namespace['R1_A7_ArmController']()
        self.arm.lowstate_sequence = 0
        self.arm.recording_lock = threading.Lock()
        self.arm.recording_history = deque(maxlen=3)
        self.arm.lowstate_buffer = namespace['DataBuffer']()
        self.message = SimpleNamespace(mode_machine=1, tick=77,
                                       motor_state=[SimpleNamespace(q=0.2, dq=0.1, tau_est=0.5)]*35,
                                       imu_state=SimpleNamespace(quaternion=[1, 0, 0, 0], gyroscope=[1, 2, 3],
                                                                 accelerometer=[4, 5, 6], rpy=[0.1, 0.2, 0.3], temperature=29))

    def test_full_imu_copy_batch_loss_and_no_future_feedback(self):
        for i in range(5):
            self.message.tick = 77 + i
            with patch.object(time, 'monotonic', return_value=10 + i / 100):
                self.arm._ingest_motor_state(self.message)
        self.message.imu_state.gyroscope[0] = 99
        result = self.arm.get_recording_samples(10_025_000_000, 1, 10_035_000_000)
        self.assertEqual([p['sequence'] for p in result['imu_packets']], [3, 4])
        self.assertEqual(result['dropped'], 1)
        self.assertEqual(result['imu_packets'][0]['gyroscope'], [1.0, 2.0, 3.0])
        self.assertEqual(result['imu_packets'][0]['temperature'], 29)
        self.assertLessEqual(result['nearest']['monotonic_ns'], 10_035_000_000)
        self.assertEqual(result['nearest']['tick'], 79)
        self.assertEqual(len(self.arm.get_recording_samples(10_040_000_000, None, 11_000_000_000)['imu_packets']), 1)


class CaptureTimingTests(unittest.TestCase):
    def test_source_time_drives_pairing_and_selected_head_pixels(self):
        now = time.monotonic_ns()
        arm, hand, loop, xr, head = build(now)
        capture = R1Capture(arm, hand, loop, .25, (16, 32), wrist_image_shapes={'left': (16, 32)})
        def image(seq, source_offset, arrival_offset, value, stereo=False):
            stamp = now + source_offset * 1_000_000
            return SimpleNamespace(sequence=seq, received_monotonic_ns=now + arrival_offset * 1_000_000,
                                   bgr=np.full((16, 32, 3), value, np.uint8),
                                   timing={'clock_valid': True, 'mapped_monotonic_ns': stamp,
                                           'clock_id': 'pc2', 'clock_uncertainty_ns': 1000,
                                           'clock_measured_monotonic_ns': now - 100_000_000,
                                           'stereo_skew_ns': 1000 if stereo else None})
        old = image(1, -100, -10, 11, True)
        new = image(2, -10, -1, 22, True)
        wrist = image(1, -99, -5, 33)
        capture.observe(old, {'left': wrist})
        result = capture.frame(xr, new, 'following', {'left': wrist})
        self.assertTrue(np.all(result['colors']['color_0'] == 11))
        self.assertEqual(result['color_sequences']['color_0'], 1)
        self.assertEqual(result['sample']['sources']['image']['sequence'], 1)
        self.assertEqual(result['sample']['camera_alignment']['timestamp_basis'], 'mapped_source')
        self.assertAlmostEqual(result['sample']['camera_alignment']['skew_ms'], 1)
        self.assertFalse(result['sample']['sensor_alignment']['usable'])  # feedback is 99 ms later
        self.assertEqual(result['actions']['left_arm']['qpos'], [0.6]*7)

    def test_missing_clock_is_preserved_but_not_called_synchronized(self):
        arm, hand, loop, xr, image = build(time.monotonic_ns())
        capture = R1Capture(arm, hand, loop, .25, (16, 32))
        sample = capture.frame(xr, image, 'following')['sample']
        self.assertFalse(sample['sensor_alignment']['usable'])
        self.assertEqual(sample['imu']['tick'], 100)
        self.assertEqual(len(sample['imu_packets']), 1)
        self.assertEqual(capture.frame(xr, image, 'following')['sample']['imu_packets'], [])


if __name__ == '__main__':
    unittest.main()

class SensorQualityTests(unittest.TestCase):
    def test_zero_imu_is_retained_but_not_usable_as_imu(self):
        now = time.monotonic_ns()
        arm, hand, loop, xr, image = build(now)
        arm.data['state']['imu'].update(quaternion=[0.0]*4, gyroscope=[0.0]*3,
                                        accelerometer=[0.0]*3, rpy=[0.0]*3, temperature=0, valid=False)
        image.timing = {'clock_valid': True, 'mapped_monotonic_ns': now,
                        'clock_measured_monotonic_ns': now, 'clock_uncertainty_ns': 1000,
                        'stereo_skew_ns': 1000}
        capture = R1Capture(arm, hand, loop, .25, (16, 32))
        sample = capture.frame(xr, image, 'following')['sample']
        self.assertTrue(sample['sensor_alignment']['usable'])
        self.assertFalse(sample['sensor_alignment']['imu_valid'])
        self.assertFalse(sample['sensor_alignment']['usable_with_imu'])
        self.assertEqual(sample['imu_packets'][0]['quaternion'], [0.0]*4)

    def test_checker_roundtrip_with_source_time_and_tampered_flags(self):
        import json
        import tempfile
        from teleop.utils.episode_writer import EpisodeWriter
        from tools.check_teleop_episode import validate_episode
        now = time.monotonic_ns()
        arm, hand, loop, xr, image = build(now)
        image.timing = {'clock_valid': True, 'mapped_monotonic_ns': now-1_000_000,
                        'clock_measured_monotonic_ns': now, 'clock_uncertainty_ns': 1000,
                        'stereo_skew_ns': 1000}
        capture = R1Capture(arm, hand, loop, .25, (16, 32))
        with tempfile.TemporaryDirectory() as directory:
            writer = EpisodeWriter(directory, image_size=(16, 16), rerun_log=False,
                                   metadata={'sensor_sync': {'schema': 'r1_sensor_sync_v1'},
                                             'frequency': 40, 'joint_names': {name: list(map(str, range(count)))
                                              for name, count in [('left_arm', 7), ('right_arm', 7),
                                                                  ('left_ee', 6), ('right_ee', 6), ('body', 3)]}})
            writer.create_episode()
            writer.add_item(**capture.frame(xr, image, 'following'))
            writer.save_episode(outcome='success'); writer.close()
            episode = writer.episode_dir
            report = validate_episode(episode)
            self.assertTrue(report['valid'], report['errors'])
            self.assertEqual(report['synchronized_with_imu_frames'], 1)
            path = episode/'frames.jsonl'
            frame = json.loads(path.read_text())
            frame['sample']['sensor_alignment']['usable'] = False
            path.write_text(json.dumps(frame)+'\n')
            self.assertFalse(validate_episode(episode)['valid'])
            del frame['sample']['sensor_alignment']
            path.write_text(json.dumps(frame)+'\n')
            self.assertFalse(validate_episode(episode)['valid'])

class CameraSourceTests(unittest.TestCase):
    def test_v4l2_and_stereo_stamp_the_matching_jpeg(self):
        tree = ast.parse((ROOT/'teleop/teleimager/src/teleimager/server.py').read_text())
        selected=[]
        keep={'BaseCamera': {'_write_timed_jpeg', 'get_jpeg_bytes'},
              'V4L2Camera': {'_update_frame'}, 'GstStereoRtpCamera': {'_update_frame'}}
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name in keep:
                node.body=[method for method in node.body if isinstance(method,ast.FunctionDef)
                           and method.name in keep[node.name]]
                selected.append(node)
        codec=SimpleNamespace(encode=lambda pixels: cv2.imencode('.jpg',pixels)[1].tobytes())
        ns={'np':np,'time':time,'MAX_FRAME_AGE':.25,'_turbojpeg':codec,'timestamp_jpeg':timing.timestamp_jpeg}
        exec(compile(ast.Module(body=selected,type_ignores=[]),'<camera>','exec'),ns)
        class Buffer:
            def write(self, value):self.value=value
            def read(self):return self.value
        for name in ('V4L2Camera','GstStereoRtpCamera'):
            camera=ns[name]()
            camera._enable_zmq=True;camera._enable_webrtc=False
            camera._source_sequence=0;camera._clock_id='pc2';camera._ready=threading.Event()
            camera._zmq_buffer=Buffer()
            if name=='V4L2Camera':
                camera.container=object();camera._passthrough=True
                camera._demux=iter([codec.encode(np.full((8,8,3),51,np.uint8))])
            else:
                now=time.monotonic()
                pair=[(now-.005,8,8,24,np.full((8,8,3),value,np.uint8).tobytes()) for value in (51,102)]
                pair[1]=(now-.003,*pair[1][1:])
                camera._capture=SimpleNamespace(frames=SimpleNamespace(take=lambda **kwargs:pair))
                camera._img_shape=(8,16)
            camera._update_frame()
            jpeg=camera.get_jpeg_bytes();meta=timing.jpeg_timestamp(jpeg)
            pixels=cv2.imdecode(np.frombuffer(jpeg,np.uint8),1)
            self.assertEqual(meta['source_sequence'],1)
            self.assertEqual(int(pixels[0,0,0]),51)
            self.assertLessEqual(meta['source_monotonic_ns'],time.monotonic_ns())
            if name=='GstStereoRtpCamera':
                self.assertEqual(int(pixels[0,-1,0]),102)
                self.assertAlmostEqual(meta['stereo_skew_ns']/1e6,2,places=3)
