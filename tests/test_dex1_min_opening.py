import ast
import itertools
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import set_dex1_min_opening as setting


class MinimumOpeningSettingTest(unittest.TestCase):
    def test_saves_angle_and_zero_clears_the_limit(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(setting.Path, 'home', return_value=Path(directory)):
            path = Path(directory) / '.config/xr_teleoperate/dex1_min_q'
            setting.main(['--q', '2.4'])
            self.assertAlmostEqual(float(path.read_text()), 2.4)
            setting.main(['--q', '0'])
            self.assertEqual(float(path.read_text()), 0.0)

    def test_bad_values_preserve_saved_setting(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(setting.Path, 'home', return_value=Path(directory)):
            path = Path(directory) / '.config/xr_teleoperate/dex1_min_q'
            path.parent.mkdir(parents=True)
            path.write_text('2.4\n')
            for q in ('-0.1', 'nan', 'inf', '5.4', '6'):
                with self.subTest(q=q), self.assertRaises(SystemExit):
                    setting.main(['--q', q])
                self.assertEqual(path.read_text(), '2.4\n')

    def test_capture_saves_feedback_and_failure_preserves_previous_value(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(setting.Path, 'home', return_value=Path(directory)):
            path = Path(directory) / '.config/xr_teleoperate/dex1_min_q'
            with mock.patch.object(setting, 'capture_position', return_value=2.3) as capture:
                setting.main(['--capture'])
                capture.assert_called_once_with('eno1', 33)
            with mock.patch.object(setting, 'capture_position', side_effect=ValueError('moving')):
                with self.assertRaises(SystemExit):
                    setting.main(['--capture'])
            self.assertAlmostEqual(float(path.read_text()), 2.3)

    def test_read_only_capture_rejects_motion_and_closes_subscriber(self):
        for positions, success in (([2.3, 2.301, 2.299, 2.3], True), ([2.3, 2.35, 2.4, 2.45], False)):
            with self.subTest(positions=positions):
                subscriber = mock.Mock()
                def receive(callback, queue_len):
                    for q in positions:
                        motor = SimpleNamespace(q=q, motorstate=0, vol=38.0)
                        callback(SimpleNamespace(motor_state=[motor] * 35))
                subscriber.Init.side_effect = receive
                channel = SimpleNamespace(ChannelFactoryInitialize=mock.Mock(), ChannelSubscriber=mock.Mock(return_value=subscriber))
                modules = {'unitree_sdk2py.core.channel': channel,
                           'unitree_sdk2py.idl.unitree_hg.msg.dds_': SimpleNamespace(LowState_=object)}
                times = itertools.chain([1.0, 1.1, 1.2, 1.25], itertools.repeat(1.26))
                with mock.patch.dict(sys.modules, modules), mock.patch.object(setting.time, 'sleep'), mock.patch.object(setting.time, 'monotonic', side_effect=times):
                    if success:
                        self.assertAlmostEqual(setting.capture_position('eno1', 33), 2.3)
                    else:
                        with self.assertRaisesRegex(ValueError, 'moving'):
                            setting.capture_position('eno1', 33)
                subscriber.Close.assert_called_once()
                channel.ChannelSubscriber.assert_called_once_with('rt/lowstate', object)

    def test_live_feedback_validation_rejects_fault_and_sparse_data(self):
        for mode in ('fault', 'sparse'):
            with self.subTest(mode=mode):
                subscriber = mock.Mock()
                motor = SimpleNamespace(q=2.3, motorstate=1 if mode == 'fault' else 0, vol=38.0)
                subscriber.Init.side_effect = lambda callback, _: callback(SimpleNamespace(motor_state=[motor] * 35))
                modules = {'unitree_sdk2py.core.channel': SimpleNamespace(ChannelFactoryInitialize=mock.Mock(), ChannelSubscriber=mock.Mock(return_value=subscriber)),
                           'unitree_sdk2py.idl.unitree_hg.msg.dds_': SimpleNamespace(LowState_=object)}
                with mock.patch.dict(sys.modules, modules), mock.patch.object(setting.time, 'sleep'), mock.patch.object(setting.time, 'monotonic', return_value=1.0):
                    with self.assertRaises(ValueError):
                        setting.capture_position('eno1', 33)
                subscriber.Close.assert_called_once()

    def test_main_mapping_and_both_transports_receive_the_limit(self):
        tree = ast.parse((ROOT / 'teleop/teleop_hand_and_arm.py').read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id in ('Dex1PinchRetargeter', 'Dex1LowCmdGripper', 'Dex1GripperController')]
        self.assertEqual(len(calls), 3)
        for call in calls:
            minimum = next(k.value for k in call.keywords if k.arg == 'closed_q')
            self.assertEqual(ast.unparse(minimum), 'args.dex1_min_q')


if __name__ == '__main__':
    unittest.main()
