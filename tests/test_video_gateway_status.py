import ast
import json
import os
from pathlib import Path
import tempfile
import time
import unittest

source = (Path(__file__).resolve().parents[1] / 'deploy/visionpro-video/r1_webrtc_gateway.py').read_text()
node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'read_operator_status')
namespace = dict(json=json, os=os, time=time, Path=Path)
exec(compile(ast.Module(body=[node], type_ignores=[]), '<production read_operator_status>', 'exec'), namespace)

class GatewayStatusTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'status.json'
        namespace['Path'] = lambda ignored: self.path
        self.now = 10_000_000_000
        self.status = {'schema': 'r1_teleop_status_v1', 'run_id': 'test', 'sequence': 1,
                       'sample_monotonic_ns': self.now - 100_000_000,
                       'motion': 'following', 'recording': {'state': 'recording'}}

    def tearDown(self): self.temp.cleanup()
    def read(self): return namespace['read_operator_status'](self.now)
    def write(self): self.path.write_text(json.dumps(self.status))

    def test_missing_is_not_running(self):
        self.assertEqual(self.read(), {'available': False, 'reason': 'not_running'})

    def test_fresh_real_snapshot_is_returned_with_age(self):
        self.write()
        self.assertEqual(self.read(), {'available': True, 'age_ms': 100.0, 'status': self.status})

    def test_old_following_is_not_returned(self):
        self.status['sample_monotonic_ns'] = self.now - 1_000_000_001
        self.write()
        self.assertEqual(self.read(), {'available': False, 'reason': 'stale'})

    def test_future_clock_cannot_claim_live(self):
        self.status['sample_monotonic_ns'] = self.now + 1
        self.write()
        self.assertFalse(self.read()['available'])

    def test_invalid_and_oversized_data_cannot_claim_live(self):
        for value in ['{', '[]', '{}', ' ' * 32769]:
            self.path.write_text(value)
            self.assertEqual(self.read(), {'available': False, 'reason': 'invalid'})

    def test_invalid_motion_or_clock_type_cannot_claim_live(self):
        for field, value in [('motion', 'armed'), ('sample_monotonic_ns', True), ('sequence', '2')]:
            saved = self.status[field]
            self.status[field] = value
            self.write()
            self.assertFalse(self.read()['available'])
            self.status[field] = saved

if __name__ == '__main__': unittest.main(verbosity=2)
