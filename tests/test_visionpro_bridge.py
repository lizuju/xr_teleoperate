import os
from pathlib import Path
import subprocess
import sys
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from teleop.utils.visionpro_source import VisionProMotionSource


SERVER = r'''
from concurrent.futures import ThreadPoolExecutor
import sys, time
import grpc
from visionpro_protocol import handtracking_pb2 as pb

mode = sys.argv[1]
def identity(matrix, x=0., y=0., z=0.):
    matrix.m00 = matrix.m11 = matrix.m22 = matrix.m33 = 1.
    matrix.m03, matrix.m13, matrix.m23 = x, y, z

connections = 0
def stream(request, context):
    global connections
    connections += 1
    connection = connections
    if request.Head.m00 != 888. or request.tracking_protocol_version != 1:
        context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Missing native discovery handshake")
    initial = time.monotonic()
    while context.is_active():
        now = time.monotonic()
        if (mode == 'disconnect' or (mode == 'reconnect' and connection == 1)) and now - initial > .25:
            return
        if mode == 'reconnect' and connection > 1:
            now -= 20.
        anchor = initial if mode == 'frozen' else now
        message = pb.HandUpdate(
            tracking_protocol_version=0 if mode == 'stock' else 1,
            sample_time=now, head_time=anchor, left_time=anchor, right_time=anchor,
            head_valid=True, left_valid=True, right_valid=True,
        )
        identity(message.Head, y=1.6)
        for side, x in (('left', -.3), ('right', .3)):
            hand = getattr(message, side + '_hand')
            identity(hand.wristMatrix, x, 1.25, -.6)
            for i in range(27):
                identity(hand.skeleton.jointMatrices.add(), x=i * .005)
        yield message
        time.sleep(.01)

server = grpc.server(ThreadPoolExecutor(max_workers=1))
server.add_generic_rpc_handlers((grpc.method_handlers_generic_handler(
    'handtracking.HandTrackingService', {
        'StreamHandUpdates': grpc.unary_stream_rpc_method_handler(
            stream, request_deserializer=pb.HandUpdate.FromString,
            response_serializer=pb.HandUpdate.SerializeToString,
        ),
    }),))
port = server.add_insecure_port('127.0.0.1:0')
server.start()
print(port, flush=True)
server.wait_for_termination()
'''


@unittest.skipUnless(os.environ.get("VISIONPRO_PYTHON"), "Set VISIONPRO_PYTHON to the isolated receiver Python")
class VisionProBridgeTest(unittest.TestCase):
    def connect(self, mode):
        python = os.environ["VISIONPRO_PYTHON"]
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "tools"))
        server = subprocess.Popen([python, "-u", "-c", SERVER, mode], env=env,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        def stop():
            server.terminate()
            server.wait(timeout=3.)
            server.stdout.close()
            server.stderr.close()
        self.addCleanup(stop)
        line = server.stdout.readline()
        self.assertTrue(line, server.stderr.read() if server.poll() is not None else "No port")
        source = VisionProMotionSource("127.0.0.1", python, int(line))
        self.addCleanup(source.close)
        return source

    def test_real_protobuf_stream_reaches_webxr_interface(self):
        source = self.connect("valid")
        snapshot = source.get_hand_motion_snapshot()
        self.assertTrue(snapshot["motion_data_ready"])
        self.assertEqual(snapshot["left_hand_positions"].shape, (25, 3))
        self.assertAlmostEqual(snapshot["left_hand_positions"][24, 0], -.18, places=5)
        self.assertEqual(snapshot["left_arm_pose"][0, 2], -1.)
        self.assertEqual(snapshot["right_arm_pose"][0, 2], 1.)
        self.assertTrue(source.get_tracking_diagnostics()["head_tracking"])

    def test_unpatched_app_is_rejected_before_input_is_available(self):
        with self.assertRaisesRegex(RuntimeError, "validity/timestamps"):
            self.connect("stock")

    def test_live_socket_with_frozen_anchors_cannot_keep_motion_valid(self):
        source = self.connect("frozen")
        time.sleep(.32)
        snapshot = source.get_hand_motion_snapshot()
        self.assertFalse(snapshot["motion_data_ready"])
        self.assertEqual(snapshot["left_hand_timestamp"], 0.)
        self.assertTrue(source.needs_realign)

    def test_disconnected_socket_invalidates_tracking(self):
        source = self.connect("disconnect")
        deadline = time.monotonic() + 2.
        while source.get_tracking_diagnostics()["error"] is None and time.monotonic() < deadline:
            time.sleep(.02)
        self.assertFalse(source.get_hand_motion_snapshot()["motion_data_ready"])
        self.assertTrue(source.consume_realign_required())
        self.assertIn("connection ended", source.get_tracking_diagnostics()["error"])

    def test_real_stream_reconnects_without_clearing_realignment(self):
        source = self.connect("reconnect")
        initial_stream = source.get_tracking_diagnostics()["stream_id"]
        deadline = time.monotonic() + 4.
        while source.get_tracking_diagnostics()["stream_id"] == initial_stream and time.monotonic() < deadline:
            time.sleep(.02)
        self.assertGreater(source.get_tracking_diagnostics()["stream_id"], initial_stream)
        source.get_hand_motion_snapshot()
        self.assertTrue(source.get_hand_motion_snapshot()["motion_data_ready"])
        self.assertIsNone(source.get_tracking_diagnostics()["error"])
        self.assertTrue(source.needs_realign)
        self.assertTrue(source.consume_realign_required())
        self.assertFalse(source.needs_realign)


if __name__ == "__main__":
    unittest.main()
