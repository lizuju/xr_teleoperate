import importlib.util
from pathlib import Path
import threading
import time
import unittest
from unittest import mock

import cv2
import numpy as np
import zmq


SOURCE = Path(__file__).resolve().parents[1] / "teleop/teleimager/src/teleimager/image_client.py"
SPEC = importlib.util.spec_from_file_location("collection_image_client_timing", SOURCE)
image_client = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(image_client)


class ImageClientTimingTest(unittest.TestCase):
    def setUp(self):
        self.context = zmq.Context()
        self.publisher = self.context.socket(zmq.XPUB)
        self.publisher.setsockopt(zmq.LINGER, 0)
        self.port = self.publisher.bind_to_random_port("tcp://127.0.0.1")
        self.subscriber = None
        self.releases = []

    def tearDown(self):
        for release in self.releases:
            release.set()
        if self.subscriber is not None:
            self.subscriber.stop()
        self.publisher.close()
        self.context.term()

    def start_subscriber(self, decode=True):
        self.subscriber = image_client.ZMQ_SubscriberThread(
            "127.0.0.1", self.port, self.context, request_bgr=decode
        )
        self.subscriber.start()
        self.assertTrue(self.subscriber._wait_for_start(1.0))
        self.assertTrue(self.publisher.poll(1000, zmq.POLLIN), "subscriber did not subscribe")
        self.assertEqual(self.publisher.recv(), b"\x01")
        return self.subscriber

    def wait_until(self, predicate, message="expected image state was not reached"):
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.002)
        self.fail(message)

    @staticmethod
    def jpeg(value):
        success, encoded = cv2.imencode(".jpg", np.full((12, 16, 3), value, dtype=np.uint8))
        if not success:
            raise RuntimeError("test JPEG encoding failed")
        return encoded.tobytes()

    def block_decode(self, jpg):
        entered, release = threading.Event(), threading.Event()
        self.releases.append(release)
        original = self.subscriber._decode_image

        def decode(packet):
            if packet == jpg:
                entered.set()
                release.wait(3.0)
            return original(packet)

        patcher = mock.patch.object(self.subscriber, "_decode_image", side_effect=decode)
        patcher.start()
        self.addCleanup(patcher.stop)
        return entered, release

    def assert_packet(self, frame, jpg, sequence, received_ns):
        self.assertEqual(frame.jpg, jpg)
        self.assertEqual(frame.sequence, sequence)
        self.assertEqual(frame.received_monotonic_ns, received_ns)
        expected = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        np.testing.assert_array_equal(frame.bgr, expected)

    def test_new_jpeg_during_decode_keeps_returned_packet_coherent(self):
        subscriber = self.start_subscriber()
        first, second = self.jpeg(40), self.jpeg(190)
        self.publisher.send(first)
        self.wait_until(lambda: subscriber.recv().sequence == 1)
        old = subscriber.recv()
        self.assertGreater(old.received_monotonic_ns, 0)
        entered, release = self.block_decode(second)
        self.publisher.send(second)
        self.assertTrue(entered.wait(1.0))
        received_ns, sequence, jpg = subscriber._jpg_3ring_buffer.read()
        self.assertEqual((sequence, jpg), (2, second))
        self.assertGreater(received_ns, old.received_monotonic_ns)
        for _ in range(20):
            self.assert_packet(subscriber.recv(), first, 1, old.received_monotonic_ns)
        release.set()
        self.wait_until(lambda: subscriber.recv().sequence == 2)
        self.assert_packet(subscriber.recv(), second, 2, received_ns)

    def test_first_packet_is_not_exposed_as_decoded_until_ready(self):
        subscriber = self.start_subscriber()
        jpg = self.jpeg(75)
        entered, release = self.block_decode(jpg)
        self.publisher.send(jpg)
        self.assertTrue(entered.wait(1.0))
        before = subscriber.recv()
        self.assertIsNone(before.jpg)
        self.assertIsNone(before.bgr)
        self.assertEqual((before.sequence, before.received_monotonic_ns), (0, 0))
        receive_time = subscriber._jpg_3ring_buffer.read()[0]
        release.set()
        self.wait_until(lambda: subscriber.recv().sequence == 1)
        self.assert_packet(subscriber.recv(), jpg, 1, receive_time)

    def test_no_new_packet_does_not_refresh_receive_time_or_sequence(self):
        subscriber = self.start_subscriber()
        jpg = self.jpeg(90)
        self.publisher.send(jpg)
        self.wait_until(lambda: subscriber.recv().sequence == 1)
        before = subscriber.recv()
        time.sleep(0.15)  # Cross the subscriber's 100ms empty-poll path.
        after = subscriber.recv()
        self.assert_packet(after, jpg, before.sequence, before.received_monotonic_ns)
        self.assertGreater(time.monotonic_ns() - after.received_monotonic_ns, 100_000_000)

    def test_raw_only_packet_has_its_own_receive_metadata(self):
        subscriber = self.start_subscriber(decode=False)
        jpg = self.jpeg(110)
        self.publisher.send(jpg)
        self.wait_until(lambda: subscriber.recv().sequence == 1)
        frame = subscriber.recv()
        self.assertEqual(frame.jpg, jpg)
        self.assertGreater(frame.received_monotonic_ns, 0)
        self.assertIs(frame._bgr, image_client.TeleImage._NOT_SET)
        time.sleep(0.15)
        self.assertEqual(subscriber.recv().received_monotonic_ns, frame.received_monotonic_ns)

    def test_decode_replacement_balances_pending_task_count(self):
        subscriber = self.start_subscriber()
        first, second, third = self.jpeg(20), self.jpeg(100), self.jpeg(220)
        entered, release = self.block_decode(first)
        self.publisher.send(first)
        self.assertTrue(entered.wait(1.0))
        self.publisher.send(second)
        self.wait_until(lambda: subscriber._sequence == 2 and subscriber._bgr_decode_queue.qsize() == 1)
        self.publisher.send(third)

        def third_is_queued():
            with subscriber._bgr_decode_queue.mutex:
                queued = subscriber._bgr_decode_queue.queue
                return len(queued) == 1 and queued[0][1] == 3

        self.wait_until(third_is_queued)
        with subscriber._bgr_decode_queue.mutex:
            queued = subscriber._bgr_decode_queue.queue[0]
            self.assertEqual(queued[1:], (3, third))
            self.assertEqual(subscriber._bgr_decode_queue.unfinished_tasks, 2)
        release.set()
        self.wait_until(lambda: subscriber.recv().sequence == 3)
        self.wait_until(lambda: subscriber._bgr_decode_queue.unfinished_tasks == 0)
        self.assert_packet(subscriber.recv(), third, 3, queued[0])

    def test_consumer_race_during_queue_replacement_does_not_stop_reception(self):
        subscriber = self.start_subscriber()
        first, second, third = self.jpeg(30), self.jpeg(120), self.jpeg(230)
        first_entered, first_release = self.block_decode(first)
        second_entered, second_release = self.block_decode(second)
        self.publisher.send(first)
        self.assertTrue(first_entered.wait(1.0))
        self.publisher.send(second)
        self.wait_until(lambda: subscriber._sequence == 2 and subscriber._bgr_decode_queue.qsize() == 1)
        original_full = subscriber._bgr_decode_queue.full
        raced = threading.Event()

        def full_with_consumer_race():
            result = original_full()
            if result:
                first_release.set()
                if second_entered.wait(1.0):
                    raced.set()
            return result

        with mock.patch.object(subscriber._bgr_decode_queue, "full", side_effect=full_with_consumer_race):
            self.publisher.send(third)
            self.assertTrue(raced.wait(1.0))
            second_release.set()
            self.wait_until(lambda: subscriber.recv().sequence == 3,
                            "receiver lost the packet when decoder consumed the queued image")
        self.assertTrue(subscriber.is_alive())
        self.wait_until(lambda: subscriber._bgr_decode_queue.unfinished_tasks == 0)

    def test_stop_does_not_leave_unfinished_decode_work(self):
        subscriber = self.start_subscriber()
        join = subscriber.join

        def join_after_decoder_exits(timeout=None):
            join(timeout)
            subscriber._decoder_thread.join(timeout=1.0)

        with mock.patch.object(subscriber, "join", side_effect=join_after_decoder_exits):
            subscriber.stop()
        self.assertFalse(subscriber._decoder_thread.is_alive())
        self.assertEqual(subscriber._bgr_decode_queue.unfinished_tasks, 0)
        self.assertTrue(subscriber._bgr_decode_queue.empty())

    def test_stop_discards_pending_decode_packet_without_count_leak(self):
        subscriber = self.start_subscriber()
        first, second = self.jpeg(45), self.jpeg(170)
        entered, release = self.block_decode(first)
        self.publisher.send(first)
        self.assertTrue(entered.wait(1.0))
        self.publisher.send(second)
        self.wait_until(lambda: subscriber._sequence == 2 and subscriber._bgr_decode_queue.qsize() == 1)
        stopper = threading.Thread(target=subscriber.stop, daemon=True)
        stopper.start()
        self.wait_until(lambda: not subscriber._running)
        release.set()
        stopper.join(timeout=2.0)
        self.assertFalse(stopper.is_alive())
        self.assertFalse(subscriber._decoder_thread.is_alive())
        self.assertEqual(subscriber._bgr_decode_queue.unfinished_tasks, 0)
        self.assertTrue(subscriber._bgr_decode_queue.empty())


if __name__ == "__main__":
    unittest.main()
