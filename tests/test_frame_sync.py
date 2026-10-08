import unittest

from teleop.utils.frame_sync import CameraSynchronizer, StreamHistory


MS = 1_000_000


class Frame:
    """Minimal stand-in for a teleimager TeleImage."""

    def __init__(self, sequence, received_monotonic_ns, tag=None, timing=None):
        self.sequence = sequence
        self.received_monotonic_ns = received_monotonic_ns
        self.tag = tag
        self.timing = timing


class StreamHistoryTest(unittest.TestCase):
    def test_repeats_are_ignored_and_capacity_is_bounded(self):
        history = StreamHistory("head", capacity=3)
        for sequence in range(1, 6):
            self.assertTrue(history.offer(sequence, sequence * MS, sequence))
        self.assertEqual(len(history), 3)
        self.assertEqual([frame.sequence for frame in history._frames], [3, 4, 5])
        # A sequence already seen must not be stored twice even after eviction.
        self.assertFalse(history.offer(5, 5 * MS, 5))

    def test_unusable_packets_are_refused(self):
        history = StreamHistory("head")
        self.assertFalse(history.offer(0, MS, "payload"))
        self.assertFalse(history.offer(1, 0, "payload"))
        self.assertFalse(history.offer(1, MS, None))
        self.assertFalse(history.offer(None, MS, "payload"))
        self.assertFalse(history.offer("x", MS, "payload"))
        self.assertEqual(len(history), 0)

    def test_nearest_picks_the_closest_arrival(self):
        history = StreamHistory("left_wrist")
        for sequence, offset in enumerate((0, 40, 80, 120), start=1):
            history.offer(sequence, offset * MS, offset)
        self.assertEqual(history.nearest(50 * MS).payload, 40)
        self.assertEqual(history.nearest(59 * MS).payload, 40)
        self.assertEqual(history.nearest(61 * MS).payload, 80)
        self.assertEqual(history.nearest(1000 * MS).payload, 120)

    def test_nearest_never_rewinds_past_the_floor(self):
        history = StreamHistory("left_wrist")
        for sequence, offset in enumerate((0, 40, 80), start=1):
            history.offer(sequence, offset * MS, offset)
        # For the floor of 2 the frame at 0 ms is not eligible even though it is
        # the closest to the target.
        self.assertEqual(history.nearest(0, min_sequence=2).payload, 40)

    def test_latest_respects_the_floor(self):
        history = StreamHistory("head")
        history.offer(4, MS, "a")
        self.assertIsNone(history.latest(min_sequence=5))
        self.assertEqual(history.latest(min_sequence=4).payload, "a")

    def test_clear_empties_the_history(self):
        history = StreamHistory("head")
        history.offer(1, MS, "a")
        history.clear()
        self.assertEqual(len(history), 0)
        self.assertIsNone(history.latest())


class CameraSynchronizerTest(unittest.TestCase):
    def setUp(self):
        self.sync = CameraSynchronizer(["head", "left_wrist", "right_wrist"], capacity=8)

    def test_pair_anchors_on_the_head_and_reports_offsets(self):
        self.sync.offer("head", Frame(1, 100 * MS))
        self.sync.offer("left_wrist", Frame(1, 88 * MS))
        self.sync.offer("right_wrist", Frame(1, 112 * MS))
        pairing = self.sync.pair(120 * MS)
        self.assertEqual(pairing.anchor, "head")
        self.assertEqual(pairing.anchor_ns, 100 * MS)
        self.assertAlmostEqual(pairing.offsets_ms["head"], 0.0)
        self.assertAlmostEqual(pairing.offsets_ms["left_wrist"], -12.0)
        self.assertAlmostEqual(pairing.offsets_ms["right_wrist"], 12.0)
        self.assertAlmostEqual(pairing.skew_ms, 24.0)

    def test_pair_is_none_only_when_no_stream_has_anything(self):
        self.assertIsNone(self.sync.pair(100 * MS))
        # One stream alone still yields a pairing; the others are simply absent,
        # which the capture reports as a missing camera rather than a fake frame.
        self.sync.offer("left_wrist", Frame(1, 100 * MS))
        pairing = self.sync.pair(100 * MS)
        self.assertEqual(pairing.anchor, "left_wrist")
        self.assertIsNone(pairing.frames["head"])
        self.assertEqual(pairing.skew_ms, 0.0)
        self.assertFalse(self.sync.alignment(pairing)["aligned"])

    def test_no_qualifying_pair_records_the_latest_complete_set(self):
        self.sync.offer("head", Frame(1, 200 * MS))
        self.sync.offer("left_wrist", Frame(1, 260 * MS))
        self.sync.offer("right_wrist", Frame(1, 220 * MS))
        pairing = self.sync.pair(260 * MS)
        self.assertEqual(pairing.anchor, "head")
        self.assertAlmostEqual(pairing.skew_ms, 60.0)
        self.assertAlmostEqual(pairing.offsets_ms["head"], 0.0)
        self.assertAlmostEqual(pairing.offsets_ms["right_wrist"], 20.0)
        self.assertTrue(self.sync.alignment(pairing)["complete"])
        self.assertFalse(self.sync.alignment(pairing)["aligned"])

    def test_pair_chooses_a_qualifying_complete_set_from_history(self):
        self.sync.offer("head", Frame(1, 100 * MS))
        self.sync.offer("right_wrist", Frame(1, 100 * MS))
        for sequence, offset in enumerate((60, 90, 130), start=1):
            self.sync.offer("left_wrist", Frame(sequence, offset * MS))
        # The newest wrist frame is 30 ms late; the one at 90 ms is 10 ms early.
        pairing = self.sync.pair(200 * MS)
        self.assertEqual(pairing.frames["left_wrist"].received_monotonic_ns, 90 * MS)
        self.assertAlmostEqual(pairing.skew_ms, 10.0)

    def test_qualifying_pairing_beats_latest_pairing_at_different_rates(self):
        """Reproduce the shape of the live rig: 15 Hz head against ~23 Hz palms."""
        # Shifted off zero: a zero receive time is refused as unusable.
        epoch = 1000
        head_times = [epoch + offset for offset in (0, 66, 133, 200, 266, 333)]
        wrist_times = [epoch + offset for offset in
                       (10, 43, 77, 110, 143, 176, 210, 243, 276, 310, 343)]
        latest_skews, nearest_skews = [], []
        for grid in range(epoch, epoch + 334, 25):  # a 40 Hz control loop
            heads = [time_ms for time_ms in head_times if time_ms <= grid]
            wrists = [time_ms for time_ms in wrist_times if time_ms <= grid]
            if not heads or not wrists:
                continue
            anchor = heads[-1]
            latest_skews.append(abs(wrists[-1] - anchor))
            sync = CameraSynchronizer(["head", "left_wrist"], capacity=8)
            for sequence, time_ms in enumerate(heads, start=1):
                sync.offer("head", Frame(sequence, time_ms * MS))
            for sequence, time_ms in enumerate(wrists, start=1):
                sync.offer("left_wrist", Frame(sequence, time_ms * MS))
            pairing = sync.pair(grid * MS)
            chosen = pairing.frames["left_wrist"].received_monotonic_ns / MS
            self.assertIn(chosen, wrists)
            nearest_skews.append(pairing.skew_ms)
        self.assertGreaterEqual(max(latest_skews), 40.0)
        self.assertLessEqual(max(nearest_skews), 24.0)
        self.assertLess(max(nearest_skews), max(latest_skews))

    def test_last_used_sequence_stops_a_stream_from_being_rewound(self):
        self.sync.offer("head", Frame(1, 100 * MS))
        self.sync.offer("left_wrist", Frame(7, 90 * MS))
        self.sync.pair(100 * MS)
        self.assertEqual(self.sync.last_used["left_wrist"], 7)
        # A later sample whose anchor sits between frame 6 and 7 must not fall
        # back to frame 6 even though it is nearer.
        self.sync.offer("head", Frame(2, 140 * MS))
        self.sync.offer("left_wrist", Frame(6, 130 * MS))
        pairing = self.sync.pair(140 * MS)
        self.assertEqual(pairing.frames["left_wrist"].sequence, 7)

    def test_alignment_block_flags_samples_over_the_tolerance(self):
        sync = CameraSynchronizer(["head", "left_wrist"], tolerance_ms=25.0)
        sync.offer("head", Frame(1, 100 * MS))
        sync.offer("left_wrist", Frame(1, 90 * MS))
        block = sync.alignment(sync.pair(100 * MS))
        self.assertEqual(block["anchor"], "head")
        self.assertEqual(block["anchor_monotonic_ns"], 100 * MS)
        self.assertEqual(block["tolerance_ms"], 25.0)
        self.assertTrue(block["aligned"])
        self.assertAlmostEqual(block["skew_ms"], 10.0)

        sync.clear()
        sync.offer("head", Frame(2, 200 * MS))
        sync.offer("left_wrist", Frame(2, 165 * MS))
        block = sync.alignment(sync.pair(200 * MS))
        self.assertAlmostEqual(block["skew_ms"], 35.0)
        self.assertFalse(block["aligned"])

    def test_alignment_skew_matches_the_offset_spread(self):
        self.sync.offer("head", Frame(1, 100 * MS))
        self.sync.offer("left_wrist", Frame(1, 80 * MS))
        self.sync.offer("right_wrist", Frame(1, 118 * MS))
        block = self.sync.alignment(self.sync.pair(120 * MS))
        self.assertAlmostEqual(block["skew_ms"],
                               max(block["offset_ms"].values()) - min(block["offset_ms"].values()))
        self.assertFalse(block["aligned"])

    def test_new_qualifying_wrist_frames_beat_an_old_perfect_pair(self):
        self.sync.offer("head", Frame(1, 100 * MS))
        for name in ("left_wrist", "right_wrist"):
            self.sync.offer(name, Frame(1, 100 * MS))
        self.sync.offer("left_wrist", Frame(2, 124 * MS))
        self.sync.offer("right_wrist", Frame(2, 122 * MS))
        pairing = self.sync.pair(130 * MS)
        self.assertEqual({name: frame.sequence for name, frame in pairing.frames.items()},
                         {"head": 1, "left_wrist": 2, "right_wrist": 2})
        self.assertEqual(pairing.skew_ms, 24.0)
        self.assertTrue(self.sync.alignment(pairing)["aligned"])

    def test_freshness_is_the_oldest_camera_time_not_the_tightest_spread(self):
        for name in self.sync.histories:
            self.sync.offer(name, Frame(1, 100 * MS))
        for name, stamp in (("head", 180), ("left_wrist", 190), ("right_wrist", 202)):
            self.sync.offer(name, Frame(2, stamp * MS))
        pairing = self.sync.pair(210 * MS)
        self.assertTrue(all(frame.sequence == 2 for frame in pairing.frames.values()))
        self.assertEqual(pairing.skew_ms, 22.0)

    def test_opposite_offsets_cannot_mask_a_spread_over_tolerance(self):
        for name, stamp in (("head", 100), ("left_wrist", 82), ("right_wrist", 118)):
            self.sync.offer(name, Frame(1, stamp * MS))
        block = self.sync.alignment(self.sync.pair(120 * MS))
        self.assertTrue(all(abs(value) < 25 for value in block["offset_ms"].values()))
        self.assertEqual(block["skew_ms"], 36.0)
        self.assertFalse(block["aligned"])

    def test_short_head_dropout_reuses_an_aligned_set_then_accepts_new_frames(self):
        for name in self.sync.histories:
            self.sync.offer(name, Frame(1, 100 * MS))
        self.sync.pair(100 * MS)
        for name in ("left_wrist", "right_wrist"):
            self.sync.offer(name, Frame(2, 130 * MS))
        repeated = self.sync.pair(135 * MS)
        self.assertTrue(all(frame.sequence == 1 for frame in repeated.frames.values()))
        self.sync.offer("head", Frame(2, 133 * MS))
        resumed = self.sync.pair(140 * MS)
        self.assertTrue(all(frame.sequence == 2 for frame in resumed.frames.values()))
        self.assertEqual(resumed.skew_ms, 3.0)

    def test_10hz_head_and_30hz_wrists_keep_the_latest_qualifying_history(self):
        streams = {"head": list(range(1000, 2001, 100)),
                   "left_wrist": list(range(1008, 2001, 33)),
                   "right_wrist": list(range(1016, 2001, 33))}
        offered = {name: 0 for name in streams}
        selected_head_sequences = []
        for tick in range(1025, 2001, 25):
            for name, stamps in streams.items():
                while offered[name] < len(stamps) and stamps[offered[name]] <= tick:
                    self.sync.offer(name, Frame(offered[name] + 1, stamps[offered[name]] * MS))
                    offered[name] += 1
            pairing = self.sync.pair(tick * MS)
            self.assertTrue(self.sync.alignment(pairing)["aligned"])
            oldest = min(frame.received_monotonic_ns for frame in pairing.frames.values())
            self.assertLessEqual(tick * MS - oldest, 125 * MS)
            selected_head_sequences.append(pairing.frames["head"].sequence)
        self.assertEqual(selected_head_sequences, sorted(selected_head_sequences))
        self.assertGreaterEqual(selected_head_sequences[-1], 10)
        self.assertTrue(all(len(history) <= 8 for history in self.sync.histories.values()))

    def test_unaligned_fallback_advances_sequence_floors(self):
        for name, stamp in (("head", 100), ("left_wrist", 70), ("right_wrist", 60)):
            self.sync.offer(name, Frame(1, stamp * MS))
        self.sync.pair(100 * MS)
        for name, stamp in (("head", 200), ("left_wrist", 170), ("right_wrist", 160)):
            self.sync.offer(name, Frame(2, stamp * MS))
        fallback = self.sync.pair(210 * MS)
        self.assertFalse(self.sync.alignment(fallback)["aligned"])
        self.assertEqual(self.sync.last_used, {name: 2 for name in self.sync.histories})
        self.sync.offer("head", Frame(3, 250 * MS))
        self.sync.offer("left_wrist", Frame(1, 250 * MS))
        self.sync.offer("right_wrist", Frame(1, 250 * MS))
        pairing = self.sync.pair(260 * MS)
        self.assertEqual(pairing.frames["left_wrist"].sequence, 2)
        self.assertEqual(pairing.frames["right_wrist"].sequence, 2)

    def test_source_time_is_checked_per_frame_and_not_replaced_by_receive_time(self):
        for name in self.sync.histories:
            self.sync.offer(name, Frame(1, 190 * MS,
                                       timing={"clock_valid": False, "mapped_monotonic_ns": 200 * MS}))
            self.sync.offer(name, Frame(2, 201 * MS,
                                       timing={"clock_valid": True, "mapped_monotonic_ns": 180 * MS}))
        pairing = self.sync.pair(210 * MS)
        self.assertEqual(pairing.timestamp_basis, "mapped_source")
        self.assertEqual(pairing.anchor_ns, 180 * MS)
        self.assertTrue(all(frame.sequence == 2 for frame in pairing.frames.values()))

    def test_future_mapped_frames_do_not_beat_valid_history(self):
        for name in self.sync.histories:
            self.sync.offer(name, Frame(1, 180 * MS,
                                       timing={"clock_valid": True, "mapped_monotonic_ns": 179 * MS}))
            self.sync.offer(name, Frame(2, 200 * MS,
                                       timing={"clock_valid": True, "mapped_monotonic_ns": 220 * MS}))
        pairing = self.sync.pair(210 * MS)
        self.assertTrue(all(frame.sequence == 1 for frame in pairing.frames.values()))
        self.assertEqual(pairing.anchor_ns, 179 * MS)

    def test_missing_clock_keeps_original_receive_times_and_invalid_timing(self):
        for name, stamp in (("head", 100), ("left_wrist", 108), ("right_wrist", 116)):
            self.sync.offer(name, Frame(1, stamp * MS, timing={"clock_valid": name != "head"}))
        pairing = self.sync.pair(120 * MS)
        self.assertEqual(pairing.timestamp_basis, "host_receive")
        self.assertEqual(pairing.anchor_ns, 100 * MS)
        self.assertFalse(pairing.frames["head"].payload.timing["clock_valid"])

    def test_camera_epoch_change_clears_only_its_history_and_floor(self):
        for name in self.sync.histories:
            self.sync.offer(name, Frame(7, 100 * MS,
                                       timing={"clock_valid": True, "mapped_monotonic_ns": 100 * MS,
                                               "clock_id": "pc2", "source_epoch": "capture-1"}))
        self.sync.pair(100 * MS)
        self.sync.offer("left_wrist", Frame(1, 110 * MS,
                                            timing={"clock_valid": True, "mapped_monotonic_ns": 110 * MS,
                                                    "clock_id": "pc2", "source_epoch": "capture-2"}))
        self.assertEqual(len(self.sync.histories["left_wrist"]), 1)
        self.assertNotIn("left_wrist", self.sync.last_used)
        self.assertEqual(self.sync.last_used["head"], 7)
        pairing = self.sync.pair(120 * MS)
        self.assertEqual(pairing.frames["left_wrist"].sequence, 1)
        self.assertEqual(pairing.frames["head"].sequence, 7)

    def test_clock_id_change_cannot_use_previous_clock_history(self):
        self.sync.offer("head", Frame(9, 100 * MS,
                                     timing={"clock_valid": True, "mapped_monotonic_ns": 100 * MS,
                                             "clock_id": "old-pc2"}))
        self.sync.offer("head", Frame(1, 200 * MS,
                                     timing={"clock_valid": True, "mapped_monotonic_ns": 200 * MS,
                                             "clock_id": "new-pc2"}))
        for name in ("left_wrist", "right_wrist"):
            self.sync.offer(name, Frame(1, 100 * MS,
                                       timing={"clock_valid": True, "mapped_monotonic_ns": 100 * MS,
                                               "clock_id": "pc2"}))
        pairing = self.sync.pair(210 * MS)
        self.assertEqual(pairing.frames["head"].sequence, 1)
        self.assertFalse(self.sync.alignment(pairing)["aligned"])

    def test_single_stream_pairing_has_zero_skew(self):
        sync = CameraSynchronizer(["head"], tolerance_ms=25.0)
        sync.offer("head", Frame(1, 100 * MS))
        block = sync.alignment(sync.pair(100 * MS))
        self.assertEqual(block["skew_ms"], 0.0)
        self.assertTrue(block["aligned"])

    def test_unknown_anchor_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "anchor"):
            CameraSynchronizer(["head"], anchor="left_wrist")

    def test_offering_an_unknown_stream_is_harmless(self):
        self.assertFalse(self.sync.offer("top", Frame(1, MS)))

    def test_clear_resets_the_pairing_state(self):
        self.sync.offer("head", Frame(1, 100 * MS))
        self.sync.offer("left_wrist", Frame(4, 90 * MS))
        self.sync.pair(100 * MS)
        self.sync.clear()
        self.assertEqual(self.sync.last_used, {})
        self.assertIsNone(self.sync.pair(100 * MS))


if __name__ == "__main__":
    unittest.main()
