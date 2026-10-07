from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from check_visionpro_input import summarize


class DiagnosticSummaryTests(unittest.TestCase):
    def test_packet_metrics_are_not_multiplied_by_polling_and_hold_is_not_event_count(self):
        rows = [dict(stream_id=1, source_age_ms=100.+i, hold_reason='input_stale',
                     transport={'packets_received': 1, 'bridge_pending_ms': 200., 'upstream_age_max_ms': 700.})
                for i in range(3)]
        rows.append(dict(stream_id=1, source_age_ms=5., hold_reason='input_stale',
                         transport={'packets_received': 2, 'bridge_pending_ms': 1., 'upstream_age_max_ms': 700.}))
        result = summarize(rows)
        self.assertEqual(result['metrics']['bridge_pending_ms']['samples'], 2)
        self.assertEqual(result['metrics']['bridge_pending_ms']['over_100ms'], 1)
        self.assertEqual(result['metrics']['source_age_ms']['samples'], 4)
        self.assertEqual(result['last_transport_by_stream']['1']['upstream_age_max_ms'], 700.)
        self.assertEqual(result['hold_reason_samples']['input_stale'], 4)

    def test_empty_diagnostic_does_not_claim_zero_latency(self):
        result = summarize([])
        self.assertEqual(result['samples'], 0)
        self.assertEqual(result['metrics']['upstream_age_ms'], {'samples': 0})


if __name__ == '__main__':
    unittest.main()
