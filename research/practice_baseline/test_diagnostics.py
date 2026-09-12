from __future__ import annotations

import unittest

from baseline import OUTPUT, ROOT, audit_log, load, verify_hashes
from diagnostics import RecordedPort


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.actions = [{'operation': 'measure', 'channel': 7, 'position': [1.0, 2.0],
                         'response': {'accepted': True, 'measure_result': 'direction', 'svd_deg': 80.25}}]

    def test_recorded_observation_only(self):
        port = RecordedPort(self.actions)
        self.assertEqual(port.measure((1.0, 2.0), 7), {'result': 'direction', 'bearing_deg': 80.25})
        self.assertEqual(port.index, 1)

    def test_changed_action_rejected(self):
        for operation, position, channel in (('measure', (1.0, 2.1), 7),
                                             ('measure', (1.0, 2.0), 8), ('clear', (1.0, 2.0), 7)):
            port = RecordedPort(self.actions)
            with self.assertRaisesRegex(ValueError, 'diverged'):
                port.respond(operation, position, channel)

    def test_unrecorded_action_rejected(self):
        port = RecordedPort(self.actions)
        port.measure((1.0, 2.0), 7)
        with self.assertRaisesRegex(ValueError, 'unrecorded'):
            port.measure((1.0, 2.0), 7)

    def test_audit_cannot_read_outside_own_practice_logs(self):
        with self.assertRaisesRegex(ValueError, 'practice request logs'):
            audit_log(ROOT / 'reports')

    def test_all_registered_runs_match_independent_audit(self):
        for record in load(OUTPUT / 'runs.json'):
            audit = audit_log(ROOT / 'logs' / 'practice' / record['run_id'])
            self.assertEqual(audit['cleared'], record['source_count'])
            self.assertAlmostEqual(audit['virtual_s'], record['virtual_s'])
            self.assertEqual(audit['log_sha256'], record['log_sha256'])

    def test_frozen_algorithm_and_interface_hashes_unchanged(self):
        hashes = verify_hashes()
        self.assertEqual(hashes['frozen_hashes'], load(OUTPUT / 'protocol.json')['frozen_hashes'])


if __name__ == '__main__':
    unittest.main()
