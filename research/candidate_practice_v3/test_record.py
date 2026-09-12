from __future__ import annotations

import unittest

from record import ROOT, evidence, successful


class PracticeRecordTests(unittest.TestCase):
    def setUp(self):
        self.audit = {'client_status': 'completed', 'exited': True, 'error': None, 'cleared': 10}

    def test_matching_public_count_and_normal_exit(self):
        self.assertTrue(successful(self.audit, 10))

    def test_normal_exit_with_missing_source_is_not_success(self):
        self.assertFalse(successful(self.audit, 11))

    def test_complete_clearance_without_confirmed_exit_is_not_success(self):
        self.assertFalse(successful({**self.audit, 'exited': False}, 10))

    def test_error_is_not_success(self):
        self.assertFalse(successful({**self.audit, 'error': 'uncertain'}, 10))

    def test_evidence_scope_is_limited_to_outputs(self):
        with self.assertRaises(ValueError):
            evidence(ROOT / 'B题.pdf')


if __name__ == '__main__':
    unittest.main()
