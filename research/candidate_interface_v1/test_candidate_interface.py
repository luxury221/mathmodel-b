from __future__ import annotations

import ast
import contextlib
import importlib
import io
import os
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import Mock, patch

from candidate_setup import (
    ROOT,
    Fault,
    MockArena,
    OfflineRuleWorld,
    candidate_policy,
    candidate_session,
    client_module,
    original_policy,
    storage,
    validation,
)


class CandidateInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        root = Path(os.environ.get('B2026_CANDIDATE_VALIDATION_OUTPUT', validation.VALIDATION_ROOT / 'unit_only'))
        self.output = root / 'unit_logs' / f'{self._testMethodName}_{uuid.uuid4().hex[:8]}.jsonl'

    def fixture(self, **options):
        config = options.pop('config', client_module.ClientConfig(request_timeout_s=0.3, retry_backoff_s=0.001))
        arena = self.stack.enter_context(MockArena(OfflineRuleWorld([], 33110000), **options))
        journal = self.stack.enter_context(storage.Journal(self.output))
        client = client_module.RobotClient(arena.base_url, arena.robot_id, journal, mock_token=arena.token, config=config)
        return client, arena

    def test_q3_frozen_combined_metadata(self):
        policy, metadata = candidate_policy.make_policy(None, 'q3')
        self.assertTrue(policy.joint_enabled and policy.prune_enabled)
        self.assertEqual(metadata['variant'], 'E_joint+v3_combined')
        self.assertEqual(metadata['network'], 'grid7')
        self.assertFalse(metadata['ground_truth_access'])

    def test_q4_rejected_before_any_request(self):
        client, arena = self.fixture()
        summary = candidate_session.run_session(client, 'q4')
        self.assertEqual(summary['status'], 'needs_attention')
        self.assertFalse(client.entered)
        self.assertEqual(arena.requests, [])

    def test_session_control_flow_identical_to_frozen_session(self):
        functions = []
        for package in ('b2026_robot', 'b2026_candidate'):
            tree = ast.parse((ROOT / 'src' / package / 'session.py').read_text(encoding='utf-8'))
            function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run_session')
            functions.append(ast.dump(function, include_attributes=False))
        self.assertEqual(*functions)

    def test_authoritative_clock_implementation_is_reused(self):
        original, _metadata = original_policy.make_policy(None, 'q3')
        candidate, _metadata = candidate_policy.make_policy(None, 'q3')
        self.assertIs(candidate.account.__func__.__code__, original.account.__func__.__code__)
        self.assertEqual(candidate.account.__func__.__module__, 'b2026_robot.policy')

    def test_frozen_station_order_is_unchanged(self):
        original, _metadata = original_policy.make_policy(None, 'q3')
        candidate, _metadata = candidate_policy.make_policy(None, 'q3')
        self.assertEqual(candidate.route.tolist(), original.route.tolist())

    def test_changed_frozen_candidate_is_rejected(self):
        with patch.object(validation, 'file_hash', return_value='changed'), self.assertRaises(client_module.RobotError):
            candidate_policy.make_policy(None, 'q3')

    def test_stale_validation_blocks_practice(self):
        verification = {'passed': True, 'candidate_mode': 'combined', 'problem': 'q3', 'official_calls': 0}
        with patch.object(Path, 'read_text', return_value=str(validation.VALIDATION_ROOT / 'test_evidence')), \
                patch.object(validation, 'load', side_effect=[verification, {'source_hashes': {'old': 'hash'}}]), \
                patch.object(validation, 'source_hashes', return_value={'new': 'hash'}), \
                self.assertRaisesRegex(client_module.RobotError, 'changed since'):
            validation.require_validation()

    def test_cli_requires_visible_practice_confirmation(self):
        runner = importlib.import_module('run_candidate_practice')
        with patch('sys.argv', ['run_candidate_practice', '--robot-id', 'local-test-team']), \
                patch.object(runner, 'RobotClient') as client_factory, contextlib.redirect_stderr(io.StringIO()), \
                self.assertRaises(SystemExit):
            runner.main()
        client_factory.assert_not_called()

    def test_cli_requires_successful_local_validation(self):
        runner = importlib.import_module('run_candidate_practice')
        with patch('sys.argv', ['run_candidate_practice', '--robot-id', 'local-test-team', '--confirm-practice-ready']), \
                patch.object(runner, 'require_validation', side_effect=client_module.RobotError('not validated')), \
                patch.object(runner, 'RobotClient') as client_factory, contextlib.redirect_stderr(io.StringIO()), \
                self.assertRaises(SystemExit):
            runner.main()
        client_factory.assert_not_called()

    def test_short_budget_exits_without_physical_action(self):
        client, arena = self.fixture(remaining_s=2)
        summary = candidate_session.run_session(client, 'q3')
        self.assertEqual(summary['status'], 'needs_attention')
        self.assertTrue(client.exited)
        self.assertEqual([entry['path'] for entry in arena.requests], ['/enter', '/exit'])
        self.assertEqual(arena.world.trace, [])

    def test_zero_budget_never_queries_exit(self):
        client, arena = self.fixture(remaining_s=0)
        summary = candidate_session.run_session(client, 'q3')
        self.assertEqual(summary['status'], 'needs_attention')
        self.assertEqual([entry['path'] for entry in arena.requests], ['/enter'])

    def test_retry_exhaustion_blocks_further_actions(self):
        client, arena = self.fixture(faults=[Fault('/measure', 'drop_before'), Fault('/measure', 'drop_before')],
                                     config=client_module.ClientConfig(max_attempts=2, retry_backoff_s=0))
        summary = candidate_session.run_session(client, 'q3')
        self.assertEqual(summary['status'], 'needs_attention')
        self.assertTrue(client.uncertain)
        self.assertEqual([entry['path'] for entry in arena.requests], ['/enter', '/measure', '/measure'])
        self.assertEqual(arena.requests[1], arena.requests[2])
        self.assertEqual(len(arena.executions), 1)

    def test_watchdog_can_exit_during_candidate_computation(self):
        client, arena = self.fixture(remaining_s=1, config=client_module.ClientConfig(exit_margin_s=0.6))
        policy, metadata = candidate_policy.make_policy(client, 'q3')

        def slow_run():
            deadline = time.monotonic() + 2
            while not client.exited and time.monotonic() < deadline:
                time.sleep(0.02)
            raise client_module.BudgetStop('candidate observes watchdog exit')

        with patch.object(policy, 'run', side_effect=slow_run), \
                patch.object(candidate_session, 'make_policy', return_value=(policy, metadata)):
            summary = candidate_session.run_session(client, 'q3')
        self.assertEqual(summary['status'], 'needs_attention')
        self.assertTrue(client.exited)
        self.assertEqual([entry['path'] for entry in arena.requests], ['/enter', '/exit'])

    def test_computation_exception_preserves_safe_exit(self):
        client, arena = self.fixture()
        policy = Mock()
        policy.run.side_effect = ValueError('injected candidate geometry error')
        with patch.object(candidate_session, 'make_policy', return_value=(policy, {})):
            summary = candidate_session.run_session(client, 'q3')
        self.assertEqual(summary['status'], 'needs_attention')
        self.assertEqual([entry['path'] for entry in arena.requests], ['/enter', '/exit'])

    def test_accounting_mismatch_is_not_silently_accepted(self):
        client, arena = self.fixture(faults=[Fault('/measure', 'time_shift', 1)])
        summary = candidate_session.run_session(client, 'q3')
        self.assertEqual(summary['status'], 'needs_attention')
        self.assertEqual(client.stop_reason, 'accounting_mismatch')
        self.assertEqual(len(arena.world.trace), 1)


if __name__ == '__main__':
    unittest.main()
