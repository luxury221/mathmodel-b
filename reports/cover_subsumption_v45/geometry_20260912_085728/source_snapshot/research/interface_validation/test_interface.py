from __future__ import annotations

import contextlib
import http.client
import importlib
import io
import json
import math
import os
import threading
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from bootstrap import (
    ROOT,
    OfflineRuleWorld,
    Source,
    client_module,
    policy_module,
    session_module,
    storage,
)
from mock_server import Fault, MockArena


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class InterfaceTests(unittest.TestCase):
    def setUp(self):
        root = Path(os.environ.get('B2026_INTERFACE_RUN_ROOT', ROOT / 'reports' / 'interface_validation_v1'))
        self.output = storage.d_path(root / 'unit_logs' / f'{self._testMethodName}_{uuid.uuid4().hex[:8]}')
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)

    def fixture(self, sources=(), faults=(), config=None, clock=time.monotonic, **arena_options):
        world = OfflineRuleWorld(sources, 21911001)
        arena = self.stack.enter_context(MockArena(world, faults=faults, clock=clock, **arena_options))
        journal = self.stack.enter_context(storage.Journal(self.output / f'{uuid.uuid4().hex}.jsonl'))
        client = client_module.RobotClient(
            arena.base_url, arena.robot_id, journal, mock_token=arena.token, clock=clock,
            config=config or client_module.ClientConfig(request_timeout_s=0.3, retry_backoff_s=0.001),
        )
        return client, arena

    def raw_post(self, arena, path, payload, **headers):
        raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode('utf-8')
        connection = http.client.HTTPConnection('127.0.0.1', arena.server.server_port, timeout=2)
        try:
            connection.request('POST', path, raw, {'Content-Type': 'application/json', **headers})
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def base(self, arena, request_id='raw-1', **fields):
        return {'arena_id': 'default', 'robot_id': arena.robot_id, 'request_id': request_id, **fields}

    def state(self, client):
        return client.position, client.receiver_channel, client.virtual_seconds, client.accepted_actions

    def test_official_199_second_example(self):
        client, arena = self.fixture()
        client.enter()
        self.assertEqual(client.measure((300, 400), 1)['virtual_time_s'], 105)
        self.assertEqual(client.measure((300, 400), 2)['virtual_time_s'], 111)
        self.assertEqual(client.clear((300, 0), 3)['virtual_time_s'], 194)
        self.assertEqual(client.receiver_channel, 2)
        self.assertEqual(client.measure((300, 0), 2)['virtual_time_s'], 199)
        self.assertEqual(client.exit()['virtual_time_s'], 199)
        self.assertEqual(len(arena.executions), 6)

    def test_result_mapping_and_clear_does_not_tune_receiver(self):
        client, arena = self.fixture([Source(3, (0.0, 0.0), 1000.0, None)])
        client.enter()
        adapter = policy_module.ObservationAdapter(client)
        self.assertEqual(adapter.measure((100, 0), 3)['result'], 'direction')
        self.assertEqual(adapter.measure((0, 0), 3), {'result': 'near'})
        self.assertEqual(adapter.measure((0, 0), 2), {'result': 'no_signal'})
        self.assertEqual(adapter.clear((0, 0), 3), {'result': 'success'})
        self.assertEqual(client.receiver_channel, 2)
        self.assertEqual(adapter.clear((0, 0), 3), {'result': 'no_target_in_range'})
        self.assertEqual(arena.world.remaining, set())

    def test_rejection_zero_does_not_rollback_any_state(self):
        client, arena = self.fixture()
        client.enter()
        client.measure((300, 400), 2)
        previous = self.state(client)
        arena.faults.append(Fault('/clear', 'reject'))
        with self.assertRaises(client_module.RequestRejected):
            client.clear((1200, 1200), 8)
        self.assertEqual(self.state(client), previous)
        count = len(arena.requests)
        with self.assertRaises(client_module.RobotError):
            client.exit()
        self.assertEqual(len(arena.requests), count)

    def test_http_errors_are_not_blindly_retried(self):
        for status in (400, 404, 405, 409, 413, 415, 429, 500):
            with self.subTest(status=status):
                client, arena = self.fixture(faults=[Fault('/measure', 'status', status)])
                client.enter()
                with self.assertRaises(client_module.RequestRejected) as caught:
                    client.measure((5, 5), 3)
                self.assertEqual(caught.exception.status, status)
                self.assertEqual(client.virtual_seconds, 0)
                self.assertEqual(client.retry_count, 0)
                self.assertEqual(len(arena.requests), 2)

    def test_retry_after_drop_before_execution(self):
        client, arena = self.fixture(faults=[Fault('/measure', 'drop_before')])
        client.enter()
        client.measure((0, 0), 1)
        self.assertEqual(arena.requests[1], arena.requests[2])
        self.assertEqual(client.virtual_seconds, 5)
        self.assertEqual(len(arena.executions), 2)

    def test_retry_after_lost_success_for_all_endpoints(self):
        for path in ('/enter', '/measure', '/clear', '/exit'):
            with self.subTest(path=path):
                client, arena = self.fixture([Source(1, (0, 0), 1000, None)], faults=[Fault(path, 'drop_after')])
                client.enter()
                client.measure((0, 0), 1)
                self.assertEqual(client.clear((0, 0), 1)['clear_result'], 'success')
                client.exit()
                requests = [item for item in arena.requests if item['path'] == path]
                self.assertEqual(len(requests), 2)
                self.assertEqual(requests[0], requests[1])
                self.assertEqual(len(arena.executions), 4)
                self.assertEqual(client.virtual_seconds, 10)
                self.assertEqual(client.retry_count, 1)

    def test_truncated_body_retries_same_action(self):
        client, arena = self.fixture(faults=[Fault('/measure', 'truncate_after')])
        client.enter()
        client.measure((0, 0), 1)
        self.assertEqual(len(arena.executions), 2)
        self.assertEqual(arena.requests[1], arena.requests[2])
        self.assertEqual(client.virtual_seconds, 5)

    def test_timeout_after_execution_does_not_duplicate_action(self):
        config = client_module.ClientConfig(request_timeout_s=0.08, retry_backoff_s=0.002)
        client, arena = self.fixture(faults=[Fault('/measure', 'delay_after', 0.2)], config=config)
        client.enter()
        client.measure((10, 0), 1)
        self.assertEqual(len(arena.executions), 2)
        self.assertEqual(client.virtual_seconds, 7)
        self.assertEqual(arena.requests[1], arena.requests[2])

    def test_retry_exhaustion_blocks_new_actions_and_exit(self):
        config = client_module.ClientConfig(max_attempts=2, retry_backoff_s=0)
        client, arena = self.fixture(faults=[Fault('/measure', 'drop_before'), Fault('/measure', 'drop_before')], config=config)
        client.enter()
        with self.assertRaises(client_module.TransportFailure):
            client.measure((0, 0), 1)
        self.assertTrue(client.uncertain)
        self.assertIsNotNone(client.pending)
        self.assertFalse(client.can_exit())
        self.assertEqual(client.virtual_seconds, 0)
        count = len(arena.requests)
        with self.assertRaises(client_module.RobotError):
            client.measure((0, 0), 2)
        self.assertEqual(len(arena.requests), count)

    def test_closed_connection_before_send_is_bounded(self):
        client, arena = self.fixture(config=client_module.ClientConfig(max_attempts=2, retry_backoff_s=0))
        client.enter()
        with patch('http.client.HTTPConnection.connect', side_effect=ConnectionRefusedError('local injected refusal')), \
                self.assertRaises(client_module.TransportFailure):
            client.measure((0, 0), 1)
        self.assertFalse(client.uncertain)
        self.assertFalse(client.can_exit())
        self.assertEqual(len(arena.requests), 1)

    def test_lost_exit_with_closed_interface_is_not_reported_as_success(self):
        client, arena = self.fixture(faults=[Fault('/exit', 'drop_after')], close_after_exit=True,
                                     config=client_module.ClientConfig(max_attempts=2, retry_backoff_s=0))
        client.enter()
        with self.assertRaises(client_module.TransportFailure):
            client.exit()
        self.assertTrue(arena.exited)
        self.assertFalse(client.exited)
        self.assertTrue(client.uncertain)

    def test_malformed_success_response_stops_with_uncertain_state(self):
        client, arena = self.fixture(faults=[Fault('/measure', 'malformed_after')])
        client.enter()
        with self.assertRaises(client_module.ProtocolError):
            client.measure((0, 0), 1)
        self.assertEqual(arena.world.virtual_seconds, 5)
        self.assertEqual(client.virtual_seconds, 0)
        self.assertTrue(client.uncertain)
        self.assertEqual(client.retry_count, 0)
        self.assertFalse(client.can_exit())

    def test_microsecond_clock_accepts_rounding_but_detects_wrong_cost(self):
        client, arena = self.fixture()
        client.enter()
        for index in range(12):
            client.measure((index * math.sqrt(2), index / 3), index % 20 + 1)
        self.assertEqual(client.virtual_seconds, arena.world.virtual_seconds)
        arena.faults.append(Fault('/clear', 'time_shift', 1))
        with self.assertRaises(client_module.AccountingMismatch):
            client.clear((0, 0), 5)
        self.assertEqual(client.stop_reason, 'accounting_mismatch')

    def test_short_enter_budget_not_assumed_to_be_1200(self):
        clock = FakeClock()
        client, arena = self.fixture(clock=clock, remaining_s=12, config=client_module.ClientConfig(exit_margin_s=3))
        client.enter()
        self.assertEqual(client.remaining_seconds(), 12)
        clock.advance(9.1)
        with self.assertRaises(client_module.BudgetStop):
            client.measure((0, 0), 1)
        self.assertEqual(len(arena.requests), 1)
        client.exit()
        self.assertEqual(len(arena.requests), 2)

    def test_enter_response_latency_consumes_local_budget(self):
        clock = FakeClock()
        client, _arena = self.fixture(clock=clock, remaining_s=12)
        original = client._read_wire

        def delayed(*args):
            response = original(*args)
            if args[2] == '/enter':
                clock.advance(2)
            return response

        with patch.object(client, '_read_wire', side_effect=delayed):
            client.enter()
        self.assertEqual(client.remaining_seconds(), 10)

    def test_expired_deadline_never_queries_exit(self):
        clock = FakeClock()
        client, arena = self.fixture(clock=clock, remaining_s=12)
        client.enter()
        clock.advance(12)
        self.assertFalse(client.can_exit())
        with self.assertRaises(client_module.BudgetStop):
            client.exit()
        self.assertEqual(len(arena.requests), 1)

    def test_zero_remaining_time_sends_no_more_requests(self):
        client, arena = self.fixture(remaining_s=0)
        client.enter()
        self.assertFalse(client.can_exit())
        with self.assertRaises(client_module.BudgetStop):
            client.measure((0, 0), 1)
        self.assertEqual(len(arena.requests), 1)

    def test_virtual_budget_uses_movement_switch_and_worst_clear_cost(self):
        client, arena = self.fixture(max_virtual_s=10)
        client.enter()
        client.measure((0, 0), 2)
        with self.assertRaises(client_module.BudgetStop):
            client.clear((0, 0), 1)
        client.exit()
        self.assertEqual(client.virtual_seconds, 6)
        self.assertEqual(len(arena.executions), 3)

    def test_concurrent_new_action_is_rejected_locally(self):
        client, arena = self.fixture()
        client.enter()
        entered, release = threading.Event(), threading.Event()
        original = client._read_wire
        failures = []

        def held(*args):
            entered.set()
            if not release.wait(2):
                raise RuntimeError('test release missing')
            return original(*args)

        def first_action():
            try:
                client.measure((0, 0), 1)
            except (RuntimeError, OSError) as error:
                failures.append(error)

        with patch.object(client, '_read_wire', side_effect=held):
            thread = threading.Thread(target=first_action)
            thread.start()
            try:
                self.assertTrue(entered.wait(2))
                with self.assertRaises(client_module.ConcurrentAction):
                    client.clear((0, 0), 1)
            finally:
                release.set()
                thread.join(3)
        self.assertEqual(failures, [])
        self.assertEqual([item['path'] for item in arena.requests], ['/enter', '/measure'])

    def test_watchdog_exits_during_policy_computation(self):
        client, arena = self.fixture(remaining_s=1, config=client_module.ClientConfig(exit_margin_s=0.6))

        class SlowPolicy:
            def run(self):
                deadline = time.monotonic() + 2
                while not client.exited and time.monotonic() < deadline:
                    time.sleep(0.02)
                raise client_module.BudgetStop('policy observes watchdog exit')

        with patch.object(session_module, 'make_policy', return_value=(SlowPolicy(), {})):
            summary = session_module.run_session(client, 'q3')
        self.assertEqual(summary['status'], 'needs_attention')
        self.assertTrue(client.exited)
        self.assertEqual([item['path'] for item in arena.requests], ['/enter', '/exit'])

    def test_computation_failure_attempts_one_safe_exit(self):
        client, arena = self.fixture()
        policy = unittest.mock.Mock()
        policy.run.side_effect = ValueError('injected geometry failure')
        with patch.object(session_module, 'make_policy', return_value=(policy, {})):
            summary = session_module.run_session(client, 'q4')
        self.assertEqual(summary['status'], 'needs_attention')
        self.assertTrue(summary['exited'])
        self.assertEqual([item['path'] for item in arena.requests], ['/enter', '/exit'])

    def test_raw_request_id_conflict_does_not_execute(self):
        client, arena = self.fixture()
        client.enter()
        first = self.base(arena, position={'x': 0, 'y': 0}, channel=1)
        status, original = self.raw_post(arena, '/measure', first)
        self.assertEqual(status, 200)
        self.assertEqual(self.raw_post(arena, '/measure', first)[1], original)
        self.assertEqual(self.raw_post(arena, '/measure', {**first, 'channel': 2})[0], 409)
        self.assertEqual(self.raw_post(arena, '/clear', first)[0], 409)
        self.assertEqual(arena.world.virtual_seconds, 5)

    def test_invalid_or_unknown_fields_do_not_consume_request_id(self):
        _client, arena = self.fixture()
        first = self.base(arena)
        self.assertFalse(self.raw_post(arena, '/enter', {**first, 'misspelled': 1})[1]['accepted'])
        self.assertEqual(self.raw_post(arena, '/enter', {**first, 'robot_id': 12})[0], 400)
        self.assertTrue(self.raw_post(arena, '/enter', first)[1]['accepted'])

    def test_http_format_and_duplicate_keys(self):
        _client, arena = self.fixture()
        payload = self.base(arena)
        self.assertEqual(self.raw_post(arena, '/enter/', payload)[0], 404)
        self.assertEqual(self.raw_post(arena, '/enter?x=1', payload)[0], 404)
        self.assertEqual(self.raw_post(arena, '/enter', payload, **{'Content-Type': 'text/plain'})[0], 415)
        self.assertEqual(self.raw_post(arena, '/enter', payload, **{'Content-Encoding': 'gzip'})[0], 415)
        self.assertEqual(self.raw_post(arena, '/enter', b'{"arena_id":"default","arena_id":"x"}')[0], 400)
        self.assertEqual(self.raw_post(arena, '/enter', b' ' * 65537)[0], 413)
        self.assertEqual(arena.executions, [])

    def test_new_actions_have_unique_ids_even_at_identical_position(self):
        client, arena = self.fixture()
        client.enter()
        client.measure((0, 0), 1)
        client.measure((0, 0), 1)
        identifiers = [json.loads(item['body_utf8'])['request_id'] for item in arena.requests]
        self.assertEqual(len(set(identifiers)), 3)
        self.assertEqual(client.virtual_seconds, 10)

    def test_invalid_coordinates_channels_and_identifiers_are_local(self):
        client, arena = self.fixture()
        client.enter()
        for position, channel in (((math.nan, 0), 1), ((math.inf, 0), 1), ((2000001, 0), 1),
                                  ((True, 0), 1), ((0, 0), True), ((0, 0), 1.5), ((0, 0), 21)):
            with self.subTest(position=position, channel=channel), self.assertRaises((ValueError, TypeError)):
                client.measure(position, channel)
        for name in ('', 'x\x00', 'x\u200b', '中' * 22):
            with self.subTest(identifier=name), self.assertRaises(ValueError):
                client_module.identifier(name, 'robot_id', 64)
        self.assertEqual(len(arena.requests), 1)

    def test_strict_response_validation(self):
        common = {'accepted': True, 'real_timestamp_ms': 123, 'virtual_time_s': 5, 'measure_result': 'direction', 'svd_deg': 1}
        invalid = [dict(common, accepted=1), dict(common, real_timestamp_ms=True), dict(common, virtual_time_s=-1),
                   dict(common, svd_deg=360), dict(common, svd_deg=float('nan')), dict(common, measure_result='near'),
                   dict(common, measure_result='unexpected')]
        for response in invalid:
            with self.subTest(response=response), self.assertRaises(client_module.ProtocolError):
                client_module.validate_response('/measure', 200, json.dumps(response).encode())
        with self.assertRaises(client_module.ProtocolError):
            client_module.validate_response('/measure', 500, json.dumps(common).encode())
        with self.assertRaises(client_module.ProtocolError):
            client_module.validate_response('/measure', 200, b'{"accepted":true,"accepted":false}')

    def test_mock_capability_mismatch_sends_no_actions(self):
        client, arena = self.fixture()
        client.mock_token = 'wrong-token'
        with self.assertRaises(client_module.RobotError):
            client.enter()
        self.assertEqual(arena.requests, [])

    def test_official_opt_in_and_loopback_restrictions(self):
        client, arena = self.fixture()
        with self.assertRaises(ValueError):
            client_module.RobotClient(arena.base_url, arena.robot_id, client.journal)
        for url in ('http://example.com:2026', 'http://192.0.2.1:2026', 'https://127.0.0.1:2026',
                    'http://127.0.0.1:2026/x', 'http://user@127.0.0.1:2026', 'http://127.0.0.1:2026?a=1'):
            with self.subTest(url=url), self.assertRaises(ValueError):
                client_module.RobotClient(url, arena.robot_id, client.journal, allow_official=True)
        self.assertEqual(arena.requests, [])

    def test_cli_cannot_connect_without_explicit_confirmation(self):
        cli = importlib.import_module('run_robot')
        with patch('sys.argv', ['run_robot', '--problem', 'q3', '--robot-id', 'local-test']), \
                patch('http.client.HTTPConnection.connect', side_effect=AssertionError('network must not be touched')), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as result:
            cli.main()
        self.assertEqual(result.exception.code, 2)

    def test_journal_is_ordered_complete_and_on_d_drive(self):
        client, arena = self.fixture()
        client.enter()
        client.measure((0, 0), 1)
        client.exit()
        entries = [json.loads(line) for line in client.journal.path.read_text(encoding='utf-8').splitlines()]
        for request in arena.requests:
            identifier = json.loads(request['body_utf8'])['request_id']
            relevant = [entry for entry in entries if entry.get('request_id', entry.get('payload', {}).get('request_id')) == identifier]
            self.assertEqual([entry['event'] for entry in relevant], ['request', 'attempt', 'response', 'committed'])
            self.assertEqual(relevant[0]['body_utf8'], request['body_utf8'])
        self.assertEqual(client.journal.path.drive.upper(), 'D:')
        with self.assertRaises(ValueError):
            storage.d_path('C:/forbidden-artifact.json')

    def test_journal_failure_does_not_retry_an_executed_action(self):
        client, arena = self.fixture()
        client.enter()
        original = client.journal.write

        def failing(event, **fields):
            if event == 'response':
                raise OSError('injected disk-full failure')
            return original(event, **fields)

        with patch.object(client.journal, 'write', side_effect=failing), self.assertRaises(OSError):
            client.measure((0, 0), 1)
        self.assertEqual(len(arena.executions), 2)
        self.assertEqual(client.retry_count, 0)
        self.assertTrue(client.uncertain)
        self.assertFalse(client.can_exit())

    def test_no_actions_after_successful_exit(self):
        client, arena = self.fixture()
        client.enter()
        client.exit()
        for operation in (client.enter, client.exit, lambda: client.measure((0, 0), 1)):
            with self.assertRaises(client_module.RobotError):
                operation()
        self.assertEqual(len(arena.requests), 2)

    def test_all_frozen_networks_construct_without_reading_ground_truth(self):
        client, arena = self.fixture()
        for problem, network, count in (('q3', 'dual21', 7), ('q4', 'dual21', 21), ('q4', 'grid25', 25)):
            with self.subTest(problem=problem, network=network):
                policy, metadata = policy_module.make_policy(client, problem, network)
                self.assertEqual(policy.stations.shape, (count, 2))
                self.assertEqual(metadata['variant'], 'E_joint' if problem == 'q3' else 'F_route')
                self.assertFalse(metadata['ground_truth_access'])
                self.assertEqual(set(policy.port.__dataclass_fields__), {'measure', 'clear'})
        self.assertEqual(arena.requests, [])

    def test_connection_delay_cannot_send_action_past_reserved_deadline(self):
        clock = FakeClock()
        config = client_module.ClientConfig(exit_margin_s=3, max_attempts=1)
        client, arena = self.fixture(clock=clock, remaining_s=12, config=config)
        client.enter()
        original = http.client.HTTPConnection.connect

        def slow_connect(connection):
            original(connection)
            clock.advance(9.5)

        with patch('http.client.HTTPConnection.connect', new=slow_connect), self.assertRaises(client_module.BudgetStop):
            client.measure((0, 0), 1)
        self.assertEqual(len(arena.requests), 1)
        self.assertFalse(client.uncertain)
        self.assertIsNone(client.pending)
        self.assertTrue(client.can_exit())
        client.exit()
        self.assertEqual(len(arena.requests), 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
