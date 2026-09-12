from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'research/frozen_practice_v33'))
import adapter
import practice
import numpy as np
from geometry import dual_ring_network, seven_network


def load_trace(directory):
    events = [json.loads(line) for line in (directory / 'requests.jsonl').read_text(encoding='utf-8').splitlines()]
    requests = {event['payload']['request_id']: event for event in events if event['event'] == 'request'}
    responses = {event['request_id']: json.loads(event['body_utf8']) for event in events if event['event'] == 'response'}
    commits = [event for event in events if event['event'] == 'committed']
    trace = []
    position = [0.0, 0.0]
    receiver = 1
    clock = 0.0
    maximum_error = 0.0
    for event in commits:
        path = event['path']
        if path not in ('/enter', '/measure', '/clear', '/exit'):
            raise ValueError('An undocumented endpoint was used')
        if path not in ('/measure', '/clear'):
            continue
        payload = requests[event['request_id']]['payload']
        response = responses[event['request_id']]
        destination = [payload['position']['x'], payload['position']['y']]
        channel = payload['channel']
        clock += math.dist(destination, position) / 5
        if path == '/measure':
            clock += 5 + int(channel != receiver)
            receiver = channel
            observation = {'result': response['measure_result']}
            if observation['result'] == 'direction':
                observation['bearing_deg'] = response['svd_deg']
        else:
            clock += 5 if response['clear_result'] == 'success' else 3
            observation = {'result': response['clear_result']}
        clock = round(clock, 6)
        maximum_error = max(maximum_error, abs(clock - event['virtual_time_s']))
        if not response['accepted'] or receiver != event['receiver_channel'] or destination != event['position'] or maximum_error > 1e-6:
            raise ValueError('Per-action official clock/state audit failed')
        trace.append({'action': path[1:], 'channel': channel, 'position': destination, 'response': observation, 'virtual_seconds': event['virtual_time_s']})
        position = destination
    if len({event['request_id'] for event in commits}) != len(commits):
        raise ValueError('A request committed more than once')
    return trace, maximum_error


class ReplyTape:
    def __init__(self, trace):
        self.trace = trace
        self.index = 0
        self.current = None

    def consume(self, action, point, channel):
        if self.index >= len(self.trace):
            raise ValueError('Replay requested an action beyond the official log')
        expected = self.trace[self.index]
        if action != expected['action'] or channel != expected['channel'] or not np.array_equal(point, expected['position']):
            raise ValueError(f'Official replay action differs at index {self.index}')
        self.index += 1
        self.current = expected
        return dict(expected['response'])

    def measure(self, point, channel):
        return self.consume('measure', point, channel)

    def clear(self, point, channel):
        return self.consume('clear', point, channel)


class DiagnosticMixin:
    def __init__(self, tape, *args, **kwargs):
        self.replay_tape = tape
        self.category = 'unclassified'
        self.cost_categories = defaultdict(lambda: {'actions': 0, 'movement_m': 0.0, 'virtual_seconds': 0.0})
        super().__init__(tape, *args, **kwargs)

    def measure(self, channel, position, active=False, opportunistic=False):
        original = self.category
        self.category = 'active_localization' if active else 'opportunistic_direction' if opportunistic else 'discovery_radio'
        try:
            return super().measure(channel, position, active, opportunistic)
        finally:
            self.category = original

    def clear(self, channel, position):
        original = self.category
        self.category = 'unknown_optical' if self.states[channel].status == 'UNKNOWN' else 'known_source_clear'
        try:
            return super().clear(channel, position)
        finally:
            self.category = original

    def account(self, position, channel, operation, result):
        action = self.replay_tape.current
        if action['action'] != operation or action['channel'] != channel or action['response']['result'] != result:
            raise ValueError('Replay accounting differs')
        cost = self.cost_categories[self.category]
        cost['actions'] += 1
        cost['movement_m'] += float(np.linalg.norm(np.asarray(position) - self.position))
        cost['virtual_seconds'] += action['virtual_seconds'] - self.virtual_seconds
        self.position = np.asarray(position).copy()
        if operation == 'measure':
            self.receiver_channel = channel
        self.virtual_seconds = action['virtual_seconds']


def replay(directory, problem):
    trace, error = load_trace(directory)
    tape = ReplyTape(trace)
    selected = adapter.ScanEconomyPolicy if problem == 'q3' else adapter.ShapedProbePolicy
    policy_class = type('DiagnosticReplay', (DiagnosticMixin, selected), {})
    policy = policy_class(tape, problem, seven_network() if problem == 'q3' else dual_ring_network(),
                          'E_joint' if problem == 'q3' else 'F_route', 'grid7' if problem == 'q3' else 'dual21', mode=adapter.SELECTED[problem])
    result = policy.run()
    if tape.index != len(trace) or result['virtual_seconds'] != trace[-1]['virtual_seconds']:
        raise ValueError('Replay did not consume the complete official trajectory')
    last_clear = max(index for index, action in enumerate(trace) if action['action'] == 'clear' and action['response']['result'] == 'success')
    return {'actions': len(trace), 'max_clock_error': error, 'categories': dict(policy.cost_categories),
            'post_last_clear_seconds_diagnostic_only': trace[-1]['virtual_seconds'] - trace[last_clear]['virtual_seconds'],
            'same_actions': True, 'policy_stats': result['stats']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    batch, protocol, rows = practice.verify_batch(args.batch)
    output = args.output.resolve()
    if output.drive.upper() != 'D:' or output.exists():
        raise ValueError('Use a fresh D-drive audit directory')
    if len(rows) != 10 or any(row['state'] != 'completed' for row in rows) or any(sum(row['problem'] == problem for row in rows) != 5 for problem in adapter.SELECTED):
        raise ValueError('The fixed official practice batch is incomplete')
    for relative, digest in protocol['source_hashes'].items():
        if hashlib.sha256((batch / 'source_snapshot' / relative).read_bytes()).hexdigest() != digest:
            raise ValueError('Frozen batch snapshot differs')
    output.mkdir(parents=True)
    audits = []
    for row in rows:
        directory = Path(row['log_directory'])
        audited = practice.audit_log(directory)
        if json.loads(json.dumps(audited)) != row['audit']:
            raise ValueError('Stored official audit differs from recalculation')
        session = adapter.load(directory / 'summary.json')
        if session['candidate_source_hashes'] != protocol['source_hashes']:
            raise ValueError('An official run used different source files')
        for label in ('before_start_evidence', 'ready_evidence', 'result_evidence'):
            if practice.evidence(row[label]['path']) != row[label]:
                raise ValueError('Public screenshot hash differs')
        reproduced = replay(directory, row['problem'])
        record = {'problem': row['problem'], 'case_code': row['case_code'], 'source_count': row['source_count'],
                  'seconds_per_source': row['seconds_per_source'], 'replay': reproduced}
        audits.append(record)
        practice.save(output / (row['case_code'] + '.json'), record)
        print('REPLAY', row['problem'], row['case_code'], reproduced['actions'], 'exact', flush=True)
    summaries = {}
    for problem, target in (('q3', 180), ('q4', 300)):
        selected = [row for row in rows if row['problem'] == problem]
        mean = statistics.mean(row['seconds_per_source'] for row in selected)
        summaries[problem] = {'runs': len(selected), 'successes': len(selected), 'mean_seconds_per_source': mean,
                               'median': statistics.median(row['seconds_per_source'] for row in selected),
                               'min': min(row['seconds_per_source'] for row in selected), 'max': max(row['seconds_per_source'] for row in selected),
                               'target': target, 'gap': mean - target, 'target_met_in_this_batch': mean < target,
                               'movement': statistics.mean(row['audit']['move_m'] / 5 / row['source_count'] for row in selected),
                               'radio': statistics.mean((5 * row['audit']['measure_count'] + row['audit']['components']['switch_s']) / row['source_count'] for row in selected),
                               'retries': sum(row['audit']['retry_count'] for row in selected)}
    final = {'official_practices': 10, 'formal_tests': 0, 'passed': True, 'summaries': summaries,
             'actions': sum(row['replay']['actions'] for row in audits), 'exact_replays': len(audits),
             'source_files_verified': len(protocol['source_hashes']), 'cases': audits,
             'comparison_warning': 'New random official cases; not paired with previous batches and not proof of a population guarantee',
             'diagnostic_warning': 'Movement is attributed to the next action; category costs are not independently removable savings',
             'auditor_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    practice.save(output / 'final_verification.json', final)
    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
