from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'research' / 'joint_search_v4'))
from experiment import hashes, save


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def audit_trial(folder, output):
    events = [json.loads(line) for line in (folder / 'requests.jsonl').read_text(encoding='utf-8').splitlines()]
    requests = {event['payload']['request_id']: event for event in events if event['event'] == 'request'}
    responses = {}
    for event in events:
        if event['event'] == 'response':
            try:
                body = json.loads(event['body_utf8'])
            except (ValueError, KeyError):
                continue
            if body.get('accepted'):
                responses[event['request_id']] = body
    committed = [event for event in events if event['event'] == 'committed']
    session = load(folder / 'session.json')
    scene_id = folder.name.rsplit('_', 1)[0]
    mode = 'adaptive' if session['problem'] == 'q3' else 'sweep_previous'
    direct = load(output / 'traces' / (scene_id + '__' + mode + '.json'))
    direct_record = load(output / 'records' / (scene_id + '__' + mode + '.json'))
    position = np.zeros(2)
    receiver = 1
    raw = 0.0
    rounded = 0.0
    action_index = 0
    max_error = 0.0
    for event in committed:
        path = event['path']
        if path in ('/measure', '/clear'):
            request = requests[event['request_id']]['payload']
            response = responses[event['request_id']]
            destination = np.array([request['position']['x'], request['position']['y']])
            action = direct[action_index]
            if action['action'] != path[1:] or action['position'] != destination.tolist() or action['channel'] != request['channel']:
                raise ValueError('Physical action differs from direct evaluation')
            if action['response']['result'] != response[path[1:] + '_result']:
                raise ValueError('Physical response differs')
            if 'bearing_deg' in action['response'] and action['response']['bearing_deg'] != response['svd_deg']:
                raise ValueError('Bearing differs')
            movement = float(np.linalg.norm(destination - position)) / 5
            raw += movement
            rounded += movement
            if path == '/measure':
                raw += 5
                rounded += 5
                if receiver != request['channel']:
                    raw += 1
                    rounded += 1
                receiver = request['channel']
            else:
                cost = 5 if response['clear_result'] == 'success' else 3
                raw += cost
                rounded += cost
            rounded = round(rounded, 6)
            position = destination
            action_index += 1
        elif path not in ('/enter', '/exit'):
            raise ValueError('Unknown committed operation')
        error = abs(rounded - event['virtual_time_s'])
        max_error = max(max_error, error)
        if error != 0 or receiver != event['receiver_channel'] or position.tolist() != event['position']:
            raise ValueError('Exact per-action rounded clock or state differs')
    if action_index != len(direct) or abs(raw - direct_record['virtual_seconds']) > 1e-9:
        raise ValueError('Unrounded reference or action count differs')
    if rounded != session['virtual_time_s']:
        raise ValueError('Final authoritative clock differs')
    old_checks = load(folder / 'checks.json')
    other_checks_pass = all(value for name, value in old_checks.items() if name != 'same_clock')
    delta = abs(rounded - raw)
    rounding_bound = action_index * 0.5e-6 + 1e-8
    return {'trial': folder.name, 'passed': other_checks_pass and delta <= rounding_bound,
            'original_aggregate_clock_check': old_checks['same_clock'], 'all_other_original_checks': other_checks_pass,
            'actions_checked': action_index, 'per_action_clock_error': max_error,
            'unrounded_seconds': raw, 'http_rounded_seconds': rounded,
            'aggregate_rounding_difference_s': delta, 'proved_rounding_bound_s': rounding_bound,
            'request_log_sha256': hashlib.sha256((folder / 'requests.jsonl').read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.drive.upper() != 'D:' or not output.is_relative_to(ROOT / 'reports' / 'joint_search_v4'):
        raise ValueError('Audit existing D-drive local evidence only')
    destination = output / 'http_clock_audit.json'
    if destination.exists():
        raise ValueError('Do not overwrite an existing audit')
    trials = [audit_trial(folder, output) for folder in sorted((output / 'http').iterdir())]
    unchanged = hashes() == load(output / 'selection.json')['source_hashes']
    result = {'created_local': datetime.now().isoformat(), 'official_calls': 0, 'source_hashes_unchanged': unchanged,
              'original_results_preserved': True,
              'explanation': 'MockArena rounds the accumulated world clock to 6 decimals after every physical action. The original aggregate 1e-6 comparison was incorrectly stricter than cumulative rounding. This audit reproduces every committed action exactly; it does not change the policy, direct metric, or original checks.',
              'auditor_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'mock_server_sha256': hashlib.sha256((ROOT / 'research' / 'interface_validation' / 'mock_server.py').read_bytes()).hexdigest(),
              'trials': trials, 'passed': unchanged and len(trials) == 6 and all(trial['passed'] for trial in trials)}
    save(destination, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result['passed']:
        raise AssertionError('Independent HTTP clock audit failed')


if __name__ == '__main__':
    main()
