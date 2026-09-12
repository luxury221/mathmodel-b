from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import adapter
from adapter import ROOT, SELECTED, load
from b2026_robot.client import ClientConfig, RobotClient
from b2026_robot.storage import Journal

sys.path.insert(0, str(ROOT / 'research/practice_baseline'))
from baseline import audit_log


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def evidence(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT / 'outputs') or path.suffix.lower() != '.png' or not path.is_file():
        raise ValueError('Require a saved public-window screenshot on D')
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def verify_batch(batch):
    batch = Path(batch).resolve()
    if not batch.is_relative_to(ROOT / 'reports/frozen_practice_v33'):
        raise ValueError('Invalid practice batch directory')
    protocol = load(batch / 'protocol.json')
    adapter.require_validation(protocol['validation'])
    if protocol['source_hashes'] != adapter.hashes() or protocol['selected'] != SELECTED or protocol['mode'] != 'practice_only':
        raise ValueError('Fixed batch configuration differs')
    return batch, protocol, load(batch / 'runs.json')


def summarize(batch, rows):
    summary = {}
    for problem in SELECTED:
        selected = [row for row in rows if row['problem'] == problem]
        complete = bool(selected) and all(row['state'] == 'completed' for row in selected)
        summary[problem] = {'attempts': len(selected), 'all_cleared': sum(row.get('all_cleared', False) for row in selected),
                            'fixed_batch_complete': len(selected) == 5 and complete,
                            'mean_seconds_per_source': sum(row['seconds_per_source'] for row in selected) / len(selected) if complete else None}
    save(batch / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description='Frozen q3/q4 candidates; explicit official PRACTICE only; never launch the simulator')
    parser.add_argument('command', choices=('freeze', 'begin', 'ready', 'run', 'finish', 'view'))
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--validation', type=Path)
    parser.add_argument('--problem', choices=('q3', 'q4'))
    parser.add_argument('--evidence', type=Path)
    parser.add_argument('--case-code')
    parser.add_argument('--robot-id')
    parser.add_argument('--total', type=int)
    parser.add_argument('--omni', type=int)
    parser.add_argument('--directional', type=int)
    parser.add_argument('--confirm-practice-only', action='store_true')
    args = parser.parse_args()
    batch = args.batch.resolve()
    if args.command == 'freeze':
        validated = adapter.require_validation(args.validation)
        if batch.exists() or not batch.is_relative_to(ROOT / 'reports/frozen_practice_v33'):
            raise ValueError('Use a new registered D-drive batch')
        batch.mkdir(parents=True)
        frozen = adapter.hashes()
        save(batch / 'protocol.json', {'created_local': datetime.now().isoformat(), 'source_hashes': frozen,
                                      'selected': SELECTED, 'validation': str(validated), 'mode': 'practice_only',
                                      'planned_each_problem': 5, 'formal_calls': 0})
        for relative in frozen:
            destination = batch / 'source_snapshot' / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, destination)
        save(batch / 'runs.json', [])
        print(batch, flush=True)
        return
    batch, protocol, rows = verify_batch(batch)
    if args.command == 'begin':
        if args.problem not in SELECTED or any(row['state'] != 'completed' for row in rows) or sum(row['problem'] == args.problem for row in rows) >= 5:
            raise ValueError('Prior attempt needs attention, invalid problem, or fixed batch limit reached')
        rows.append({'attempt': len(rows) + 1, 'problem': args.problem, 'state': 'awaiting_ready',
                     'before_start_evidence': evidence(args.evidence), 'began_local': datetime.now().isoformat()})
    elif args.command in ('ready', 'run', 'finish'):
        if not rows:
            raise ValueError('Register the attempt before clicking start')
        row = rows[-1]
        if args.command == 'ready':
            if row['state'] != 'awaiting_ready' or not args.case_code:
                raise ValueError('The latest attempt is not awaiting public readiness')
            if any(other.get('case_code') == args.case_code for other in rows[:-1]):
                raise ValueError('Case code already registered')
            row.update({'state': 'ready', 'case_code': args.case_code, 'ready_evidence': evidence(args.evidence)})
        elif args.command == 'run':
            if row['state'] != 'ready' or not args.confirm_practice_only or not args.robot_id:
                raise ValueError('No connection made: require registered practice readiness and robot ID')
            captured = Path(row['ready_evidence']['path'])
            if evidence(captured) != row['ready_evidence'] or time.time() - captured.stat().st_mtime > 600:
                raise ValueError('Ready evidence changed or is stale; do not enter')
            output = ROOT / 'logs/practice' / f'{time.strftime("%Y%m%d_%H%M%S")}_{row["problem"]}_frozen_v33_{uuid.uuid4().hex[:8]}'
            output.mkdir(parents=True, exist_ok=False)
            row.update({'state': 'running', 'log_directory': str(output)})
            save(batch / 'runs.json', rows)
            with Journal(output / 'requests.jsonl') as journal:
                client = RobotClient('http://127.0.0.1:2026', args.robot_id, journal, config=ClientConfig(), allow_official=True)
                result = adapter.run_session(client, row['problem'])
                result.update({'validation_evidence': protocol['validation'], 'public_case_code': row['case_code'],
                               'mode_note': 'Visible practice page and readiness screenshot inspected; HTTP does not reveal mode'})
                save(output / 'summary.json', result)
            row['state'] = 'awaiting_public_result' if result['status'] == 'completed' else 'needs_attention'
            row['client_status'] = result['status']
            print(json.dumps({key: result[key] for key in ('problem', 'status', 'virtual_time_s', 'cleared_channels', 'error')}, ensure_ascii=False), flush=True)
        else:
            if row['state'] not in ('awaiting_public_result', 'needs_attention') or None in (args.total, args.omni, args.directional):
                raise ValueError('Public post-exit aggregate counts are required')
            if not 10 <= args.total <= 16 or min(args.omni, args.directional) < 0 or args.omni + args.directional != args.total:
                raise ValueError('Invalid public source totals')
            if row['problem'] == 'q3' and args.directional != 0:
                raise ValueError('Q3 must have no directional sources')
            audit = audit_log(Path(row['log_directory']))
            complete = audit['client_status'] == 'completed' and audit['exited'] and not audit['error'] and audit['cleared'] == args.total
            row.update({'state': 'completed' if complete else 'needs_attention', 'source_count': args.total,
                        'omni': args.omni, 'directional': args.directional, 'all_cleared': complete,
                        'seconds_per_source': audit['virtual_s'] / args.total if complete else None,
                        'audit': audit, 'result_evidence': evidence(args.evidence)})
    save(batch / 'runs.json', rows)
    print(json.dumps(summarize(batch, rows), ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
