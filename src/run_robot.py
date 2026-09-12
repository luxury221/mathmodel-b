from __future__ import annotations

import argparse
import json
import time
import uuid

from b2026_robot.client import ClientConfig, RobotClient
from b2026_robot.session import run_session
from b2026_robot.storage import ROOT, Journal, d_path


def main():
    parser = argparse.ArgumentParser(description='Explicitly authorized official practice runner; no simulator launch.')
    parser.add_argument('--problem', choices=('q3', 'q4'), required=True)
    parser.add_argument('--robot-id', required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:2026')
    parser.add_argument('--q4-network', choices=('dual21', 'grid25'), default='dual21')
    parser.add_argument('--confirm-practice-ready', action='store_true')
    parser.add_argument('--output-root', type=d_path, default=ROOT / 'logs' / 'practice')
    args = parser.parse_args()
    if not args.confirm_practice_ready:
        parser.error('No network call made. Confirm the simulator is in PRACTICE mode and ready before opting in.')
    output = d_path(args.output_root) / f'{time.strftime("%Y%m%d_%H%M%S")}_{args.problem}_{uuid.uuid4().hex[:8]}'
    output.mkdir(parents=True, exist_ok=False)
    with Journal(output / 'requests.jsonl') as journal:
        client = RobotClient(args.base_url, args.robot_id, journal, config=ClientConfig(), allow_official=True)
        summary = run_session(client, args.problem, args.q4_network)
        summary['mode_note'] = 'User-confirmed practice; the protocol cannot detect practice versus formal mode.'
        (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Artifacts: {output}')
    return 0 if summary['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
