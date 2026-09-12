from __future__ import annotations

import argparse
import json
import time
import uuid

from b2026_candidate.session import run_session
from b2026_candidate.validation import require_validation
from b2026_robot.client import ClientConfig, RobotClient, RobotError
from b2026_robot.storage import ROOT, Journal, d_path


def main():
    parser = argparse.ArgumentParser(description='Validated q3 v3 candidate; explicit practice only, no simulator launch.')
    parser.add_argument('--robot-id', required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:2026')
    parser.add_argument('--confirm-practice-ready', action='store_true')
    args = parser.parse_args()
    if not args.confirm_practice_ready:
        parser.error('No connection made. First confirm the visible simulator is in q3 PRACTICE and ready.')
    try:
        validation = require_validation()
    except RobotError as error:
        parser.error(str(error))
    output = d_path(ROOT / 'logs' / 'practice') / f'{time.strftime("%Y%m%d_%H%M%S")}_q3_v3_{uuid.uuid4().hex[:8]}'
    output.mkdir(parents=True, exist_ok=False)
    with Journal(output / 'requests.jsonl') as journal:
        client = RobotClient(args.base_url, args.robot_id, journal, config=ClientConfig(), allow_official=True)
        summary = run_session(client, 'q3')
        summary['validation_evidence'] = str(validation)
        summary['mode_note'] = 'User-confirmed q3 practice; protocol does not reveal practice versus formal mode.'
        (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Artifacts: {output}')
    return 0 if summary['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
