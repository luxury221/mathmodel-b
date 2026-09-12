from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'research/frozen_practice_v33'))
import adapter
import practice


@pytest.mark.parametrize('state,confirmed,robot', [('ready', False, 'unit-team'), ('ready', True, None), ('awaiting_ready', True, 'unit-team')])
def test_run_without_ready_authorization_never_constructs_client(monkeypatch, tmp_path, state, confirmed, robot):
    rows = [{'problem': 'q3', 'state': state}]
    monkeypatch.setattr(practice, 'verify_batch', lambda _batch: (tmp_path, {}, rows))
    def forbidden(*_arguments, **_keywords):
        raise AssertionError('A client must not be constructed')
    monkeypatch.setattr(practice, 'RobotClient', forbidden)
    arguments = ['practice.py', 'run', '--batch', str(tmp_path)]
    if confirmed:
        arguments.append('--confirm-practice-only')
    if robot:
        arguments.extend(['--robot-id', robot])
    monkeypatch.setattr(sys, 'argv', arguments)
    with pytest.raises(ValueError, match='No connection made'):
        practice.main()


def test_invalid_network_is_rejected_before_factory(monkeypatch):
    def forbidden(*_arguments, **_keywords):
        raise AssertionError('Original factory must not be called')
    monkeypatch.setattr(adapter, 'original_factory', forbidden)
    with pytest.raises(ValueError, match='registered'):
        adapter.make_policy(object(), 'q4', 'grid25')


def test_unfinished_attempt_blocks_new_start(monkeypatch, tmp_path):
    rows = [{'problem': 'q3', 'state': 'awaiting_public_result'}]
    monkeypatch.setattr(practice, 'verify_batch', lambda _batch: (tmp_path, {}, rows))
    monkeypatch.setattr(sys, 'argv', ['practice.py', 'begin', '--batch', str(tmp_path), '--problem', 'q4'])
    with pytest.raises(ValueError, match='Prior attempt'):
        practice.main()


def test_stale_screenshot_blocks_connection(monkeypatch, tmp_path):
    screenshot = tmp_path / 'ready.png'
    screenshot.write_bytes(b'unit screenshot fixture')
    os.utime(screenshot, (1, 1))
    evidence = {'path': str(screenshot), 'sha256': 'fixture'}
    rows = [{'problem': 'q4', 'state': 'ready', 'ready_evidence': evidence}]
    monkeypatch.setattr(practice, 'verify_batch', lambda _batch: (tmp_path, {}, rows))
    monkeypatch.setattr(practice, 'evidence', lambda _path: evidence)
    def forbidden(*_arguments, **_keywords):
        raise AssertionError('A client must not be constructed')
    monkeypatch.setattr(practice, 'RobotClient', forbidden)
    monkeypatch.setattr(sys, 'argv', ['practice.py', 'run', '--batch', str(tmp_path), '--robot-id', 'unit-team', '--confirm-practice-only'])
    with pytest.raises(ValueError, match='stale'):
        practice.main()
