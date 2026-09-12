from __future__ import annotations

import logging
import threading

from .client import ConcurrentAction, RobotError
from .policy import make_policy

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


def run_session(client, problem, q4_network='dual21', *, watchdog=True):
    finished = threading.Event()
    watchdog_errors = []
    monitor = None
    result = None
    error_text = None
    metadata = {}

    def watch_deadline():
        while not finished.wait(0.05):
            remaining = client.remaining_seconds()
            if remaining is None or remaining > client.config.exit_margin_s:
                continue
            client.request_stop('real_time_reserve')
            if client.can_exit():
                try:
                    client.exit()
                except ConcurrentAction:
                    continue
                except Exception as error:
                    LOGGER.exception('Deadline exit failed')
                    watchdog_errors.append(f'{type(error).__name__}: {error}')
            if client.exited or client.blocked_reason or remaining <= 0:
                return

    try:
        policy, metadata = make_policy(client, problem, q4_network)
        client.journal.write('policy_ready', **metadata)
        client.enter()
        if watchdog:
            monitor = threading.Thread(target=watch_deadline, name='robot-deadline-watchdog', daemon=True)
            monitor.start()
        result = policy.run()
    except (Exception, KeyboardInterrupt) as error:
        LOGGER.exception('Session stopped before policy completion')
        error_text = f'{type(error).__name__}: {error}'
        client.request_stop('policy_or_interface_error')
    finally:
        finished.set()
        if monitor is not None:
            monitor.join()
        if client.can_exit():
            try:
                client.exit()
            except (RobotError, OSError) as error:
                error_text = error_text or f'{type(error).__name__}: {error}'
    summary = {**metadata, **client.summary(), 'policy_completed': result is not None,
               'error': error_text, 'watchdog_errors': watchdog_errors,
               'status': 'completed' if result is not None and client.exited and not error_text
               and not client.stop_event.is_set() and not watchdog_errors else 'needs_attention'}
    if result is not None:
        summary['policy_stats'] = result['stats']
        summary['declared_absent'] = result['declared_absent']
    client.journal.write('session_finished', summary=summary)
    return summary
