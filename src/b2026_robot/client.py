from __future__ import annotations

import http.client
import ipaddress
import json
import math
import numbers
import threading
import time
import unicodedata
import uuid
from dataclasses import dataclass
from urllib.parse import urlsplit

MAX_BODY = 65536


class RobotError(RuntimeError):
    pass


class ProtocolError(RobotError):
    pass


class RequestRejected(RobotError):
    def __init__(self, status, response):
        super().__init__(f'HTTP {status}, accepted={response["accepted"]}; stopped without state advancement')
        self.status = status
        self.response = response


class TransportFailure(RobotError):
    pass


class ConcurrentAction(RobotError):
    pass


class BudgetStop(RobotError):
    pass


class AccountingMismatch(RobotError):
    pass


def identifier(value, name, limit):
    if not isinstance(value, str) or not value:
        raise ValueError(f'{name} must be a nonempty string')
    if any(unicodedata.category(character) in ('Cc', 'Cf', 'Cs') for character in value):
        raise ValueError(f'{name} contains control, format or surrogate characters')
    if len(value.encode('utf-8')) > limit:
        raise ValueError(f'{name} exceeds {limit} UTF-8 bytes')
    return value


def finite_number(value, name, lower=0, upper=math.inf):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f'{name} must be a finite number')
    result = float(value)
    if not math.isfinite(result) or not lower <= result <= upper:
        raise ValueError(f'{name} is outside [{lower}, {upper}]')
    return result


def action_fields(position, channel):
    if len(position) != 2:
        raise ValueError('position must have two coordinates')
    coordinates = [finite_number(value, 'coordinate', -2e6, 2e6) for value in position]
    channel_number = finite_number(channel, 'channel', 1, 20)
    if not channel_number.is_integer():
        raise ValueError('channel must be an integer')
    return {'position': dict(zip(('x', 'y'), coordinates)), 'channel': int(channel_number)}


def strict_json(raw):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate JSON key: {key}')
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f'Invalid JSON number: {value}')

    value = json.loads(raw.decode('utf-8'), object_pairs_hook=unique_object, parse_constant=invalid_constant)
    if not isinstance(value, dict):
        raise TypeError('JSON response must be an object')
    return value


def validate_response(path, status, raw):
    try:
        response = strict_json(raw)
        if type(response['accepted']) is not bool:
            raise ValueError('accepted must be boolean')
        finite_number(response['real_timestamp_ms'], 'real_timestamp_ms')
        finite_number(response['virtual_time_s'], 'virtual_time_s')
        if not response['accepted']:
            if response['virtual_time_s'] != 0:
                raise ValueError('Rejected responses must carry virtual_time_s=0')
            return response
        if status != 200:
            raise ValueError('accepted=true requires HTTP 200')
        if path == '/enter':
            maximum = finite_number(response['max_real_duration_s'], 'max_real_duration_s', 0, 1200)
            finite_number(response['max_virtual_duration_s'], 'max_virtual_duration_s', 0, 360000)
            remaining = finite_number(response['remaining_real_duration_s'], 'remaining_real_duration_s', 0, maximum)
            if not remaining.is_integer():
                raise ValueError('remaining_real_duration_s must be integer')
        elif path == '/measure':
            if response['measure_result'] not in ('direction', 'near', 'no_signal'):
                raise ValueError('Unknown measure_result')
            if response['measure_result'] == 'direction':
                if finite_number(response['svd_deg'], 'svd_deg', 0, 360) == 360:
                    raise ValueError('svd_deg must be less than 360')
            elif 'svd_deg' in response:
                raise ValueError('Only direction responses may carry svd_deg')
        elif path == '/clear':
            if response['clear_result'] not in ('success', 'no_target_in_range'):
                raise ValueError('Unknown clear_result')
        elif path == '/exit' and response['exit_reason'] != 'user_exit':
            raise ValueError('Unexpected exit_reason')
        return response
    except (ValueError, KeyError, TypeError, RecursionError, OverflowError) as error:
        raise ProtocolError(f'Invalid {path} response: {error}') from error


@dataclass(frozen=True)
class ClientConfig:
    request_timeout_s: float = 3.0
    max_attempts: int = 4
    retry_backoff_s: float = 0.15
    exit_margin_s: float = 10.0
    virtual_margin_s: float = 0.01
    startup_budget_s: float = 20.0

    def __post_init__(self):
        for name in ('request_timeout_s', 'startup_budget_s'):
            if finite_number(getattr(self, name), name) <= 0:
                raise ValueError(f'{name} must be positive')
        for name in ('retry_backoff_s', 'exit_margin_s', 'virtual_margin_s'):
            finite_number(getattr(self, name), name)
        if type(self.max_attempts) is not int or not 1 <= self.max_attempts <= 10:
            raise ValueError('max_attempts must be in 1..10')


class RobotClient:
    def __init__(self, base_url, robot_id, journal, *, config=None, mock_token=None,
                 allow_official=False, clock=time.monotonic, sleeper=time.sleep):
        parsed = urlsplit(base_url)
        if parsed.scheme != 'http' or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('Only plain HTTP loopback URLs without credentials/query are allowed')
        if parsed.path not in ('', '/') or not parsed.port or not ipaddress.ip_address(parsed.hostname).is_loopback:
            raise ValueError('Use a literal loopback IP with an explicit port and no path')
        if bool(mock_token) == bool(allow_official):
            raise ValueError('Choose a mock capability or explicitly allow an official connection')
        self.host, self.port = parsed.hostname, parsed.port
        self.robot_id = identifier(robot_id, 'robot_id', 64)
        self.journal = journal
        self.config = config or ClientConfig()
        self.mock_token = mock_token
        self.authorized = allow_official
        self.clock, self.sleep = clock, sleeper
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.stop_reason = None
        self.blocked_reason = None
        self.pending = None
        self.uncertain = False
        self.entered = False
        self.exited = False
        self.position = (0.0, 0.0)
        self.receiver_channel = 1
        self.virtual_seconds = 0.0
        self.deadline = None
        self.max_virtual_seconds = 360000.0
        self.session_id = uuid.uuid4().hex
        self.sequence = 0
        self.last_action = None
        self.cleared_channels = set()
        self.accepted_actions = 0
        self.retry_count = 0
        self.journal.write('client_created', base_url=base_url, robot_id=robot_id,
                           mode='official_explicit_opt_in' if allow_official else 'local_mock',
                           session_id=self.session_id, config=self.config.__dict__)

    def remaining_seconds(self):
        return None if self.deadline is None else max(0.0, self.deadline - self.clock())

    def request_stop(self, reason):
        if not self.stop_event.is_set():
            self.stop_reason = reason
            self.stop_event.set()

    def _guard(self, path, fields):
        if self.blocked_reason or self.uncertain or self.pending:
            raise RobotError(f'No new action after unresolved or rejected request: {self.blocked_reason}')
        if self.exited:
            raise RobotError('Session already exited; no further interface calls are allowed')
        if path == '/enter':
            if self.entered:
                raise RobotError('A client cannot enter twice')
            return
        if not self.entered:
            raise RobotError('Enter must succeed before actions')
        if self.clock() >= self.deadline:
            raise BudgetStop('Real deadline reached; do not query /exit after termination')
        if path == '/exit':
            return
        if self.stop_event.is_set() or self.clock() >= self.deadline - self.config.exit_margin_s:
            self.request_stop('real_time_reserve')
            raise BudgetStop('Reserved real time for a safe exit')
        position = tuple(fields['position'].values())
        worst_cost = math.dist(self.position, position) / 5 + 5
        if path == '/measure':
            worst_cost += fields['channel'] != self.receiver_channel
        if self.virtual_seconds + worst_cost >= self.max_virtual_seconds - self.config.virtual_margin_s:
            self.request_stop('virtual_time_reserve')
            raise BudgetStop('Next action could exhaust the virtual time budget')

    def _read_wire(self, connection, method, path, raw, deadline):
        headers = {'Content-Type': 'application/json; charset=utf-8', 'Connection': 'close'}
        remaining = deadline - self.clock()
        if remaining <= 0:
            raise TimeoutError('Request deadline reached before sending')
        if connection.sock is not None:
            connection.sock.settimeout(remaining)
        connection.request(method, path, body=raw, headers=headers)
        remaining = deadline - self.clock()
        if remaining <= 0:
            raise TimeoutError('Request deadline reached before headers')
        if connection.sock is not None:
            connection.sock.settimeout(remaining)
        with connection.getresponse() as response:
            if response.length is not None and response.length > MAX_BODY:
                raise ProtocolError('Response body exceeds 65536 bytes')
            remaining = deadline - self.clock()
            if remaining <= 0:
                raise TimeoutError('Request deadline reached before body')
            if connection.sock is not None:
                connection.sock.settimeout(remaining)
            payload = response.read(MAX_BODY + 1)
            if response.length not in (None, 0):
                raise http.client.IncompleteRead(payload, response.length)
            if len(payload) > MAX_BODY:
                raise ProtocolError('Response body exceeds 65536 bytes')
            content_type = response.getheader('Content-Type', '').split(';')[0].strip().lower()
            if content_type != 'application/json':
                raise ProtocolError('Response Content-Type is not application/json')
            return response.status, payload

    def _verify_mock(self):
        if self.authorized:
            return
        connection = http.client.HTTPConnection(self.host, self.port, timeout=self.config.request_timeout_s)
        try:
            connection.connect()
            status, payload = self._read_wire(
                connection, 'GET', '/__b2026_local_mock__', b'', self.clock() + self.config.request_timeout_s,
            )
            proof = strict_json(payload)
            if status != 200 or proof != {'service': 'b2026-local-mock', 'token': self.mock_token}:
                raise RobotError('Local mock capability mismatch; no action sent')
            self.authorized = True
            self.journal.write('mock_verified', host=self.host, port=self.port)
        finally:
            connection.close()

    def _apply(self, path, fields, response, earliest_start):
        old_time = self.virtual_seconds
        expected = old_time
        if path == '/enter':
            self.entered = True
            self.deadline = earliest_start + response['remaining_real_duration_s']
            self.max_virtual_seconds = float(response['max_virtual_duration_s'])
            expected = 0.0
        elif path in ('/measure', '/clear'):
            destination = (fields['position']['x'], fields['position']['y'])
            expected += math.dist(self.position, destination) / 5
            self.position = destination
            if path == '/measure':
                expected += 5 + int(fields['channel'] != self.receiver_channel)
                self.receiver_channel = fields['channel']
            else:
                expected += 5 if response['clear_result'] == 'success' else 3
                if response['clear_result'] == 'success':
                    self.cleared_channels.add(fields['channel'])
        elif path == '/exit':
            self.exited = True
        self.virtual_seconds = float(response['virtual_time_s'])
        self.accepted_actions += 1
        self.last_action = (path, fields, response)
        self.journal.write('committed', path=path, request_id=fields['request_id'],
                           position=self.position, receiver_channel=self.receiver_channel,
                           virtual_time_s=self.virtual_seconds, expected_time_s=expected,
                           remaining_real_s=self.remaining_seconds())
        if abs(self.virtual_seconds - expected) > 3e-6:
            self.request_stop('accounting_mismatch')
            raise AccountingMismatch(f'{path}: expected {expected}, server {self.virtual_seconds}')

    def _post(self, path, action=None):
        if not self.lock.acquire(blocking=False):
            raise ConcurrentAction('Another action is in flight; concurrent new actions are forbidden')
        try:
            fields = action or {}
            self._guard(path, fields)
            self._verify_mock()
            self.sequence += 1
            fields = {'arena_id': 'default', 'robot_id': self.robot_id,
                      'request_id': f'{self.session_id}-{self.sequence}', **fields}
            raw = json.dumps(fields, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')
            earliest_start = self.clock()
            deadline = earliest_start + self.config.startup_budget_s if path == '/enter' else self.deadline
            if path in ('/measure', '/clear'):
                deadline -= self.config.exit_margin_s
            self.journal.write('request', path=path, payload=fields, body_utf8=raw.decode('utf-8'))
            self.pending = (path, fields)
            transport_error = None
            for attempt in range(1, self.config.max_attempts + 1):
                remaining = deadline - self.clock()
                if remaining <= 0:
                    break
                timeout = min(self.config.request_timeout_s, remaining)
                connection = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
                self.journal.write('attempt', request_id=fields['request_id'], attempt=attempt)
                wire_response = None
                attempt_deadline = min(deadline, self.clock() + timeout)
                try:
                    connection.connect()
                    if self.clock() >= deadline and not self.uncertain:
                        self.pending = None
                        self.request_stop('real_time_reserve')
                        raise BudgetStop('Connection consumed action budget; no action sent')
                    if self.clock() >= attempt_deadline:
                        raise TimeoutError('Connection consumed the attempt budget')
                    self.uncertain = True
                    wire_response = self._read_wire(connection, 'POST', path, raw, attempt_deadline)
                except (OSError, http.client.HTTPException) as error:
                    transport_error = error
                    self.journal.write('transport_error', request_id=fields['request_id'], attempt=attempt,
                                       error_type=type(error).__name__, message=str(error), uncertain=self.uncertain)
                except ProtocolError:
                    self.blocked_reason = 'invalid_response'
                    raise
                finally:
                    connection.close()
                if wire_response is not None:
                    status, payload = wire_response
                    self.journal.write('response', request_id=fields['request_id'], attempt=attempt,
                                       http_status=status, body_utf8=payload.decode('utf-8', errors='replace'))
                    try:
                        response = validate_response(path, status, payload)
                    except ProtocolError:
                        self.blocked_reason = 'invalid_response'
                        raise
                    if status != 200 or not response['accepted']:
                        self.blocked_reason = 'request_rejected'
                        self.pending = None
                        self.uncertain = transport_error is not None
                        raise RequestRejected(status, response)
                    self.pending = None
                    self.uncertain = False
                    self._apply(path, fields, response, earliest_start)
                    return response
                if attempt < self.config.max_attempts and self.clock() < deadline:
                    self.retry_count += 1
                    self.sleep(min(self.config.retry_backoff_s * 2 ** (attempt - 1), max(0, deadline - self.clock())))
            if transport_error is None and not self.uncertain:
                self.pending = None
                self.request_stop('real_time_reserve')
                raise BudgetStop('No action sent: request deadline elapsed before the first attempt')
            self.blocked_reason = 'transport_exhausted'
            raise TransportFailure(f'{path} unavailable; action outcome uncertain={self.uncertain}: {transport_error}')
        except OSError:
            self.blocked_reason = 'journal_or_local_io_failure'
            raise
        finally:
            self.lock.release()

    def enter(self):
        return self._post('/enter')

    def measure(self, position, channel):
        return self._post('/measure', action_fields(position, channel))

    def clear(self, position, channel):
        return self._post('/clear', action_fields(position, channel))

    def exit(self):
        return self._post('/exit')

    def can_exit(self):
        return (self.entered and not self.exited and not self.blocked_reason and not self.uncertain
                and self.pending is None and self.clock() < self.deadline
                and self.virtual_seconds < self.max_virtual_seconds)

    def summary(self):
        return {'entered': self.entered, 'exited': self.exited, 'position': self.position,
                'receiver_channel': self.receiver_channel, 'virtual_time_s': self.virtual_seconds,
                'accepted_actions': self.accepted_actions, 'retry_count': self.retry_count,
                'cleared_channels': sorted(self.cleared_channels), 'uncertain': self.uncertain,
                'pending_request': self.pending, 'blocked_reason': self.blocked_reason,
                'stop_reason': self.stop_reason, 'remaining_real_s': self.remaining_seconds()}
