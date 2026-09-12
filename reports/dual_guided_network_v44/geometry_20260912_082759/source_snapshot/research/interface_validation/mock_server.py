from __future__ import annotations

import http.server
import json
import math
import threading
import time
import unicodedata
import uuid
from dataclasses import dataclass


@dataclass
class Fault:
    path: str
    kind: str
    value: object = None


class MockArena:
    """Public-protocol test double, not the official simulator or hidden cases."""

    def __init__(self, world, robot_id='local-test-team', *, faults=(), remaining_s=1200,
                 max_virtual_s=360000, clock=time.monotonic, close_after_exit=False):
        self.world = world
        self.robot_id = robot_id
        self.faults = list(faults)
        self.remaining_s = remaining_s
        self.max_virtual_s = max_virtual_s
        self.clock = clock
        self.close_after_exit = close_after_exit
        self.token = uuid.uuid4().hex
        self.entered = False
        self.exited = False
        self.deadline = None
        self.cache = {}
        self.requests = []
        self.executions = []
        self.lock = threading.Lock()
        self.server = None
        self.thread = None

    def response(self, accepted, **fields):
        return {'accepted': accepted, 'real_timestamp_ms': time.time_ns() // 1_000_000,
                'virtual_time_s': round(self.world.virtual_seconds, 6) if accepted else 0, **fields}

    def reject(self, status=200):
        return status, self.response(False), None

    def validate(self, path, raw):
        def object_pairs(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError('duplicate key')
                result[key] = value
            return result

        def reject_number(_value):
            raise ValueError('nonfinite JSON constant')

        data = json.loads(raw.decode('utf-8'), object_pairs_hook=object_pairs, parse_constant=reject_number)
        if not isinstance(data, dict):
            raise TypeError('not an object')
        required = {'arena_id', 'robot_id', 'request_id'}
        if path in ('/measure', '/clear'):
            required |= {'position', 'channel'}
        if not required <= data.keys():
            raise ValueError('missing fields')
        for name, limit in (('robot_id', 64), ('request_id', 128)):
            value = data[name]
            if not isinstance(value, str) or not 1 <= len(value.encode('utf-8')) <= limit:
                raise ValueError('bad identifier')
            if any(unicodedata.category(character) in ('Cc', 'Cf', 'Cs') for character in value):
                raise ValueError('invisible identifier')
        if not isinstance(data['arena_id'], str):
            raise TypeError('bad arena type')
        unknown = data.keys() - required
        if path in ('/measure', '/clear'):
            position, channel = data['position'], data['channel']
            if type(channel) not in (int, float) or not 1 <= channel <= 20 or int(channel) != channel:
                raise ValueError('invalid channel')
            if not isinstance(position, dict) or not {'x', 'y'} <= position.keys():
                raise ValueError('invalid position')
            unknown |= position.keys() - {'x', 'y'}
            for coordinate in position.values():
                if type(coordinate) not in (int, float) or not math.isfinite(coordinate) or abs(coordinate) > 2e6:
                    raise ValueError('invalid coordinate')
        if unknown or data['arena_id'] != 'default' or data['robot_id'] != self.robot_id:
            return None
        return data

    def handle(self, path, raw):
        with self.lock:
            self.requests.append({'path': path, 'body_utf8': raw.decode('utf-8', errors='replace')})
            if path not in ('/enter', '/measure', '/clear', '/exit'):
                return self.reject(404)
            try:
                data = self.validate(path, raw)
            except (ValueError, TypeError, OverflowError, RecursionError):
                return self.reject(400)
            if data is None:
                return self.reject()
            if self.entered and (self.clock() >= self.deadline or self.world.virtual_seconds >= self.max_virtual_s):
                return 0, None, None
            if self.exited and self.close_after_exit:
                return 0, None, None
            previous = self.cache.get(data['request_id'])
            if previous is not None:
                old_path, old_body, old_response = previous
                if path != old_path or raw != old_body:
                    return self.reject(409)
                return 200, old_response, None
            if self.exited:
                return 0, None, None
            fault = next((entry for entry in self.faults if entry.path == path), None)
            if fault is not None:
                self.faults.remove(fault)
                if fault.kind == 'drop_before':
                    return 0, None, fault
                if fault.kind == 'reject':
                    return self.reject()
                if fault.kind == 'status':
                    return self.reject(fault.value)
            if path == '/enter':
                if self.entered:
                    return self.reject()
                self.entered = True
                self.deadline = self.clock() + self.remaining_s
                response = self.response(True, max_virtual_duration_s=self.max_virtual_s,
                                         max_real_duration_s=1200, remaining_real_duration_s=self.remaining_s)
            elif not self.entered:
                return self.reject()
            elif path == '/exit':
                self.exited = True
                response = self.response(True, exit_reason='user_exit')
            else:
                position = (data['position']['x'], data['position']['y'])
                channel = int(data['channel'])
                result = getattr(self.world, path[1:])(position, channel)
                self.world.virtual_seconds = round(self.world.virtual_seconds, 6)
                fields = {path[1:] + '_result': result['result']}
                if 'bearing_deg' in result:
                    fields['svd_deg'] = result['bearing_deg']
                response = self.response(True, **fields)
            self.cache[data['request_id']] = (path, raw, response)
            self.executions.append({'path': path, 'request_id': data['request_id']})
            return 200, response, fault

    def __enter__(self):
        arena = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, *_args):
                return

            def send_json(self, status, response, fault=None):
                if status == 0 or (fault is not None and fault.kind == 'drop_after'):
                    self.close_connection = True
                    return
                if fault is not None and fault.kind == 'delay_after':
                    time.sleep(fault.value)
                if fault is not None and fault.kind == 'time_shift':
                    response = {**response, 'virtual_time_s': response['virtual_time_s'] + fault.value}
                raw = json.dumps(response, ensure_ascii=False, allow_nan=False).encode('utf-8')
                if fault is not None and fault.kind == 'malformed_after':
                    raw = b'{"accepted":true, "broken":'
                self.send_response(status)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Content-Length', str(len(raw)))
                self.send_header('Connection', 'close')
                self.end_headers()
                try:
                    self.wfile.write(raw[:len(raw) // 2] if fault is not None and fault.kind == 'truncate_after' else raw)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass
                self.close_connection = True

            def do_GET(self):
                if self.path == '/__b2026_local_mock__':
                    self.send_json(200, {'service': 'b2026-local-mock', 'token': arena.token})
                else:
                    self.send_json(405 if self.path in ('/enter', '/measure', '/clear', '/exit') else 404,
                                   arena.response(False))

            def do_POST(self):
                self.connection.settimeout(2)
                content_type = self.headers.get('Content-Type', '').lower().replace(' ', '')
                if (content_type not in ('application/json', 'application/json;charset=utf-8')
                        or self.headers.get('Content-Encoding', 'identity').lower() != 'identity'):
                    self.send_json(415, arena.response(False))
                    return
                try:
                    length = int(self.headers.get('Content-Length', '-1'))
                    if length < 0:
                        raise ValueError('missing length')
                except ValueError:
                    self.send_json(400, arena.response(False))
                    return
                if length > 65536:
                    self.send_json(413, arena.response(False))
                    return
                try:
                    raw = self.rfile.read(length)
                except OSError:
                    self.close_connection = True
                    return
                if len(raw) != length:
                    self.close_connection = True
                    return
                self.send_json(*arena.handle(self.path, raw))

        self.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = False
        self.base_url = f'http://127.0.0.1:{self.server.server_port}'
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': 0.02}, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_error):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
