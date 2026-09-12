from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def d_path(path):
    resolved = Path(path).resolve()
    if resolved.drive.upper() != 'D:':
        raise ValueError('All generated artifacts must remain on drive D.')
    return resolved


class Journal:
    def __init__(self, path):
        self.path = d_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open('x', encoding='utf-8', newline='\n')
        self.lock = threading.Lock()

    def write(self, event, **fields):
        record = {'event': event, 'local_time_ns': time.time_ns(), **fields}
        text = json.dumps(record, ensure_ascii=False, allow_nan=False)
        with self.lock:
            self.stream.write(text + '\n')
            self.stream.flush()
            os.fsync(self.stream.fileno())

    def close(self):
        self.stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *_error):
        self.close()
