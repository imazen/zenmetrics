import fcntl
import json
import os
from pathlib import Path
import tempfile
import unittest
from check_dvifmish_timing import protected_window

class WindowTests(unittest.TestCase):
    def test_host_lock_covers_window_and_releases_on_failure(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as directory:
            root = Path(directory)
            lock, receipt = root/'host.lock', root/'window.json'
            with self.assertRaisesRegex(RuntimeError, 'control failed'):
                with protected_window(lock, 0, receipt) as fd:
                    self.assertIsInstance(fd, int)
                    with lock.open('a') as rival:
                        with self.assertRaises(BlockingIOError):
                            fcntl.flock(rival, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self.assertIn('settled_unix_ns', json.loads(receipt.read_bytes()))
                    raise RuntimeError('control failed')
            with lock.open('a') as rival:
                fcntl.flock(rival, fcntl.LOCK_EX | fcntl.LOCK_NB)
            value = json.loads(receipt.read_bytes())
            self.assertLessEqual(value['acquired_unix_ns'], value['settled_unix_ns'])
            self.assertLessEqual(value['settled_unix_ns'], value['releasing_unix_ns'])
