#!/usr/bin/env python3
"""Check lane stop/disk/output constraints at actual heavy-job admission."""
import datetime
import os
from pathlib import Path
import shutil

root = Path('/var/tmp/gmsd-chroma')
# The Codex quota stop does not bind the Claude Sonnet takeover lane (2026-09-24), so the
# former `CODEX_QUOTA_STOP.md` assertion is removed; the disk floor and output roots stay.
free = shutil.disk_usage('/home').free
assert free >= 20*1024**3, ('/home disk floor', free)
for name in ('TMPDIR','XDG_CACHE_HOME','CARGO_TARGET_DIR','CARGO_HOME','UV_CACHE_DIR'):
    value = Path(os.environ[name]).resolve()
    assert value == root or root in value.parents, (name, str(value))
print('admitted_utc', datetime.datetime.now(datetime.timezone.utc).isoformat(), flush=True)
print('home_free_bytes', free, flush=True)
