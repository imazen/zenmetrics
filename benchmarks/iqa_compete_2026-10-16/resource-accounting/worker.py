#!/usr/bin/env python3
"""Exec a diagnostic worker with the harness's unchanged worker environment."""
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from durable import environment

binary, config_path, scratch_path = sys.argv[1:]
scratch = Path(scratch_path).resolve()
scratch.mkdir()
config = json.loads(Path(config_path).read_bytes())
os.execve(binary, [binary, config_path], environment(config, scratch))
