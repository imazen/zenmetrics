"""Locate the configured resource wrapper and preserve macOS process containment."""
import os
from pathlib import Path
import shutil


def resource_wrapper():
    configured = os.environ.get('IQA_RUN_HEAVY')
    executable = configured or shutil.which('run-heavy')
    if not executable or not Path(executable).is_file() or not os.access(executable, os.X_OK):
        raise RuntimeError('Set IQA_RUN_HEAVY to the executable resource wrapper')
    return str(Path(executable).resolve())


def timeout_command(system, seconds):
    if not 1 <= seconds <= 1100:
        raise ValueError('scope must be within 1..1100 seconds')
    if system == 'Darwin':
        # Keep the timed child in the wrapper's process group: its RSS guard
        # must terminate the entire workload, including on memory pressure.
        return ['gtimeout', '--foreground', '--kill-after=20s', f'{seconds}s']
    if system == 'Linux':
        return ['timeout', '--kill-after=20s', f'{seconds}s']
    raise RuntimeError('Resource supervision is not qualified on ' + system)
