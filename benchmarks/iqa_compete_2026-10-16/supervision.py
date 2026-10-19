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


def scoring_command(command, *, system, seconds, memory_log=None):
    """Keep the scorer directly inside its deadline; observe RSS from outside.

    GNU time is an observing parent, not the child the deadline must kill.
    Putting time inside timeout can let a signal-resistant scorer outlive it.
    """
    wrapped = [*timeout_command(system, seconds), *map(str, command)]
    if memory_log is not None:
        if system != 'Linux':
            raise RuntimeError('This GNU time memory receipt is Linux-specific')
        wrapped = ['/usr/bin/time', '-v', '-o', str(memory_log), *wrapped]
    return wrapped
