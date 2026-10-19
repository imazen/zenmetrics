#!/usr/bin/env python3
"""Observe native checkpoint fixture scores without changing numerical gates.

This standalone probe compiles the unmodified modules with Rust's default
target settings. A successful run means the observation executed; acceptance
remains the existing checkpoint tests. Keep all output for platform diagnosis.
"""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    repo = Path(__file__).resolve().parents[2]
    output = Path.home() / 'tmp' / ('checkpoint-math-' + str(os.getpid()))
    output.mkdir(parents=True, exist_ok=False)
    files = {'flip.rs': repo / 'crates/zenmetrics-cli/src/metrics/flip.rs',
             'ssimulacra1.rs': repo / 'crates/zenmetrics-cli/src/metrics/ssimulacra1.rs',
             'probe.rs': Path(__file__).with_name('probe_checkpoint_math.rs')}
    for name, source in files.items():
        (output / name).write_bytes(source.read_bytes())
    (output / 'main.rs').write_text('#[path="flip.rs"] mod flip;\n'
                                   '#[path="ssimulacra1.rs"] mod ssimulacra1;\n'
                                   'include!("probe.rs");\n')
    env = dict(os.environ, TMPDIR=str(output))
    for key in ['RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS', 'CARGO_BUILD_RUSTFLAGS']:
        env.pop(key, None)
    binary = output / ('probe.exe' if os.name == 'nt' else 'probe')
    commands = [['rustc', '--edition=2024', '-O', str(output / 'main.rs'), '-o', str(binary)],
                [str(binary)]]
    records = []
    for name, command in zip(['compile', 'observe'], commands, strict=True):
        with (output / (name + '.log')).open('x') as log:
            result = subprocess.run(command, cwd=output, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=120)
        records.append(dict(command=command, returncode=result.returncode,
                            log_sha256=digest(output / (name + '.log'))))
        print((output / (name + '.log')).read_text(), end='', flush=True)
        assert result.returncode == 0, (name, result.returncode, str(output))
    revision_command = (['jj', 'log', '--no-graph', '--ignore-working-copy', '-r', '@', '-T', 'commit_id']
                        if (repo / '.jj').exists() else ['git', 'rev-parse', 'HEAD'])
    record = dict(source=subprocess.check_output(revision_command, cwd=repo, text=True).strip(),
                  platform=platform.platform(), machine=platform.machine(),
                  rustc=subprocess.check_output(['rustc', '-vV'], text=True),
                  module_sha256={name: digest(output / name) for name in files},
                  worker_sha256=digest(Path(__file__)), commands=records,
                  qualified_timing_cells=0, numeric_acceptance='existing checkpoint tests; no new bounds')
    (output / 'observation.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
