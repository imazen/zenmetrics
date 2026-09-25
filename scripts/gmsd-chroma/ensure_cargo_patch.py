#!/usr/bin/env python3
"""Keep the frozen zenbench patch in this lane's private Cargo home.

Clippy's child Cargo invocation does not retain the outer --config patch.
The private config makes both invocations resolve the same harness source.
No user configuration or credentials are copied or changed.
"""
import os
from pathlib import Path

root = Path('/var/tmp/gmsd-chroma')
cargo_home = root/'cache/cargo'
assert Path(os.environ['CARGO_HOME']).resolve() == cargo_home
path = cargo_home/'config.toml'
expected = ('[patch.crates-io]\n'
            'zenbench = { path = "/var/tmp/gmsd-chroma/zenbench-snapshot" }\n')
if path.exists():
    assert path.read_text() == expected, 'unexpected lane Cargo config'
else:
    path.write_text(expected)
print('lane_cargo_patch', path)
