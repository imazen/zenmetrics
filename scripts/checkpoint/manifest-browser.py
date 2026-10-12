#!/usr/bin/env python3
"""Bind the qualified browser package to source, compiler and actual worker controls."""
import json
from pathlib import Path
import subprocess
import sys
from package import artifact_hashes

package, source, report = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
if len(source) != 40 or any(c not in '0123456789abcdef' for c in source):
    raise ValueError('Full source commit required')
controls = json.loads(report.read_text())
for key in ['native_oracle_exact', 'repeated', 'alpha_refused', 'rejected_input_recovery', 'isolation_refused']:
    if controls.get(key) is not True:
        raise ValueError(f'Missing browser qualification: {key}')
if controls['ready'].get('type') != 'ready' or controls['ready'].get('threads') != 2:
    raise ValueError('Two real Rayon workers required')
if not isinstance(controls['ready'].get('metrics'), list) or 'zensim' not in controls['ready']['metrics']:
    raise ValueError('Worker must report its actual selectable metrics')
metadata = {'schema': 1, 'source_commit': source,
    'target': 'wasm32-unknown-unknown', 'profile': 'browser-simd128-relaxed-rayon',
    'rustc': subprocess.check_output(['rustc', '+nightly-2026-10-06', '-Vv'], text=True),
    'wasm_bindgen': '0.2.129', 'required': ['secure-context', 'COOP: same-origin',
        'COEP: require-corp', 'SharedArrayBuffer', 'simd128', 'relaxed-simd'],
    'metrics': controls['ready']['metrics'],
    'cached_profile': 'zensim/codec_target',
    'native_spatial_exports': ['zensim', 'butteraugli', 'dssim', 'cvvdp', 'psnr', 'gmsd'],
    'qualification': {k: controls[k] for k in ['native_oracle_exact', 'repeated',
        'alpha_refused', 'rejected_input_recovery', 'isolation_refused', 'browser', 'ready']},
    'files': artifact_hashes(package)}
(package / 'checkpoint.json').write_text(json.dumps(metadata, indent=2) + '\n')
