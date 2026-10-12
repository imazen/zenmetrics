#!/usr/bin/env python3
"""Native bundle materialization control, without workload execution or upload."""
import json
from pathlib import Path
import shutil
import sys

import build_benchmark_study as study
from test_benchmark_study import StudyTests

runner, output = (Path(p).resolve() for p in sys.argv[1:])
output.mkdir(parents=True, exist_ok=False)
fixture = StudyTests()
fixture.setUp()
try:
    fixture.qualify()
    spec = fixture.write()
    # Keep the fixture and its bound source paths after this control finishes.
    root = output / 'sources'
    shutil.copytree(fixture.root, root)
    spec = root / spec.name
    generated = output / 'generated'
    study.write_study(spec, generated)
    result = study.package_study(spec, generated, runner, study.file_sha(runner), output / 'bundles')
    assert result['bundles'] == 12 and not result['executed'] and not result['staged']
    observations = study.load(Path(result['observations']))['observations']
    assert len(observations) == 12
    assert {row['job']['replicate'] for row in observations} == {0, 1, 2}
    assert all(row['job']['runner_sha'] == study.file_sha(runner) for row in observations)
    identities = study.load(output / 'bundles' / 'identities.json')
    for row in identities['cells']:
        assert study.file_sha(Path(row['bundle']) / 'bundle.json') == row['bundle_sha']
    print(json.dumps(dict(native_bundle_control='passed', bundles=12,
                         workload='fixture bytes never executed', uploaded=False)))
finally:
    fixture.doCleanups()
