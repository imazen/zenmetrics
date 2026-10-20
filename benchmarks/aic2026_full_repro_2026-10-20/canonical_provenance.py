#!/usr/bin/env python3
"""Bind crop exports to their actual reference calls and retained attempts."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from canonical_compare import backup
from canonical_run import sha


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def checked_document(path, methods=False):
    values = json.loads(path.read_text())
    bindings = ([(entry['source'], entry['source_sha256']) for entry in values.values()]
                if methods else values.items())
    for source, expected in bindings:
        if sha(source) != expected:
            raise ValueError('supplementary reference source changed: '+source)
    return dict(path=str(path), sha256=sha(path), entries=values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('runtime_versions', type=Path)
    parser.add_argument('--source-bindings', type=Path)
    parser.add_argument('--reference-methods', type=Path)
    parser.add_argument('--input-bindings', type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    config = json.loads(Path(manifest['config']).read_text())
    root = Path(manifest['scratch'])
    output = Path(manifest['output']).parent
    versions = json.loads(args.runtime_versions.read_text())
    statuses = json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {}
    report = dict(schema='aic-canonical-crops-v1',
                  recorded_at=datetime.now(timezone.utc).isoformat(),
                  cohort_rows=manifest['cohort_rows'],
                  input_manifest=dict(path=str(args.manifest), sha256=sha(args.manifest)),
                  runtime_metadata=dict(path=str(args.runtime_versions), sha256=sha(args.runtime_versions)),
                  native_source=manifest['source_commit'],
                  binary=dict(path=manifest['binary'], sha256=manifest['binary_sha256']),
                  batch_precision=manifest['batch_precision'],
                  driver_source=manifest['driver_source'],
                  report_writer=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)),
                  bindings=manifest['bindings'],
                  metrics={}, addendum={}, exports={})
    report['published_reference_plan'] = manifest.get('published_comparison')
    published = root/'fsim-vsi-reference.json'
    report['published_comparisons'] = json.loads(published.read_text()) if published.exists() else None
    report['supplementary_source_bindings'] = (checked_document(args.source_bindings)
                                              if args.source_bindings else None)
    report['supplementary_reference_methods'] = (checked_document(args.reference_methods, methods=True)
                                                if args.reference_methods else None)
    report['input_bindings'] = checked_document(args.input_bindings) if args.input_bindings else None
    for group in ('metrics', 'addendum'):
        for name, spec in manifest[group].items():
            entry = dict(spec)
            entry['installed_runtime_metadata'] = versions.get(spec.get('python', 'python3'))
            entry['adapter'] = (config.get('resident_adapters', {}).get(name)
                                or config.get('octave', {}).get(name))
            if name in ('brisque', 'niqe', 'piqe'):
                details = {
                    'brisque': ('PIQ 0.8.0', 'piq.brisque(x, data_range=255., reduction="none")',
                                'float32 RGB code values'),
                    'niqe': ('BasicSR 1.4.2', 'calculate_niqe(BGR8, crop_border=0)',
                             'BGR8 input; unchanged BasicSR internal conversion'),
                    'piqe': ('pypiqe 1.2', 'piqe(BGR8)[0]',
                             'BGR8 input; unchanged pypiqe internal conversion'),
                }
                entry['package'], entry['call'], entry['dtype'] = details[name]
            if name.startswith('pyiqa-'):
                entry.update(device='cpu', dtype='float32 RGB / 255',
                             call="pyiqa.create_metric(name, device='cpu', **options)(distorted, reference)",
                             options=dict(name='vsi' if name == 'pyiqa-vsi' else 'fsim',
                                          chromatic=False if name == 'pyiqa-fsim' else
                                          ('default True' if name == 'pyiqa-fsimc' else 'not applicable')))
            if report['supplementary_reference_methods']:
                entry.update(report['supplementary_reference_methods']['entries'].get(name, {}))
            entry['attempts'] = {}
            for role in (('canonical',) if group == 'addendum' else ('native', 'canonical')):
                path = root/'logs'/f'{name}.{role}.receipt.json'
                if path.exists():
                    receipt = json.loads(path.read_text())
                    result = Path(receipt['result'])
                    if receipt['result_sha256'] is not None and sha(result) != receipt['result_sha256']:
                        raise ValueError('attempt output changed: '+str(result))
                    entry['attempts'][role] = receipt
                else:
                    entry['attempts'][role] = dict(status='pending', receipt=str(path))
            if name in statuses:
                entry['coverage'] = statuses[name]
            report[group][name] = entry
    for name in ('metrics_cropped.canonical.csv', 'metrics_cropped.zenmetrics-extra.csv',
                 'metrics_cropped.pyiqa-fsim-vsi.csv'):
        path = output/name
        report['exports'][name] = dict(sha256=sha(path), bytes=path.stat().st_size) if path.exists() else None
    path = output/'provenance.canonical.json'
    backup(path)
    path.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False)+'\n')
    print(path)


if __name__ == '__main__':
    main()
