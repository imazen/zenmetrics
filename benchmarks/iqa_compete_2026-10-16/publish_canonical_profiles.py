#!/usr/bin/env python3
"""Export retained CPU profiles without treating samples as causal attribution."""
import argparse
import json
from pathlib import Path
import re

from compare_canonical import sha
from reference_publication import register_outputs


def publish(index, output):
    index = index.resolve(strict=True)
    jobs = json.loads(index.read_text())
    rows = []
    for job in jobs['profiles']:
        root = Path(job['output']).resolve(strict=True)
        receipt_path = root/'receipt.json'
        receipt = json.loads(receipt_path.read_text())
        for path, key in [(root/'perf.data', 'profile_sha256'),
                          (root/'report.txt', 'report_sha256'),
                          (Path(job['binary']), 'binary_sha256'),
                          (Path(job['config']), 'config_sha256')]:
            if sha(path) != receipt[key]:
                raise ValueError('retained profile binding changed: '+str(path))
        symbols = []
        for line in (root/'report.txt').read_text().splitlines():
            match = re.match(r'^\s+([\d.]+)%\s+(\S+)\s+(\S+)\s+\[([.k])\]\s+(.*)$', line)
            if match:
                symbols.append(dict(self_percent=float(match[1]), process=match[2],
                                    object=match[3], domain=match[4], symbol=match[5]))
        wrappers = [line for line in (root.parent/'profile-wrapper.log').read_text().splitlines()
                    if line.startswith('run-heavy: done ')]
        if not wrappers or any('rc=0 ' not in line for line in wrappers):
            raise ValueError('successful resource receipt required: '+str(root))
        rows.append(dict(case=root.parent.name, threads=job['threads'], raw_directory=str(root),
                         receipt=receipt, receipt_sha256=sha(receipt_path),
                         top_self_symbols=symbols[:12], wrapper_receipts=wrappers))
    # Validate everything before creating public output; never overwrite an export.
    output.mkdir()
    outputs = []
    shards = []
    for number, row in enumerate(rows):
        path = output/f'profile-{number:02d}.json'
        text = json.dumps(row, indent=2)+'\n'
        if len(text.encode()) > 30_000:
            raise ValueError('profile projection exceeds generated-data limit: '+row['case'])
        path.write_text(text)
        outputs.append((path, 'provenance'))
        shards.append(dict(file=path.name, sha256=sha(path), case=row['case']))
    path = output/'index.json'
    path.write_text(json.dumps(dict(source_index=str(index), source_index_sha256=sha(index),
        exporter_sha256=sha(Path(__file__)), profiles=shards,
        reference_note=jobs.get('reference_note'),
        qualification='CPU cycle samples after warmup, including calibration and measured rounds. '
        'No GPU-kernel profile, causal share of a timing gap, or qualified regression decision.'), indent=2)+'\n')
    outputs.append((path, 'provenance'))
    register_outputs(outputs)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('index', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(json.dumps(dict(profiles=publish(args.index, args.output), output=str(args.output))))


if __name__ == '__main__':
    main()
