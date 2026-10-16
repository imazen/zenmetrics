#!/usr/bin/env python3
"""Install only the validated complete calibration and input receipts."""
import csv
import hashlib
import json
from pathlib import Path


def main():
    root = Path.home() / 'tmp/pieapp-port'
    summary = json.loads((root / 'konjnd-summary.json').read_text())
    assert summary['count'] == 1008 and summary['jnd'] > 0
    assert sorted(v['count'] for v in summary['codecs'].values()) == [504, 504]
    assert hashlib.sha256((root / 'konjnd.tsv').read_bytes()).hexdigest() == summary['scores_sha256']
    with (root / 'konjnd.tsv').open() as f:
        scores = [float(r['pieapp']) for r in csv.DictReader(f, delimiter='\t')]
    assert len(scores) == 1008
    assert __import__('statistics').median(scores) == summary['jnd']
    table = Path('crates/zenmetrics-nn/models/jnd_scale.tsv')
    text = table.read_text()
    row = 'pieapp\t' + format(summary['jnd'], '.17e') + '\tPooled median raw error over 1008 KonJND pairs (ort-cuda, 2026-10-16); benchmarks/pieapp_2026-10-16_konjnd.json'
    existing = [line for line in text.splitlines() if line.startswith('pieapp\t')]
    if existing:
        assert existing == [row], 'refuse to replace a different calibration'
    else:
        table.write_text(text.rstrip('\n') + '\n' + row + '\n')
    receipt = Path('benchmarks/pieapp_2026-10-16_konjnd.json')
    payload = json.dumps(summary, indent=2) + '\n'
    if receipt.exists():
        assert receipt.read_text() == payload, 'preserve previous generated receipt'
    else:
        receipt.write_text(payload)
    print(payload)
    with (root / 'ladder-pairs.tsv').open() as f:
        ladder_pairs = list(csv.DictReader(f, delimiter='\t'))
    assert len(ladder_pairs) == 24
    inputs = sorted({path for row in ladder_pairs for path in (row['ref_path'], row['dist_path'])})
    assert len(inputs) == 48
    manifest = 'file\tsha256\n'
    for path in inputs:
        digest = hashlib.sha256()
        with open(path, 'rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                digest.update(block)
        manifest += Path(path).name + '\t' + digest.hexdigest() + '\n'
    manifest_path = Path('benchmarks/pieapp_2026-10-16_inputs.tsv')
    if manifest_path.exists():
        assert manifest_path.read_text() == manifest, 'preserve previous image manifest'
    else:
        manifest_path.write_text(manifest)
    print('ladder input manifest SHA-256', hashlib.sha256(manifest.encode()).hexdigest())


if __name__ == "__main__":
    main()
