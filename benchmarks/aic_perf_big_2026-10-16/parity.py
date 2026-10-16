"""Compare complete Butteraugli maps and CAMBI score bits on the AIC panel."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('scratch', type=Path)
    args = parser.parse_args()
    args.scratch.mkdir(parents=True, exist_ok=True)
    rows = []
    with (Path.home() / 'tmp/aic-audit/pairs.tsv').open() as source:
        pairs = list(csv.DictReader(source, delimiter='\t'))
    assert len(pairs) == 61, len(pairs)
    for pair in pairs:
        ref = Image.open(pair['ref_path']).convert('RGB')
        dis = Image.open(pair['dist_path']).convert('RGB')
        assert ref.size == dis.size
        w, h = ref.size
        r, d = args.scratch / 'r.rgb', args.scratch / 'd.rgb'
        r.write_bytes(ref.tobytes())
        d.write_bytes(dis.tobytes())
        for metric in ['butter', 'cambi']:
            outputs = []
            for label, directory in [('base', args.base), ('candidate', args.candidate)]:
                binary = directory / f'{metric}-{label}'
                if metric == 'butter':
                    path = args.scratch / f'{label}.map'
                    command = [binary, str(w), str(h), 'pair', r, d, path]
                else:
                    command = [binary, str(w), str(h), 'pair', d]
                result = subprocess.check_output(command, text=True).strip()
                digest = hashlib.sha256(path.read_bytes()).hexdigest() if metric == 'butter' else ''
                outputs.append((result, digest))
            assert outputs[0] == outputs[1], (pair['distorted'], metric, outputs)
            row = dict(key=pair['distorted'], metric=metric, score_bits=outputs[0][0], map_sha256=outputs[0][1])
            rows.append(row)
            print(json.dumps(row), flush=True)
    with args.output.open('x', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
