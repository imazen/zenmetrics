"""Round-10 frozen-binary comparisons. Caller serializes via run-heavy/flock."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys

stage, binaries, models, ladder, output = sys.argv[1:]
binaries, models, ladder, output = map(Path, (binaries, models, ladder, output))
output.mkdir(parents=True, exist_ok=True)
metrics = 'lpips_alex lpips_vgg lpips_squeeze stlpips_alex stlpips_vgg dists deepdc ckdn afine dreamsim arniqa maniqa liqe'.split()


def run(variant, model, threads, sizes, modes, budget, label, rss=False):
    dest = output / (label + '.tsv')
    if dest.exists():
        raise FileExistsError(dest)
    env = dict(os.environ, RTEN_NUM_THREADS=str(threads), ZENMETRICS_NN_MODEL=model,
               ZENMETRICS_NN_ENGINE='rten', ZENMETRICS_NN_BENCH_MODES=modes)
    command = [str(binaries / ('nn_bench_' + variant)), str(models), str(ladder),
               str(dest), '1415', sizes, str(budget)]
    if rss:
        command = ['/usr/bin/time', '-v', '-o', str(output / (label + '.time'))] + command
    print('RUN', label, flush=True)
    load_before = os.getloadavg()
    subprocess.run(command, env=env, check=True)
    load_after = os.getloadavg()
    rows = list(csv.DictReader(dest.open(), delimiter='\t'))
    if stage == 'perf':
        for row in rows:
            for index, minutes in enumerate((1, 5, 15)):
                row[f'load_{minutes}m_before'] = f'{load_before[index]:.2f}'
                row[f'load_{minutes}m_after'] = f'{load_after[index]:.2f}'
        with dest.open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter='\t')
            writer.writeheader()
            writer.writerows(rows)
    return rows


if stage == 'identity':
    rows = {}
    for variant in ('base', 'opt2', 'fixed'):
        rows[variant] = {}
        for model in metrics:
            for row in run(variant, model, 16, '256', 'compute,warm', 0, f'bits_{variant}_{model}'):
                value = float(row['score'])
                if not math.isfinite(value):
                    raise AssertionError(row)
                rows[variant][(model, row['mode'])] = struct.pack('!d', value).hex()
    with (output / 'bit_identity.tsv').open('x') as stream:
        writer = csv.writer(stream, delimiter='\t')
        writer.writerow(['metric', 'mode', 'base_bits', 'opt2_bits', 'fixed_bits', 'identical'])
        for key in rows['base']:
            bits = [rows[v][key] for v in ('base', 'opt2', 'fixed')]
            writer.writerow([*key, *bits, len(set(bits)) == 1])
    assert rows['base'] == rows['opt2'] == rows['fixed'], 'full-precision score mismatch'
elif stage == 'perf':
    # Keep each A/B pair adjacent at one size, rather than separating the
    # binaries by a complete six-size sweep on a shared host.
    for threads in (1, 16):
        for size in (64, 128, 256, 512, 1024, 2048):
            for round_ in ('a', 'b'):
                for variant in ('base', 'fixed'):
                    run(variant, 'ckdn', threads, str(size), 'compute,warm', 2,
                        f'review_ckdn_{variant}_{threads}t_n{size}_r{round_}')
elif stage == 'rss':
    for model in ('arniqa', 'afine'):
        for round_ in ('a', 'b'):
            for variant in ('base', 'fixed'):
                run(variant, model, 16, '2048', 'compute', 0,
                    f'review_rss_{model}_{variant}_r{round_}', rss=True)
else:
    raise ValueError(stage)
metadata = {'stage': stage, 'models': str(models), 'ladder': str(ladder),
            'binaries': {v: hashlib.sha256((binaries / ('nn_bench_' + v)).read_bytes()).hexdigest()
                         for v in ('base', 'opt2', 'fixed')}}
(output / (stage + '.json')).write_text(json.dumps(metadata, indent=2) + '\n')
