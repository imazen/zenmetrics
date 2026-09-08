#!/usr/bin/env python3
"""Check a fleet tar against its declared comparison job, without re-encoding."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import tarfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify(path, job):
    settings = json.loads(job['kind']['knobs'])
    assert len(settings['max_edges']) == 1, 'checker currently handles one-size scout jobs'
    defaults = dict(bit_depth=8, chroma='420', threads=1, tune=None, scm=None,
                    sb128=False, svt_reference=None, zen_intra_edge_filter=False,
                    zen_restoration_unit_search=False)

    def arm_key(config):
        return json.dumps({k: v for k, v in config.items() if k not in ('width', 'height')},
                          sort_keys=True)

    expected = {arm_key(dict(defaults, **arm)) for arm in settings['arms']}
    assert len(expected) == len(settings['arms']), 'duplicate declared arms'
    with tarfile.open(path) as archive:
        names = archive.getnames()
        assert len(names) == len(set(names)), 'duplicate archive members'

        def read(name):
            return archive.extractfile('comparison/'+name).read()

        validation = json.loads(read('validation.json'))
        assert validation['complete'] is True
        rows = [json.loads(line) for line in read('rows.jsonl').splitlines()]
        witnesses = [json.loads(line) for line in read('reconstruction-verification.jsonl').splitlines()]
        cohort = sha(read('timing-environment.json'))
        observed = collections.defaultdict(list)
        svt = set()
        sizes = set()
        checked_payloads = set()

        def witness_key(config, inp, out):
            return (json.dumps(config, sort_keys=True), inp, out)

        for row in rows:
            cfg = row['config']
            key = arm_key(cfg)
            assert key in expected, 'unexpected configuration'
            assert row['source_sha256'] == job['inputs'][0], 'wrong source'
            assert row['binary_sha256'] == settings['binary_sha256'], 'wrong executor'
            assert row['timing_cohort_sha256'] == cohort, 'wrong timing cohort'
            assert row['api_elapsed_ns'] > 0
            assert math.isfinite(row['ssimulacra2'])
            sizes.add((cfg['width'], cfg['height']))
            assert max(cfg['width'], cfg['height']) <= settings['max_edges'][0]
            payload_key = row['output_sha256']
            if payload_key not in checked_payloads:
                payload = read(f'obu/{payload_key}.obu')
                assert sha(payload) == payload_key and len(payload) == row['bytes']
                checked_payloads.add(payload_key)
            observed[key].append(row)
            if cfg['backend'] == 'zenav1-svt':
                svt.add(witness_key(cfg, row['input_sha256'], row['output_sha256']))
        assert len(sizes) == 1 and set(observed) == expected
        width, height = next(iter(sizes))
        reference = read(f"{job['inputs'][0]}-{width}x{height}-reference.png")
        assert reference[:8] == b'\x89PNG\r\n\x1a\n'
        assert (int.from_bytes(reference[16:20]), int.from_bytes(reference[20:24])) == (width, height)
        for group in observed.values():
            assert sorted(r['round'] for r in group) == list(range(settings['repeats']))
            assert len({(r['input_sha256'], r['output_sha256'], r['bytes']) for r in group}) == 1
        verified = set()
        for witness in witnesses:
            assert witness['ok'] is True and witness['error'] is None
            assert witness['source_sha256'] == job['inputs'][0]
            assert witness['measured_binary_sha256'] == settings['binary_sha256']
            assert witness['verifier_binary_sha256'] == settings['binary_sha256']
            assert witness['decoder'] == 'libaom'
            verified.add(witness_key(witness['config'], witness['input_sha256'], witness['output_sha256']))
        assert verified == svt and len(witnesses) == len(verified)
        assert validation['verified_svt_cells'] == len(svt)
        assert validation['svt_reconstruction_required'] == bool(svt)
        return dict(timed_rows=len(rows), cells=len(observed), verified_svt_cells=len(svt),
                    unique_obus=len(checked_payloads), timing_cohort_sha256=cohort,
                    source_sha256=job['inputs'][0], executor_sha256=settings['binary_sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifact', type=Path)
    parser.add_argument('job', type=Path, help='DesiredJob JSON or one-job manifest array')
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    if isinstance(job, list):
        assert len(job) == 1
        job = job[0]
    print(json.dumps(verify(args.artifact, job), sort_keys=True))


if __name__ == '__main__':
    main()
