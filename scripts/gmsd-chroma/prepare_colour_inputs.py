#!/usr/bin/env python3
"""Freeze the full reporting population from pixel-only TRAIN key tables."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path('/var/tmp/gmsd-chroma')
KEYS = Path('/var/tmp/gmsbank/peer_gmsd')
SOURCES = {
    'kadid_train': ('kadid_train', '0d573a4452d908c79c7128f62d3e2d3ddc02c9ef4a57c0ba732507b9dd00d398'),
    'tid_train': ('tid2013', 'dd75b9f0f0ab334a84af8fcdb1f3d6b8998ec79c8add0da1bf55688fe3a4b391'),
    'konfig_train': ('konfig_train', '2c0418bad69032dee36fe50dc15112817368e0e3ccf07930141a41394c6f5f76'),
}


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    import pyarrow.parquet as pq
    out = ROOT/'colour_v2'
    out.mkdir(exist_ok=True)
    assert not (out/'population.json').exists(), 'population already frozen'
    pairs = []
    stimuli = []
    metadata_hashes = {}
    for dataset, (source, digest) in SOURCES.items():
        path = KEYS/(source+'.keys.tsv')
        assert sha(path) == digest
        with path.open() as f:
            rows = list(csv.reader(f, delimiter='\t'))
        assert len({r[1] for r in rows}) == {'kadid_train':40, 'tid_train':25, 'konfig_train':3}[dataset]
        by_key = {r[0]: r for r in rows}
        metadata = Path('/var/tmp/rev4-featbank/bank')/source/'labels__human.parquet'
        metadata_hashes[dataset] = sha(metadata)
        # Column projection: never request/decode human_score. Identity
        # aliases retain separate stimulus identities and distortion kinds.
        records = pq.read_table(metadata, columns=['pair_key','source_row_id','codec','knob']).to_pylist()
        seen = set()
        for record in records:
            kind = record['codec'].removeprefix('kadid_').removeprefix('tid_')
            if dataset == 'kadid_train' and kind not in {'04','05','06','07','08'}:
                continue
            if dataset == 'tid_train' and kind not in {'02','18','22','23'}:
                continue
            if dataset == 'konfig_train' and kind != 'colordiffusion':
                continue
            key, rh, dh, w, h, rp, dp = by_key[record['pair_key']]
            assert hashlib.sha256((rh+dh+'legacy-rgb8').encode()).hexdigest() == key
            stimuli.append(dict(pair_key=key, stimulus_id=record['source_row_id'],
                                set=dataset, distortion=kind, knob=record['knob'],
                                ref_sha=rh, role='TRAIN'))
            if key in seen:
                continue
            seen.add(key)
            pairs.append(dict(pair_key=key, set=dataset, distortion=kind, role='TRAIN',
                              ref_sha=rh, dist_sha=dh, width=int(w), height=int(h),
                              ref_path=rp, dist_path=dp))
    assert len({(r['set'],r['pair_key']) for r in pairs}) == len(pairs)
    columns = ['ref_path','dist_path','image_path','codec','q','knob_tuple_json']
    with (out/'score_pairs.tsv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter='\t', lineterminator='\n')
        writer.writeheader()
        for i,r in enumerate(pairs):
            writer.writerow(dict(ref_path=r['ref_path'], dist_path=r['dist_path'],
                                 image_path=r['ref_path'], codec=r['set']+':'+r['distortion'], q=i,
                                 knob_tuple_json=json.dumps(dict(pair_key=r['pair_key']),sort_keys=True)))
    counts = Counter(r['set']+':'+r['distortion'] for r in stimuli)
    assert len({(r['set'],r['stimulus_id']) for r in stimuli}) == len(stimuli)
    manifest = dict(schema='gmsd-chroma-colour-population-v2', source_key_sha256=SOURCES,
                    source_metadata_sha256=metadata_hashes,
                    metadata_columns_read=['pair_key','source_row_id','codec','knob'],
                    counts=dict(counts), pairs=pairs, stimuli=stimuli,
                    score_pairs_sha256=sha(out/'score_pairs.tsv'),
                    human_labels_read=False,
                    tid_role_authority='DATA_SPLITS.md section 8.1, all 25 refs TRAIN-only')
    (out/'population.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('reporting_pairs',len(pairs))
    print('reporting_stimuli',len(stimuli))
    print('counts',json.dumps(dict(counts),sort_keys=True))
    print('population_sha256',sha(out/'population.json'))
    print('score_pairs_sha256',sha(out/'score_pairs.tsv'))


if __name__ == '__main__':
    main()
