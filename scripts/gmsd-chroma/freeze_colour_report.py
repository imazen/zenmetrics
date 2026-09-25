#!/usr/bin/env python3
"""Join frozen TRAIN predictions to labels only after both numeric gates pass.

The peer table must be normalized by a separately documented source join:
pair_key,fast_ssim2. Its manifest pins that table, the source files and the
actual fast-ssim2 CPU producer. GPU columns cannot silently enter this arm.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path('/var/tmp/gmsd-chroma')
POPULATION_SHA = '13db0836bff1966a8593b89907752e3c7bdc9b9fc26116039280f33aadf91940'
SOURCES = dict(kadid_train='kadid_train',tid_train='tid2013',konfig_train='konfig_train')


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def table(path):
    with path.open() as f:
        rows = list(csv.DictReader(f,delimiter='\t'))
    result = {r['pair_key']:r for r in rows}
    assert len(result) == len(rows), ('duplicate pair key',str(path))
    return result


def main():
    import pyarrow.parquet as pq
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--predictions',type=Path,required=True)
    ap.add_argument('--predictions-manifest',type=Path,required=True)
    ap.add_argument('--peers',type=Path,required=True)
    ap.add_argument('--peer-manifest',type=Path,required=True)
    ap.add_argument('--panel',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args = ap.parse_args()
    assert ROOT in args.output.resolve().parents
    manifest_path = args.output.with_suffix('.manifest.json')
    assert not args.output.exists() and not manifest_path.exists()
    for name in ['parity.json','ms_parity.json']:
        assert json.loads((ROOT/name).read_text())['passed'] is True
    population = ROOT/'colour_v2/population.json'
    assert sha(population) == POPULATION_SHA
    pop = json.loads(population.read_text())
    assert pop['human_labels_read'] is False
    predictions_manifest = json.loads(args.predictions_manifest.read_text())
    assert predictions_manifest['table_sha256'] == sha(args.predictions)
    assert predictions_manifest['population_sha256'] == POPULATION_SHA
    assert predictions_manifest['implementation_commit']
    assert predictions_manifest['binary_sha256']
    assert predictions_manifest['source_sha256']
    peer_manifest = json.loads(args.peer_manifest.read_text())
    assert peer_manifest['table_sha256'] == sha(args.peers)
    assert peer_manifest['producer'] == 'fast-ssim2-cpu'
    assert peer_manifest['source_sha256'] and peer_manifest['join_description']
    predictions, peers = table(args.predictions),table(args.peers)
    keys = {r['pair_key'] for r in pop['pairs']}
    assert predictions.keys() == peers.keys() == keys
    for key in keys:
        for arm in ['gmsd','mdsi','ms_gmsdc']:
            assert math.isfinite(float(predictions[key][arm]))
        assert math.isfinite(float(peers[key]['fast_ssim2']))
    # All rows, signs, predictors and gates are fixed before this label read.
    labels = {}
    for dataset,source in SOURCES.items():
        admitted = [r for r in pop['stimuli'] if r['set'] == dataset]
        assert all(r['role'] == 'TRAIN' for r in admitted)
        path = Path('/var/tmp/rev4-featbank/bank')/source/'labels__human.parquet'
        assert sha(path) == pop['source_metadata_sha256'][dataset]
        values = pq.read_table(path,columns=['pair_key','source_row_id','human_score'],
                               filters=[('source_row_id','in',[r['stimulus_id'] for r in admitted])]).to_pylist()
        indexed = {r['source_row_id']:r for r in values}
        assert len(indexed) == len(values) == len(admitted)
        for stimulus in admitted:
            record = indexed[stimulus['stimulus_id']]
            assert record['pair_key'] == stimulus['pair_key']
            value = float(record['human_score'])
            assert math.isfinite(value)
            labels[(dataset,stimulus['stimulus_id'])] = value
    fields = ['pair_key','stimulus_id','ref_sha','set','distortion','role','quality',
              'gmsd','mdsi','ms_gmsdc','fast_ssim2']
    with args.output.open('x') as f:
        writer = csv.DictWriter(f,fieldnames=fields,delimiter='\t',lineterminator='\n')
        writer.writeheader()
        for stimulus in pop['stimuli']:
            key = stimulus['pair_key']
            row = {k:stimulus[k] for k in fields[:6]}
            row['quality'] = labels[(stimulus['set'],stimulus['stimulus_id'])]
            row.update({k:predictions[key][k] for k in ['gmsd','mdsi','ms_gmsdc']})
            row['fast_ssim2'] = peers[key]['fast_ssim2']
            writer.writerow(row)
    manifest = dict(table_sha256=sha(args.output),panel_sha256=sha(args.panel),
                    population_sha256=POPULATION_SHA,
                    implementation_frozen_before_labels=True,
                    population_committed_before_labels=True,
                    predictions_manifest_sha256=sha(args.predictions_manifest),
                    peer_manifest_sha256=sha(args.peer_manifest),
                    source_labels_sha256=pop['source_metadata_sha256'],
                    rows=len(pop['stimuli']),roles=['TRAIN'],
                    quality_orientation='stored bank human_score: higher is better')
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,sort_keys=True))


if __name__ == '__main__':
    main()
