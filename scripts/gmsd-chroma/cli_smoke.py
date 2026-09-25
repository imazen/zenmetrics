#!/usr/bin/env python3
"""Check score/score-pairs against canonical raw RGB8 on one TRAIN pair/set."""
import csv
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess

ROOT = Path('/var/tmp/gmsd-chroma')
ARMS = {'gmsd':('gmsd','gmsd_cpu_imazen_v0_1_0'),
        'mdsi':('mdsi','mdsi_cpu_imazen_v0_1_0'),
        'ms-gmsdc':('ms_gmsdc','ms_gmsdc_paper_cpu_imazen_v0_1_0')}


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def bits(value):
    return struct.pack('<d',float(value))


def main():
    import pyarrow.parquet as pq
    subprocess.run(['python3','scripts/gmsd-chroma/preflight.py'],check=True)
    subprocess.run(['bash','scripts/gmsd-chroma/finish_cli_checks.sh'],check=True)
    subprocess.run(['python3','scripts/gmsd-chroma/build_examples.py'],check=True)
    subprocess.run(['python3','scripts/gmsd-chroma/verify_build.py','colour_score'],check=True)
    out = ROOT/'cli_smoke'
    out.mkdir(exist_ok=False)
    population = ROOT/'colour_v2/population.json'
    assert sha(population) == '13db0836bff1966a8593b89907752e3c7bdc9b9fc26116039280f33aadf91940'
    rows = json.loads(population.read_text())['pairs']
    selected = [next(r for r in rows if r['set']==s)
                for s in ['kadid_train','tid_train','konfig_train']]
    assert all(r['role']=='TRAIN' for r in selected)
    with (ROOT/'colour_v2/raw_pairs.tsv').open() as f:
        raw = {r['pair_key']:r for r in csv.DictReader(f,delimiter='\t')}
    raw_tsv = out/'raw_pairs.tsv'
    with raw_tsv.open('w') as f:
        writer = csv.DictWriter(f,fieldnames=['pair_key','width','height','ref_rgb','dist_rgb'],
                                delimiter='\t',lineterminator='\n')
        writer.writeheader()
        writer.writerows(raw[r['pair_key']] for r in selected)
    pairs = out/'pairs.tsv'
    with pairs.open('w') as f:
        writer = csv.DictWriter(f,fieldnames=['ref_path','dist_path','knob_tuple_json'],
                                delimiter='\t',lineterminator='\n')
        writer.writeheader()
        for r in selected:
            writer.writerow(dict(ref_path=r['ref_path'],dist_path=r['dist_path'],
                                 knob_tuple_json=json.dumps(dict(pair_key=r['pair_key']))))
    cli = ROOT/'target/release/zenmetrics'
    raw_binary = ROOT/'target/release/examples/colour_score'
    env = {**os.environ,'RAYON_NUM_THREADS':'8'}
    commands = []

    def run(argv):
        result = subprocess.run([str(a) for a in argv],env=env,capture_output=True,text=True)
        record = dict(argv=[str(a) for a in argv],exit_code=result.returncode,
                      stdout=result.stdout,stderr=result.stderr)
        commands.append(record)
        with (out/'commands.jsonl').open('a') as f:
            f.write(json.dumps(record)+'\n')
        result.check_returncode()
        return result.stdout

    reference = out/'raw_scores.tsv'
    run([raw_binary,raw_tsv,reference])
    with reference.open() as f:
        expected = {r['pair_key']:r for r in csv.DictReader(f,delimiter='\t')}
    checked = []
    cli_digest = sha(cli)
    for arm,(raw_column,column) in ARMS.items():
        sidecar = out/(arm+'.parquet')
        run([cli,'score-pairs','--metric',arm,'--pairs-tsv',pairs,'--out-parquet',sidecar])
        values = pq.read_table(sidecar,columns=['knob_tuple_json',column]).to_pylist()
        indexed = {json.loads(r['knob_tuple_json'])['pair_key']:r[column] for r in values}
        assert len(indexed)==len(values)==len(selected)
        assert indexed.keys()==expected.keys()
        for r in selected:
            key = r['pair_key']
            output = json.loads(run([cli,'score','--metric',arm,'--reference',r['ref_path'],
                                     '--distorted',r['dist_path'],'--output','json']))
            assert output['metric']==arm and set(output['scores'])=={column}
            want = bits(expected[key][raw_column])
            assert bits(indexed[key])==want, ('score-pairs pixels or metric mismatch',arm,key)
            assert bits(output['scores'][column])==want, ('score pixels or metric mismatch',arm,key)
            checked.append(dict(arm=arm,set=r['set'],pair_key=key,bitwise_equal=True))
    assert sha(cli)==cli_digest
    report = dict(passed=True,cli_binary_sha256=cli_digest,raw_binary_sha256=sha(raw_binary),
                  cases=checked,checks=len(checked)*2,population_sha256=sha(population),
                  clippy=json.loads((ROOT/'cli_clippy_status.json').read_text()))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,sort_keys=True))


if __name__=='__main__':
    main()
