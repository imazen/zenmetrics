#!/usr/bin/env python3
"""Exercise linked-root admission using only an archive and its actual worker image.

Run under shared run-heavy admission: extraction and checksumming touch GBs.
No training, target interpretation, claims or external network access occurs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


CHECK = r'''
import json,os,sys
from pathlib import Path
sys.path.insert(0,'/opt/fleet-fits/program')
import fit_cell_exec as fit
from byc_fit_contract import environment
pin,expected_root,expected_link,label=sys.argv[1:]
data=fit.fetch_data(pin)
fit.bind_expected_paths(data)
argv=['byc_cells.py','run','--arm','mono:thr:mp32','--label',label,'--folds','mos7','--seeds','0','--jobs','1']
os.environ.update(environment(argv,data,fit.PROGRAM,fit.FIT_ROOT,fit.ensure_link))
sys.path.insert(0,str(fit.PROGRAM/'scripts/rev4_featpot'))
from v2_common import load_frozen,admission_input_roots,table_path,table_input_roots
from subjective_training_leg import dataset_groups,sha,linked_subset_exclusions
from byc_cells import dataset_holdout_collections
root=Path(os.environ['BYC_ROOT'])
assert sha(root/'wide/frozen.json')==expected_root
assert sha(root/'datasets/linked-subset-omissions.json')==expected_link
load_frozen(root,training_only=True,metadata_only=True)
admission_input_roots(root)
inventory=json.loads((data/'input_inventory.json').read_text())
def member(path,pin=None):
    resolved=Path(path).resolve()
    rel=str(resolved.relative_to(data.resolve()))
    if rel not in inventory['files'] or (pin is not None and inventory['files'][rel]!=pin):
        raise ValueError('runtime read is outside its verified inventory: '+str(path))
    return rel
packet=json.loads((root/'datasets/receipt.json').read_text())
reads=[member(packet['family_guard_path'],packet['family_guard_sha256'])]
for group in packet['groups']:
    reads.append(member(group['path'],group['table_sha256']))
    reads.append(member(group['path']+'.manifest.json',group['manifest_sha256']))
    if group.get('keys_path'):
        reads.append(member(group['keys_path'],group.get('keys_sha256')))
wide=json.loads((root/'wide/main/real/receipt.json').read_text())
for leg in wide['legs'].values():
    for part in ('full','fit','dev'):
        if part in leg:
            path=table_path(leg[part])
            reads.append(member(path,leg[part]['sha256']))
            reads.append(member(str(path)+'.manifest.json',leg[part]['manifest_sha256']))
            table_input_roots(path)
selected=[int(v) for v in (fit.PROGRAM/'byc_columns.txt').read_text().strip().split(',')]
out=Path('/scratch/relocation-admission');out.mkdir()
rows={}
for fold in ('kadid','tid2013','konfig','cid22_a25','pipal','mos7','pairs'):
    _,excluded=dataset_holdout_collections(root,fold)
    dest=out/fold;dest.mkdir()
    groups,authority=dataset_groups(root,[],selected,fold if fold in ('kadid','tid2013','konfig','cid22_a25') else 'full',0,1101,101,dest,exclude=excluded)
    receipt=json.loads((dest/'DATASET_GROUP_SELECTION.json').read_text())
    rows[fold]=dict(excluded=list(excluded),linked_exclusions=list(linked_subset_exclusions(root,fold)),retained_collections=receipt['retained_collections'],groups=len(groups),authority_sha256=authority[1])
assert rows['mos7']['linked_exclusions']==['colourlab-cognitive-science-2022']
assert all(not row['linked_exclusions'] for fold,row in rows.items() if fold!='mos7')
assert rows['mos7']['groups']==8
assert not list(out.rglob('*.bin'))
meta=json.loads((fit.PROGRAM/'build_meta.json').read_text())
record=dict(schema='byc-linked-archive-relocation-v1',image_program_source=meta['source_commit'],data_sha256=pin,root_sha256=expected_root,linked_sha256=expected_link,bindings=inventory['byc']['bindings'],verified_archive_files=len(inventory['files']),runtime_reads_checked=len(reads),folds=rows,target_payloads_interpreted=False,optimizer_invoked=False,network='none')
(out/'VERIFIED.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--image', required=True)
    p.add_argument('--scratch', type=Path, required=True)
    p.add_argument('--root-sha', required=True)
    p.add_argument('--linked-sha', required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists() or args.scratch.exists():
        raise ValueError('use fresh scratch and receipt paths')
    h = hashlib.sha256()
    with args.data.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    pin = h.hexdigest()
    cache = args.scratch/'fit-cell'
    cache.mkdir(parents=True)
    (args.scratch/'tmp').mkdir()
    os.link(args.data, cache/(pin+'.tar.gz'))
    record = json.loads(subprocess.check_output([
        'docker','run','--rm','--network','none','--memory','2g','--memory-swap','2g','--cpus','1',
        '-e','TMPDIR=/scratch/tmp','-e','PYTHONDONTWRITEBYTECODE=1','-e','OPENBLAS_NUM_THREADS=1',
        '-v',str(args.scratch.resolve())+':/scratch','--entrypoint','python3',args.image,
        '-c',CHECK,pin,args.root_sha,args.linked_sha,args.label], text=True))
    record.update(image=args.image, scratch=str(args.scratch.resolve()))
    with args.out.open('x') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
