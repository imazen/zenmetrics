#!/usr/bin/env python3
"""Inspect representative parquet schemas, including generic combined tables.

Only field names are accessed. Never read rows, statistics, labels or pixels.
Hash/number-named shard families contribute one representative per directory;
the report explicitly records how many filenames each schema represents.
"""
from collections import defaultdict
import json
from pathlib import Path
import re
import subprocess

import pyarrow.parquet as pq

ROOT = Path('/var/tmp/gmsd-chroma')


def main():
    files = subprocess.check_output(['rg','--files','--hidden','--no-ignore',
        '/mnt/v/output/zensim','/mnt/v/zen/zensim-training',
        '/mnt/v/output/zensim-multicodec-probe'],text=True).splitlines()
    exclude = re.compile(r'aic|cid22|holdout|sealed|terminal|(?:^|[/_.-])(?:eval|select|val|test)(?:[/_.-]|$)',re.I)
    groups = defaultdict(list)
    for filename in files:
        p = Path(filename)
        if p.suffix!='.parquet' or exclude.search(filename):
            continue
        stem = re.sub(r'[0-9a-fA-F]{16,}','HASH',p.stem)
        stem = re.sub(r'\d+','N',stem)
        groups[(str(p.parent),stem)].append(filename)
    out = ROOT/'peer_combined_schemas.json'
    assert not out.exists()
    records = []
    for (directory,stem),members in sorted(groups.items()):
        filename = min(members)
        names = pq.read_schema(filename).names
        peers = [s for s in names if re.search(r'ssim2|ssimulacra|peer',s,re.I)]
        record = dict(path=filename,family_pattern=stem,family_files=len(members),
                      columns=[s for s in names if not re.fullmatch(r'f\d+',s)],peer_columns=peers)
        records.append(record)
        interesting = [s for s in peers if s not in ['ssim2_gpu','ssim2_log_norm']]
        if interesting:
            print(json.dumps(dict(path=filename,peer_columns=interesting),sort_keys=True))
    out.write_text(json.dumps(dict(payload_columns_read=[],representative_schemas=records),indent=2)+'\n')
    print('schema_families',len(records))
    print('represented_files',sum(r['family_files'] for r in records))
    print('families_with_peer_columns',sum(bool(r['peer_columns']) for r in records))


if __name__=='__main__':
    main()
