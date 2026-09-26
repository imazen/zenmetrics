#!/usr/bin/env python3
"""Read parquet field names only in likely TRAIN corpus paths.

Never read row groups, values, column statistics, human labels or features.
Mixed-corpus metadata is not sufficient to admit its values later.
"""
import json
import re
import subprocess
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path('/var/tmp/gmsd-chroma')


def main():
    files = subprocess.check_output([
        'rg','--files','--hidden','--no-ignore',
        '/mnt/v/output/zensim','/mnt/v/zen/zensim-training',
        '/mnt/v/output/zensim-multicodec-probe'],text=True).splitlines()
    include = re.compile(r'kadid|konfig|(?:^|[/_])tid(?:[/_.]|2013)',re.I)
    exclude = re.compile(r'aic|cid22.?b|holdout|sealed|terminal|eval|select|valdigits|/val/|/test/',re.I)
    output = ROOT/'peer_schemas_corpus.json'
    assert not output.exists()
    records = []
    for filename in sorted(files):
        if not filename.endswith('.parquet') or not include.search(filename) or exclude.search(filename):
            continue
        names = pq.read_schema(filename).names
        peers = [s for s in names if re.search(r'ssim|peer|oracle',s,re.I)]
        record = dict(path=filename,columns=[s for s in names if not re.fullmatch(r'f\d+',s)],peer_columns=peers)
        records.append(record)
        if peers:
            print(json.dumps(dict(path=filename,peer_columns=peers),sort_keys=True))
    output.write_text(json.dumps(dict(payload_columns_read=[],files=records),indent=2)+'\n')
    print('schemas_only',len(records))
    print('with_peer_named_columns',sum(bool(r['peer_columns']) for r in records))


if __name__ == '__main__':
    main()
