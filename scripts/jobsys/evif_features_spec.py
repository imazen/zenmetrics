#!/usr/bin/env python3
"""Turn an `evif_pack_index.py` selection into a `zenfleet-ctl declare-evif-features` spec.

usage: evif_features_spec.py INDEX_DIR OUT_SPEC [--peak-mem-bytes N]

Each trade job's EVIF members are split into feature jobs of at most 128 containers
(the declaration's limit), ordered by pixel work, largest first, so that a manifest-order
claim ends a pass on short jobs. The implementation hashes bind the frozen extractor, the
executor file that reads pack members and the index manifest.
"""
import argparse
import hashlib
import json
from pathlib import Path

EXTRACTOR_SHA = 'ed4957417339779dd8ed25fe59a6166326ba9db86d5c1530b1e1c788c9d614d8'
CHUNK = 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('index_dir')
    ap.add_argument('out_spec')
    ap.add_argument('--peak-mem-bytes', type=int, default=512 * 1024 * 1024)
    a = ap.parse_args()
    index = Path(a.index_dir)
    executor = Path(__file__).with_name('evif_features.py').read_bytes()
    pins = [EXTRACTOR_SHA, hashlib.sha256(executor).hexdigest(),
            hashlib.sha256((index / 'MANIFEST.json').read_bytes()).hexdigest()]
    entries = []
    assigned = set()
    for line in (index / 'selection.jsonl').open():
        sel = json.loads(line)
        # A container shared by two trade jobs (split-family templates of one view) is extracted once.
        shas = [s for s in sel['evif_shas'] if s not in assigned]
        assigned.update(shas)
        for start in range(0, len(shas), CHUNK):
            part = shas[start:start + CHUNK]
            entries.append((sel['source']['width'] * sel['source']['height'] * len(part),
                            dict(source=sel['source'], evif_shas=part)))
    entries.sort(key=lambda e: -e[0])
    spec = dict(sources=[e for _, e in entries], implementation_shas=pins, peak_mem_bytes=a.peak_mem_bytes)
    Path(a.out_spec).write_text(json.dumps(spec) + '\n')
    print(json.dumps(dict(jobs=len(entries), containers=sum(len(e['evif_shas']) for _, e in entries),
                          megapixel_pairs=round(sum(w for w, _ in entries) / 1e6, 1), pins=pins)))


if __name__ == '__main__':
    main()
