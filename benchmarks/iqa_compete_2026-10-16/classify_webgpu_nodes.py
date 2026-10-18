#!/usr/bin/env python3
"""Classify verbose capability observations without inferring final placements."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def classify(text):
    missing = sorted(set(re.findall(
        r'webgpu kernel not found in registries for Op type: (\S+) node name: ([^\r\n]+)', text)))
    preferred = sorted(set(re.findall(
        r'Force fallback to CPU execution for node (.*?) because the CPU execution path', text)))
    return dict(registry_misses=[dict(op=op, node=node) for op, node in missing],
                registry_op_counts=dict(sorted(Counter(op for op, _ in missing).items())),
                cpu_preferred_nodes=preferred)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = dict(scope='unique observations across capability passes; not final CPU placements', models={})
    for model in ['afine', 'arniqa', 'ckdn', 'deepdc']:
        log = args.directory / (model + '.log')
        receipt = args.directory / (model + '.json')
        identity = json.loads(receipt.read_text())
        if identity['status'] != 'session_refused' or not identity['fallback_disabled'] or identity['python_wrapper_fallback']:
            raise ValueError('expected strict refusal: ' + model)
        if identity['model_sha256'] != identity['model_sha256_after']:
            raise ValueError('model changed: ' + model)
        result['models'][model] = dict(**classify(log.read_text()),
            identity={key: value for key, value in identity.items() if key != 'nodes'},
            log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
            receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest())
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
