#!/usr/bin/env python3
"""Verify each immutable candidate changes only its declared arithmetic."""
import csv
from pathlib import Path
import re

from dmm_bisect_sources import BASE, function, source

expected = {
    'baseline': set(),
    'epsilon': set(),
    'small_n_scaling': {'svd16'},
    'org2r_order': {'dorg2r_q'},
    'lartg': {'dlartgp', 'dlartg'},
    'orm2r_order': {'dorg2r_q', 'sorm2r_q', 'svd16'},
    'right_sger_order': {'dlarf_right'},
    'f64_norm': {'dnrm2'},
    'reference_norm': {'dnrm2'},
}
base = source(BASE)
base_names = set(re.findall(r'(?m)^fn (\w+)\(', base))
expected['lartg'] |= {name for name in base_names if name != 'dlartgp'
                       and re.search(r'\bdlartgp\(', function(base, name))}


def code(text, name):
    part = function(text, name)
    return part[part.index('fn '):]


with Path(__file__).with_name('dmm_bisect_sources_2026-10-17.tsv').open() as stream:
    for entry in csv.DictReader(stream, delimiter='\t'):
        name = entry['change']
        candidate = source(entry['commit'])
        prefix = candidate.split('const D_EPS:')[0]
        if name != 'baseline':
            prefix = prefix.removeprefix(f'//! Isolated bisection experiment: {name}; not qualified for shipping.\n')
            prefix = prefix.replace('const SCORE_REVISION: u32 = 2;', 'const SCORE_REVISION: u32 = 1;')
        assert prefix == base.split('const D_EPS:')[0], (name, 'resource/prefix change')
        assert candidate[candidate.index('#[cfg(test)]'):] == base[base.index('#[cfg(test)]'):], (name, 'test change')
        names = set(re.findall(r'(?m)^fn (\w+)\(', candidate))
        changed = base_names ^ names
        changed |= {fn for fn in base_names & names if code(base, fn) != code(candidate, fn)}
        assert changed == expected[name], (name, changed, expected[name])
        epsilon = re.search(r'(?m)^const D_EPS:.*$', candidate).group()
        wanted = 'const D_EPS: f32 = f32::EPSILON * 0.5;' if name == 'epsilon' else 'const D_EPS: f32 = f32::EPSILON;'
        assert epsilon == wanted, name
        assert re.search(r'(?m)^const D_SAFMIN:.*$', candidate).group() == 'const D_SAFMIN: f32 = f32::MIN_POSITIVE;'
        print(name, entry['commit'][:12], 'isolated:', ','.join(sorted(changed)) or ('D_EPS' if name == 'epsilon' else 'unchanged'))
