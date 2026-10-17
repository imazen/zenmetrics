#!/usr/bin/env python3
"""Generate independent arithmetic experiments from the frozen DMM solver.

No golden values or limits are changed. Each output differs numerically from
baseline by one named change, plus an experimental score-revision label.
"""
import argparse
from pathlib import Path
import re
import subprocess

SOURCE = 'crates/zenmetrics-nn/src/dmm.rs'
BASE = '66bb98605c838a1c78302a6fe35df8cdb4b5f64e'


def source(revision):
    return subprocess.check_output(['jj', 'file', 'show', '-r', revision, SOURCE], text=True)


def span(text, name):
    match = re.search(rf'(?m)(?:^///[^\n]*\n)*^fn {re.escape(name)}\(', text)
    assert match, name
    brace = text.index('{', match.end())
    depth = 0
    for i in range(brace, len(text)):
        depth += (text[i] == '{') - (text[i] == '}')
        if depth == 0:
            return match.start(), i + 1
    raise AssertionError(f'unclosed function {name}')


def function(text, name):
    a, b = span(text, name)
    return text[a:b]


def replace(text, name, body):
    a, b = span(text, name)
    return text[:a] + body + text[b:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base = source(BASE)
    corrected = source('4ed04e9c')
    variants = {}
    variants['epsilon'] = base.replace('const D_EPS: f32 = f32::EPSILON;', 'const D_EPS: f32 = f32::EPSILON * 0.5;')

    body = function(base, 'svd16')
    a = body.index('    // dbdsdc scale')
    b = body.index('    let mut ub', a)
    body = body[:a] + '    let mut ds = d;\n    let mut es = e;\n' + body[b:]
    a = body.index('    let mut s = [0f32; 16];', body.index('    // U = Q1'))
    body = body[:a] + '    (u, ds)\n}'
    body = body.replace('// U = Q1 @ Ub ; s = ds * orgnrm', '// U = Q1 @ Ub; no small-N bidiagonal scaling')
    variants['small_n_scaling'] = replace(base, 'svd16', body)
    variants['org2r_order'] = replace(base, 'dorg2r_q', function(source('1e67c476'), 'dorg2r_q'))
    variants['lartg'] = replace(base, 'dlartgp', function(corrected, 'dlartg')).replace('dlartgp(', 'dlartg(')

    text = replace(base, 'dorg2r_q', function(corrected, 'sorm2r_q'))
    body = function(text, 'svd16')
    body = body.replace('    // a now holds reflector tails below diag; build Q1 in place\n    dorg2r_q(&mut a, &tq); // a <- Q1\n', '    // Retain reflectors for direct SORM2R application.\n')
    body = body.replace('        let u = a; // Q1 (identity for zero matrix)', '''        let mut u = [0.0; 256];
            for i in 0..16 { u[i * 16 + i] = 1.0; }
            sorm2r_q(&a, &tq, &mut u);''')
    a = body.index('    // U = Q1 @ Ub')
    b = body.index('    let mut s = [0f32; 16];', a)
    body = body[:a] + '    sorm2r_q(&a, &tq, &mut ub);\n    let u = ub;\n' + body[b:]
    variants['orm2r_order'] = replace(text, 'svd16', body)
    variants['right_sger_order'] = replace(base, 'dlarf_right', function(corrected, 'dlarf_right'))
    variants['f64_norm'] = replace(base, 'dnrm2', function(source('3ed5c251'), 'dnrm2'))
    variants['reference_norm'] = replace(base, 'dnrm2', function(source('288c2c69'), 'dnrm2'))

    for name, text in variants.items():
        assert text != base, name
        text = text.replace('const SCORE_REVISION: u32 = 1;', 'const SCORE_REVISION: u32 = 2;')
        text = f'//! Isolated bisection experiment: {name}; not qualified for shipping.\n' + text
        (args.output / f'{name}.rs').write_text(text)
        print(name)


if __name__ == '__main__':
    main()
