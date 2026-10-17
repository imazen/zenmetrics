#!/usr/bin/env python3
"""Weight tables from LLVM: the compiler's cost model and llvm-mca's scheduling model.

1. TargetTransformInfo cost model (what the vectorizer uses), read with
   `opt -passes='print<cost-model>' -cost-kind=<throughput|latency>` on one
   scalar IR instruction or intrinsic per op class and precision, for each
   -mcpu. Costs are LLVM's abstract units, not cycles.
2. llvm-mca per-instruction tables measured into
   benchmarks/iqa_complexity_2026-10-17/weights/llvm_mca_<cpu>.tsv, mapped to op
   classes with the same scalar instructions as weights_from_probes.py.
Writes scripts/complexity/weights_llvm_<source>_<cpu>_<kind>.json.

  llvm_cost_tables.py [--opt /usr/lib/llvm-22/bin/opt] [--cpus arrowlake,znver5,x86-64-v3]
"""
import argparse, csv, json, re, subprocess, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
T = {'f32': 'float', 'f64': 'double'}
S = {'f32': 'f32', 'f64': 'f64'}
OPS = {  # op class -> IR template ({t} type, {s} intrinsic suffix)
    'add': '%r = fadd {t} %a, %b', 'mul': '%r = fmul {t} %a, %b', 'div': '%r = fdiv {t} %a, %b',
    'fma': '%r = call {t} @llvm.fma.{s}({t} %a, {t} %b, {t} %c)', 'sqrt': '%r = call {t} @llvm.sqrt.{s}({t} %a)',
    'cmp': '%r = call {t} @llvm.minnum.{s}({t} %a, {t} %b)', 'abs': '%r = call {t} @llvm.fabs.{s}({t} %a)',
    'exp': '%r = call {t} @llvm.exp.{s}({t} %a)', 'log': '%r = call {t} @llvm.log.{s}({t} %a)',
    'pow': '%r = call {t} @llvm.pow.{s}({t} %a, {t} %b)', 'trig': '%r = call {t} @llvm.cos.{s}({t} %a)',
    'load': '%r = load {t}, ptr %p', 'store': 'store {t} %a, ptr %p',
}
EXTRA = {'cvt': {'f32': '%r = fpext float %a to double', 'f64': '%r = fptrunc double %a to float'},
         'int': {'*': '%r = add i32 %i, %j'}}
MCA = {'add': ('vaddss', 'vaddsd'), 'mul': ('vmulss', 'vmulsd'), 'fma': ('vfmadd231ss', 'vfmadd231sd'),
       'div': ('vdivss', 'vdivsd'), 'sqrt': ('vsqrtss', 'vsqrtsd'), 'rcp': ('vrcpss', None),
       'rsqrt': ('vrsqrtss', None), 'cmp': ('vminss', 'vminsd'), 'abs': ('vandps_x', 'vandpd_x'),
       'cvt': ('vcvtss2sd', 'vcvtsd2ss')}


def tti(opt, cpu, kind, line, t='float'):
    ir = f'define void @f({t} %a, {t} %b, {t} %c, ptr %p, i32 %i, i32 %j) {{\n  {line}\n  ret void\n}}\n'
    with tempfile.NamedTemporaryFile('w', suffix='.ll', dir=Path.home()/'tmp', delete=False) as f:
        f.write(ir)
    out = subprocess.run([opt, '-mtriple=x86_64-pc-linux-gnu', f'-mcpu={cpu}', "-passes=print<cost-model>",
                          f'-cost-kind={kind}', '-disable-output', f.name], capture_output=True, text=True)
    Path(f.name).unlink()
    m = re.search(r'cost of (\d+(?:\.\d+)?) for instruction:\s+(?:%r = |store)', out.stdout + out.stderr)
    if not m:
        raise SystemExit(f'no cost for {cpu} {kind} {line}: {out.stderr[-300:]}')
    return float(m.group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--opt', default='/usr/lib/llvm-22/bin/opt')
    ap.add_argument('--cpus', default='arrowlake,znver5,x86-64-v3')
    a = ap.parse_args()
    ver = subprocess.run([a.opt, '--version'], capture_output=True, text=True).stdout.split('\n')[1].strip()
    for cpu in a.cpus.split(','):
        for kind in ('throughput', 'latency'):
            w = {}
            for op, tmpl in OPS.items():
                for prec in ('f32', 'f64'):
                    w.setdefault(op, {})[prec] = tti(a.opt, cpu, kind, tmpl.format(t=T[prec], s=S[prec]), T[prec])
            for op, by in EXTRA.items():
                for prec, line in by.items():
                    w.setdefault(op, {})[prec] = tti(a.opt, cpu, kind, line, 'float' if prec != 'f64' else 'double')
            for op in w:
                w[op].setdefault('mixed', w[op].get('f32', next(iter(w[op].values()))))
            (HERE/f'weights_llvm_tti_{cpu}_{kind}.json').write_text(json.dumps({
                'name': f'LLVM cost model ({ver}) -mcpu={cpu} {kind}',
                'source': f"opt -passes='print<cost-model>' -cost-kind={kind}, one scalar instruction per class "
                          '(scripts/complexity/llvm_cost_tables.py); abstract TTI units, libcalls use generic costs',
                'weights': w}, indent=1) + '\n')
    for cpu in ('arrowlake', 'znver5'):
        path = ROOT/f'benchmarks/iqa_complexity_2026-10-17/weights/llvm_mca_{cpu}.tsv'
        rows = list(csv.DictReader(open(path), delimiter='\t'))
        for kind, col in (('rthroughput', 'rthroughput'), ('latency', 'latency')):
            w = {}
            for op, (s32, s64) in MCA.items():
                for prec, inst in (('f32', s32), ('f64', s64)):
                    hit = [r for r in rows if r['instruction'] == inst and r[col] not in ('', 'NA')]
                    if inst and hit:
                        w.setdefault(op, {})[prec] = float(hit[0][col])
            for op in w:
                w[op].setdefault('mixed', w[op].get('f32', next(iter(w[op].values()))))
            model = rows[0].get('sched_model', cpu)
            (HERE/f'weights_llvm_mca_{cpu}_{kind}.json').write_text(json.dumps({
                'name': f'llvm-mca -mcpu={cpu} ({model} model) {kind}',
                'source': f'{path.relative_to(ROOT)}; scalar instruction per class; libm calls absent from mca',
                'weights': w}, indent=1) + '\n')


if __name__ == '__main__':
    main()
