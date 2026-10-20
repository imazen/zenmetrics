#!/usr/bin/env python3
"""Prepare/run an explicitly bound original HDR warm cell; preparation is file-only.

Execution belongs to the coordinating admission controller. Timed M-file bodies
are exact slices of the frozen resident fixture, including RGB8 conversion and
temporary release. No upstream metric or observer arithmetic is rewritten.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys


THREAD_ENV = ('RAYON_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'MKL_NUM_THREADS', 'RTEN_NUM_THREADS')
UNBOUND_ENV = ('OCTAVE_PATH', 'OCTAVE_EXEC_PATH', 'OCTAVE_HOME',
               'OCTAVE_SITE_INITFILE', 'OCTAVE_VERSION_INITFILE', 'LD_PRELOAD', 'LD_AUDIT')
# Exact photo512 bodies, including their bound dimensions (not the 64px grid).
BODY_SHA = {'hdrvdp': '440816f0d410742187b8d84f36d010cd8ddea5d05c5780b140af77374d7cf510',
            'hdrvdp3': '0c0fa5e5389ff3dccf93f42fdb765c89c9f212a3c26f445111c38a6db4dbf88a'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def verify(bindings):
    for path, expected in bindings.items():
        if sha(path) != expected:
            raise ValueError('bound source/input changed: ' + path)


def verified_seal(path, expected, parent_sha):
    if sha(path) != expected:
        raise ValueError('explicit preflight seal hash differs')
    seal = load(path)
    if (seal['passed'] is not True or seal['controls'] != 9
            or seal['parent_plan_sha256'] != parent_sha
            or seal['artifacts'].get(seal['preflight_receipt']) != seal['preflight_sha256']):
        raise ValueError('exact passed nine-preflight provenance required')
    verify(seal['artifacts'])
    found = set()
    for artifact in Path(seal['preflight_receipt']).parent.rglob('*'):
        if artifact.is_symlink(): raise ValueError('preflight artifact symlink refused')
        if artifact.is_file(): found.add(str(artifact))
        elif not artifact.is_dir(): raise ValueError('nonregular preflight artifact refused')
    if found != set(seal['artifacts']):
        raise ValueError('preflight artifact inventory changed')
    return seal


def digest(value):
    if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise argparse.ArgumentTypeError('explicit lowercase SHA256 required')
    return value


def split_original(text, metric):
    """Extract unchanged setup, input reads, evaluator body and cache assertions."""
    if text.count('clock=tic();\n') != 1 or text.count('elapsed=toc(clock)') != 1:
        raise ValueError('one frozen timed evaluator required')
    start = text.index('clock=tic();\n') + len('clock=tic();\n')
    end = text.index('elapsed=toc(clock)', start)
    body = text[start:end]
    if hashlib.sha256(body.encode()).hexdigest() != BODY_SHA[metric]:
        raise ValueError('original RGB8/call/release body differs')
    input_start = text.index('f=fopen(')
    warmup_start = text.index(body, input_start)
    setup = text[:input_start]
    inputs = text[input_start:warmup_start]
    suffix_start = text.index('for k=1:numel(cache_keys)\n[wl,rr]=load_spectral_resp', end)
    suffix = text[suffix_start:]
    if (inputs.count('f=fopen(') != 2 or inputs.count('fclose(f)') != 2
            or setup.count("first_boundary_spectral_cache('install'") != 1
            or "PARAMETER_CACHE_UNCHANGED passed=1" not in suffix):
        raise ValueError('original resident setup/input/cache layout differs')
    return dict(setup=setup, inputs=inputs, body=body, suffix=suffix)


def script_for(spec, samples):
    if type(samples) is not int or samples < 1:
        raise ValueError('explicit positive sample count required')
    parts = {name: Path(path).read_text() for name, path in spec['parts'].items()}
    if hashlib.sha256(parts['body'].encode()).hexdigest() != BODY_SHA[spec['metric']]:
        raise ValueError('timed body changed')
    # Match canonical_octave_cell: one COLD evaluation, then N SAMPLE rows.
    # All labels, checks, hex output and cache audits occur after toc.
    script = parts['setup'] + parts['inputs']
    script += f'first_boundary_bits=cell(1,{samples+1});\nfor call_index=0:{samples}\nclock=tic();\n'
    script += parts['body']
    script += r'''elapsed=toc(clock)*1e9;
assert(isscalar(score) && isreal(score) && isfinite(score));
assert(isscalar(elapsed) && isfinite(elapsed) && elapsed>0);
first_boundary_bits{call_index+1}=num2hex(score);
label='SAMPLE';if call_index==0;label='COLD';end
fprintf('%s %.0f %.17g\n',label,elapsed,score);fflush(stdout);
fprintf('SCORE_HEX %s\n',num2hex(score));fflush(stdout);
end
assert(all(strcmp(first_boundary_bits,first_boundary_bits{1})));
'''
    script += parts['suffix']
    return script


def prepare(parent_path, parent_sha, seal_path, seal_sha, output):
    parent_path = parent_path.resolve(strict=True)
    seal_path = seal_path.resolve(strict=True)
    if sha(parent_path) != parent_sha or sha(seal_path) != seal_sha:
        raise ValueError('explicit parent plan or seal hash differs')
    parent = load(parent_path)
    seal = verified_seal(seal_path, seal_sha, parent_sha)
    if (parent['variant'] != 'resize2201' or parent['matched_scope'] is not False
            or len(parent['cases']) != 104 or len(parent['preflight']) != 9
            or parent['expected_control_order'] != ['default', 'resident', 'resident', 'default']
            or seal['passed'] is not True or seal['controls'] != 9
            or seal['parent_plan_sha256'] != parent_sha):
        raise ValueError('unchanged resize2201 resident plan and passed nine-preflight seal required')
    verify(parent['bindings'])
    output.mkdir()
    root = output.resolve()
    helpers = root/'helpers'; helpers.mkdir()
    cell = helpers/Path(__file__).name
    shutil.copy2(__file__, cell)
    admitted = helpers/'first_boundary_hdrvdp_admitted.py'
    original_admitted = parent_path.parent/'helpers'/admitted.name
    if sha(original_admitted) != parent['bindings'][str(original_admitted)]:
        raise ValueError('frozen runtime-capture helper differs')
    shutil.copy2(original_admitted, admitted)
    parts_root = root/'original-evaluator-parts'; parts_root.mkdir()
    bindings = dict(parent['bindings'])
    bindings.update(seal['artifacts'])
    for path in (parent_path, seal_path, cell, admitted, Path(sys.executable).resolve(strict=True)):
        bindings[str(path)] = sha(path)
    specs = {}
    for metric in BODY_SHA:
        candidates = [c for c in parent['cases'] if c['panel'] == 'photo512'
                      and c['metric'] == metric and c['phase'] == 'warm' and c['threads'] == 1]
        if len(candidates) != 1:
            raise ValueError('one exact photo512/t1/warm predecessor required per route')
        old = candidates[0]
        source = Path(old['arms']['resident']['command'][-1])
        if sha(source) != bindings[str(source)] or old['evaluator_body_sha256'] != BODY_SHA[metric]:
            raise ValueError('bound predecessor wrapper/body differs')
        directory = parts_root/metric; directory.mkdir()
        parts = {}
        for name, text in split_original(source.read_text(), metric).items():
            path = directory/(name+'.m'); path.write_text(text)
            parts[name] = str(path); bindings[str(path)] = sha(path)
        # Keep the complete original wrapper as an independent readable source.
        original_copy = directory/'original-resident.m'; shutil.copy2(source, original_copy)
        bindings[str(original_copy)] = sha(original_copy)
        specs[metric] = dict(metric=metric, parts=parts, predecessor=old,
            original_wrapper=str(original_copy), original_wrapper_sha256=sha(source),
            evaluator_body_sha256=BODY_SHA[metric], threads=[1, 8])
    config = dict(schema=1, parent_plan=str(parent_path), parent_plan_sha256=parent_sha,
        preflight_seal=str(seal_path), preflight_seal_sha256=seal_sha,
        bindings=bindings, routes=specs, admitted_helper=str(admitted),
        matched_scope=False, qualification='unmatched_scope', qualified_timing_cells=0,
        display=dict(peak_nits=100, black_nits=1, pixels_per_degree=30),
        protocol='samples', cold_evaluations=1,
        timing='whole RGB8 conversion, unchanged original call, temporary release',
        outside_clock='input reads; original-helper spectral preload; sealed fixture; script setup',
        required_controls=['corresponding exact default/resident ABBA',
                           'original-cell/default/resident protocol parity',
                           'current40 native parity and resource gates'],
        execution_ready=False)
    config_path = root/'cell-config.json'; write(config_path, config)
    plan_bindings = dict(bindings, **{str(config_path): sha(config_path)})
    plans = {}
    for metric, spec in specs.items():
        cases = []
        for threads in (1, 8):
            old = spec['predecessor']
            row = {key: copy.deepcopy(old[key]) for key in
                   ('metric', 'width', 'height', 'reference', 'distorted', 'absolute_bound', 'relative_bound')}
            row.update(id=metric+f'-photo512-t{threads}-warm-resident-candidate', threads=threads,
                phase='warm', matched_scope=False, native_parity_qualified=False,
                qualification='unmatched_scope', qualified_timing_cells=0,
                scope=config['timing']+'; parameters resident; pending exact ABBA/native controls',
                canonical=dict(command=[sys.executable, str(cell), 'cell', str(config_path),
                    '--config-sha', sha(config_path), metric, '{width}', '{height}',
                    '{reference}', '{distorted}', '{threads}', '{samples}'],
                    protocol='samples', label='original official '+metric+' sRGB-display, resident parameters'),
                bound_source=old['bound_source'], bound_scope=old['bound_scope'])
            if metric == 'hdrvdp3':
                row['retained_other_gate'] = old['retained_other_gate']
            cases.append(row)
        path = root/(metric+'-candidate-plan.json')
        write(path, dict(schema=1, binary=spec['predecessor']['binary'],
            rust_source=spec['predecessor']['rust_source'], bindings=plan_bindings,
            cases=cases, matched_scope=False, qualification='unmatched_scope',
            qualified_timing_cells=0, execution_ready=False))
        plans[metric] = str(path)
    write(root/'candidate-metric-plans.json', dict(metric_plans=plans,
        binary=specs['hdrvdp']['predecessor']['binary'],
        rust_source=specs['hdrvdp']['predecessor']['rust_source'],
        matched_scope=False, qualification='unmatched_scope', execution_ready=False))
    index = dict(schema=1, config=str(config_path), config_sha256=sha(config_path),
        metric_plans=plans, plans_sha256={p: sha(p) for p in plans.values()},
        bundle_files={str(p): sha(p) for p in sorted(root.rglob('*')) if p.is_file()},
        metric_inference_executed=False, matched_scope=False, qualification='unmatched_scope')
    write(root/'index.json', index)
    for path in root.rglob('*'):
        if path.is_file(): path.chmod(0o444)
    return {k: index[k] for k in ('config', 'config_sha256', 'metric_plans', 'metric_inference_executed')}


def cell(args):
    if sha(args.config) != args.config_sha:
        raise ValueError('explicit cell config SHA256 differs')
    config = load(args.config)
    verify(config['bindings'])
    spec = config['routes'][args.metric]
    old = spec['predecessor']
    if (args.width, args.height, str(args.reference.resolve(strict=True)),
            str(args.distorted.resolve(strict=True))) != (
            old['width'], old['height'], old['reference'], old['distorted']):
        raise ValueError('unbound input identity or dimensions')
    if args.threads not in spec['threads'] or args.samples < 1:
        raise ValueError('explicit 1/8 thread budget and positive sample count required')
    if any(os.environ.get(key) != str(args.threads) for key in THREAD_ENV[:4]):
        raise ValueError('canonical environment and explicit thread argument disagree')
    if any(os.environ.get(key) for key in UNBOUND_ENV):
        raise ValueError('unbound runtime override')
    for role in ('reference', 'distorted'):
        if Path(old[role]).stat().st_size != args.width*args.height*3:
            raise ValueError('exact packed RGB8 input length required')
    parent = load(config['parent_plan'])
    if (sha(config['parent_plan']) != config['parent_plan_sha256']
            or sha(config['preflight_seal']) != config['preflight_seal_sha256']):
        raise ValueError('parent plan or preflight seal changed')
    seal = verified_seal(config['preflight_seal'], config['preflight_seal_sha256'],
                         config['parent_plan_sha256'])
    bundle = Path(config['parent_plan']).parent
    if list((bundle/'execution-cwd').iterdir()):
        raise ValueError('original evaluator cwd is not empty')
    root = Path(os.environ['TMPDIR']).resolve(strict=True)
    script = root/'hdr-resident-entry.m'
    with script.open('x') as stream: stream.write(script_for(spec, args.samples))
    helper_spec = importlib.util.spec_from_file_location('hdr_bound_capture', config['admitted_helper'])
    helper = importlib.util.module_from_spec(helper_spec); helper_spec.loader.exec_module(helper)
    env = dict(os.environ, RTEN_NUM_THREADS=str(args.threads), LC_ALL='C')
    for key in ('DISPLAY', 'WAYLAND_DISPLAY', 'XAUTHORITY'): env.pop(key, None)
    write(root/'hdr-cell-identity.json', dict(config_sha256=args.config_sha, metric=args.metric,
        threads=args.threads, samples=args.samples, evaluator_body_sha256=BODY_SHA[args.metric],
        script_sha256=sha(script), matched_scope=False, qualification='unmatched_scope'))
    before = helper.capture(parent, bundle, root/'runtime-before', env)
    def invariant(snapshot):
        return dict(files=snapshot['files'], dependencies=snapshot['dependencies'],
            environment={k: v for k, v in snapshot['environment'].items() if k not in THREAD_ENV})
    if invariant(before) != seal['baseline']:
        raise ValueError('runtime differs from sealed preflight')
    result = helper.execute([parent['octave'], '--no-gui', '--no-window-system',
        '--no-init-file', '--no-history', '--quiet', str(script)], env, root, 'original', 1100)
    write(root/'hdr-cell-exit.json', result)
    after = helper.capture(parent, bundle, root/'runtime-after', env)
    verify(config['bindings'])
    if (sha(args.config) != args.config_sha or before != after
            or sha(config['preflight_seal']) != config['preflight_seal_sha256']):
        raise ValueError('source/config/runtime changed during cell')
    verified_seal(config['preflight_seal'], config['preflight_seal_sha256'],
                  config['parent_plan_sha256'])
    print((root/'original-stdout.log').read_text(), end='', flush=True)
    print((root/'original-stderr.log').read_text(), end='', file=sys.stderr, flush=True)
    return result['returncode']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    prep = modes.add_parser('prepare')
    for name in ('parent', 'seal', 'output'): prep.add_argument(name, type=Path)
    prep.add_argument('--parent-sha', type=digest, required=True)
    prep.add_argument('--seal-sha', type=digest, required=True)
    active = modes.add_parser('cell')
    active.add_argument('config', type=Path)
    active.add_argument('--config-sha', type=digest, required=True)
    active.add_argument('metric', choices=BODY_SHA)
    for name in ('width', 'height'): active.add_argument(name, type=int)
    for name in ('reference', 'distorted'): active.add_argument(name, type=Path)
    for name in ('threads', 'samples'): active.add_argument(name, type=int)
    args = parser.parse_args()
    if args.mode == 'prepare':
        print(json.dumps(prepare(args.parent, args.parent_sha, args.seal, args.seal_sha, args.output)))
    else:
        raise SystemExit(cell(args))


if __name__ == '__main__': main()
