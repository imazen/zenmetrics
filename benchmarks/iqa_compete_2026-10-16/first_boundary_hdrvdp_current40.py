#!/usr/bin/env python3
"""Prepare current40 HDR plans and original-call controls without execution.

Preserves prior cohorts. Native source verification and root heavy admission
are required before any emitted command may run.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from first_boundary_hdrvdp_octave import script

SOURCE = 'ea767ba1a4029963bc56a74fb824e64455e4599e'
BINARY_SHA = '19898e546917c9b62badcf1733ec74aacb3e02b4e14d88af1e31923a88a54005'
GLUE_SHA = '6b790071435a1bcc943f7d3da17b2353f6de434d4a526abb3dfeee4560c31b94'
FEATURES = ['compete', 'metrics-parallel', 'zenmetrics-cli/nr']


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            result.update(chunk)
    return result.hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def check_build(receipt, observed_binary_sha):
    if (receipt['source_revision'] != SOURCE or receipt['binary_sha256'] != BINARY_SHA
            or observed_binary_sha != BINARY_SHA or receipt['adapters_applied'] is not False
            or sorted(receipt['features']) != FEATURES or receipt['lto'] != 'thin'
            or receipt['native_cpu_flags'] != 'RUSTFLAGS,CARGO_ENCODED_RUSTFLAGS,CARGO_BUILD_RUSTFLAGS unset'):
        raise ValueError('current40 source/binary/build contract differs')


def bounds(metric):
    if metric == 'hdrvdp':
        return dict(absolute_bound=5e-3, relative_bound=0,
            bound_source='crates/hdrvdp/tests/golden_srgb_display.rs:TOL_Q',
            bound_scope='Existing official sRGB-display v2 Q limit; unchanged',
            native_parity_qualified=False)
    if metric == 'hdrvdp3':
        return dict(absolute_bound=None, relative_bound=0,
            bound_source=None, bound_scope='No assigned original-Octave bound for this photo/grid cohort',
            retained_other_gate=dict(absolute_bound=1e-9,
                source='crates/hdrvdp/tests/v3_reference.rs:TOL_Q_JOD',
                scope='Frozen NumPy reference corpus only; not assigned to this official-Octave cohort'),
            native_parity_qualified=False)
    raise ValueError('base HDR metrics only')


def original_grid(plan):
    rows = [c for c in plan['cases'] if c['metric'] == 'hdrvdp-aic2026']
    identities = {(c['width'], c['height'], c['reference'], c['distorted']) for c in rows}
    if len(identities) != 12 or len(rows) != 24 or {c['threads'] for c in rows} != {1, 28}:
        raise ValueError('exact original twelve-pair one/twenty-eight-thread grid required')
    for identity in identities:
        subset = [c for c in rows if (c['width'], c['height'], c['reference'], c['distorted']) == identity]
        if sorted(c['threads'] for c in subset) != [1, 28]:
            raise ValueError('each original pair requires both budgets without duplicates')
    return rows


def prepare(build, frozen, grid_path, photo_path, output):
    build, frozen, grid_path, photo_path = [p.resolve(strict=True) for p in (build, frozen, grid_path, photo_path)]
    binary, receipt_path = build / 'target/release/iqa-compete', build / 'build-receipt.json'
    receipt = json.loads(receipt_path.read_text())
    check_build(receipt, sha(binary))
    glue = build / 'work/zenmetrics/crates/zenmetrics-cli/src/metrics/hdrvdp_aic.rs'
    if sha(glue) != GLUE_SHA:
        raise ValueError('current40 production HDR glue differs from reviewed body')
    old_grid, old_photo = [json.loads(p.read_text()) for p in (grid_path, photo_path)]
    grid = original_grid(old_grid)
    photo = next(c for c in old_photo['cases'] if c['width'] == c['height'] == 512
                 and '/pairA/' in c['reference'])
    photos = [dict(photo, id=f'photo512-t{threads}', threads=threads,
                   input_content='photo, retained S67 512x512 RGB pair') for threads in (1, 28)]
    output.mkdir()
    root = output.resolve()
    helpers = root / 'helpers'
    helpers.mkdir()
    for name in ('first_boundary_hdrvdp_current40.py', 'first_boundary_hdrvdp_octave.py'):
        shutil.copy2(Path(__file__).with_name(name), helpers / name)
    sources = root / 'source'
    sources.mkdir()
    shutil.copy2(glue, sources / 'hdrvdp_aic.rs')
    # Preserve the supplied receipt bytes, not a reworded receipt.
    shutil.copy2(receipt_path, root / 'build-receipt.json')
    shared = {str(p): sha(p) for p in [binary, receipt_path, grid_path, photo_path,
              root / 'build-receipt.json', sources / 'hdrvdp_aic.rs', *helpers.iterdir()]}
    # Keep the explicitly declared legacy runtime identity of the retained
    # prepared cohort. The entry asserts it; this does not qualify that runtime.
    preceding = json.loads((frozen / 'runtime-template.json').read_text())
    runtime = Path(preceding['octave']).resolve(strict=True)
    reference_bindings = {str(runtime): sha(runtime)}
    for tag in ('official-v2', 'official-v3', 'statistics'):
        for path in sorted((frozen / tag).rglob('*')):
            if path.is_file():
                reference_bindings[str(path.resolve())] = sha(path)
    shared.update(reference_bindings)
    emitted = []
    for panel, pairs, predecessor in [('photo512', photos, old_photo), ('original12', grid, old_grid)]:
        for pair in pairs:
            for field in ('reference', 'distorted'):
                if sha(pair[field]) != predecessor['bindings'].get(pair[field]):
                    raise ValueError('retained pair changed: ' + pair[field])
                shared[pair[field]] = predecessor['bindings'][pair[field]]
            for metric, tag in [('hdrvdp', 'official-v2'), ('hdrvdp3', 'official-v3')]:
                for phase, samples in [('first', 1), ('warm', 5)]:
                    identity = pair['id'].removeprefix('hdrvdp-aic2026-')
                    name = f'{panel}-{metric}-{identity}-{phase}-current40'
                    if Path(name).name != name:
                        raise ValueError('case id must be a single directory name')
                    tree = frozen / tag
                    local = {p: h for p, h in reference_bindings.items()
                             if Path(p).is_relative_to(tree) or '/statistics/' in p or p == str(runtime)}
                    local.update({pair[field]: shared[pair[field]] for field in ('reference', 'distorted')})
                    config = dict(metric=metric, phase=phase, samples=samples,
                        width=pair['width'], height=pair['height'], threads=pair['threads'],
                        reference=pair['reference'], distorted=pair['distorted'], tree=str(tree),
                        octave=str(runtime), octave_version=preceding['octave_version'],
                        image_version=preceding['image_version'], geomean=str(frozen / 'statistics/geomean.m'),
                        bindings=local)
                    native = dict(width=pair['width'], height=pair['height'], threads=pair['threads'],
                        reference=pair['reference'], distorted=pair['distorted'], metric_name=metric,
                        only_mode='score', timing_phase='cold' if phase == 'first' else 'warm',
                        probe_only=False, required_samples=samples, sample_target_ns=10_000_000,
                        chunk_budget_ms=240_000, locked_process_accounting=True)
                    native_path = root / (name + '-native.json')
                    write(native_path, native)
                    ref_config = root / (name + '-reference.json')
                    write(ref_config, config)
                    commands = {}
                    for role in ('canonical', 'direct-control'):
                        path = root / (name + '-' + role + '.m')
                        with path.open('x') as stream:
                            stream.write(script(config, direct_control=role == 'direct-control'))
                        shared[str(path)] = sha(path)
                        commands[role] = dict(command=[str(runtime), '--no-gui', '--no-init-file',
                            '--no-history', '--quiet', str(path)], protocol='first' if phase == 'first' else 'samples')
                    for path in (native_path, ref_config):
                        shared[str(path)] = sha(path)
                    emitted.append(dict(id=name, panel=panel, metric=metric, phase=phase,
                        width=pair['width'], height=pair['height'], threads=pair['threads'], samples=samples,
                        reference=pair['reference'], distorted=pair['distorted'],
                        input_content=pair.get('input_content'), binary=str(binary), rust_source=SOURCE,
                        native=dict(command=[str(binary), str(native_path)], protocol='iqa-row'),
                        native_config=str(native_path), reference_config=str(ref_config), **commands,
                        **bounds(metric), matched_scope=False,
                        scope='Evaluator first/warm; original spectral-file I/O timed, native embedded spectral data; no process-startup/cache-cold or runtime equivalence claim',
                        environment={key: str(pair['threads']) for key in ('RAYON_NUM_THREADS', 'OMP_NUM_THREADS',
                                     'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RTEN_NUM_THREADS')}))
    common = dict(schema=1, binary=str(binary), rust_source=SOURCE, binary_sha256=BINARY_SHA,
        build_receipt_sha256=sha(receipt_path), production_glue_sha256=GLUE_SHA, bindings=shared,
        execution_ready=False, qualified_timing_cells=0,
        required_before_execution=['Root exhaustive source verification receipt for this binary',
            'Root code review, shared heavy admission and bound runtime/image/shared-library closure'],
        exact_original_call_gate='Each canonical/direct arm twice in separate fresh processes, alternating; every scalar f64 bit must match. This does not assign native v3 parity.',
        native_thread_gate='Exact first/warm score bits must agree between thread budgets 1 and 28; retain any disagreement as a finding',
        runtime_contract=dict(octave_version=preceding['octave_version'], image_version=preceding['image_version'],
            qualification='Declared legacy runtime; original-call wiring controls may run here, native parity remains diagnostic unless runtime qualified'),
        optional_eight_threads='Not added: requested photo and original-grid panels use 1/28; changed-recipe default may get a separate successor if root needs it')
    for panel in ('photo512', 'original12'):
        for phase in ('first', 'warm'):
            write(root / f'{panel}-{phase}.plan.json', dict(common, panel=panel, phase=phase,
                cases=[c for c in emitted if c['panel'] == panel and c['phase'] == phase]))
    summary = dict(source=SOURCE, binary_sha256=BINARY_SHA, build_receipt_sha256=sha(receipt_path),
        cases=len(emitted), panels={panel: sum(c['panel'] == panel for c in emitted) for panel in ('photo512', 'original12')},
        plans={str(p): sha(p) for p in sorted(root.glob('*.plan.json'))}, bindings=shared,
        execution_ready=False, qualified_timing_cells=0)
    write(root / 'index.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('build', 'frozen', 'grid', 'photo', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    result = prepare(args.build, args.frozen, args.grid, args.photo, args.output)
    print(json.dumps({k: result[k] for k in ('source', 'binary_sha256', 'cases', 'panels', 'execution_ready')}))


if __name__ == '__main__':
    main()
