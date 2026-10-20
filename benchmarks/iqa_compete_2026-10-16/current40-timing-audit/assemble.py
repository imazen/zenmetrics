#!/usr/bin/env python3
"""Assemble the reviewed source matrix. Stdlib file reads only; launches nothing.

Inputs are the selection.py shards, validated comparative plan, and retained
verified build. Refuses an existing output directory and any source hash drift.
This is a source recorder, never an execution/qualification tool.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
from profiles import COMMON, PROFILES, profile_for, BENCH

EA = 'ea767ba1a4029963bc56a74fb824e64455e4599e'
ROOT = Path('/home/lilith/tmp/local-canonical-20261019T2330')
REPO = Path('/home/lilith/work/zen/zenmetrics')
PYLIB = Path('/home/lilith/tmp/portcheck/python/lib/python3.12/site-packages')
EXTRAS = ROOT/'oracle-extras'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    encoded = json.dumps(obj, indent=2, allow_nan=False)+'\n'
    assert len(encoded.encode()) < 30000, str(path)
    path.write_text(encoded)


def source(path, bindings, witness=None):
    path = Path(path)
    actual = sha(path)
    expected = bindings.get(str(path))
    if expected is not None:
        assert actual == expected, f'source drift: {path}'
    if witness is not None:
        assert actual == witness, f'receipt source drift: {path}'
    return dict(path=str(path), sha256=actual, selected_plan_sha256=expected,
                selected_plan_matches=(None if expected is None else True),
                receipt_expected_sha256=witness,
                proof=('plan-bound source' if expected else 'receipt-bound source' if witness else
                       'observed source; selected plan does not bind this path'))


def canonical_sources(metric, selected, plan):
    bindings = plan['bindings']
    argv = selected['canonical']['command']
    profile = profile_for(metric)
    paths = []
    if selected['python_entry']:
        paths.append(Path(selected['python_entry']['path']))
    if profile in ('vmaf-model','vmaf-feature'):
        paths.append(REPO/BENCH/'libvmaf_rgb_compare.cpp')
    elif profile == 'pyiqa':
        manifest = Path(argv[argv.index('--fixed-manifest')+1])
        cfg = read(manifest)
        paths += [manifest, Path(cfg['output']['path']),
                  REPO/BENCH/'check_fixed_factor_oracle.py',
                  Path('/home/lilith/tmp/portcheck/pyiqa-env/lib/python3.12/site-packages/pyiqa/archs/ssim_arch.py')]
    elif profile == 'octave':
        specpath = Path(argv[2])
        spec = read(specpath)[metric]
        main = Path(spec['source'])
        paths += [specpath, main]
        if metric == 'msssim': paths.append(main.parent/'ssim_index_new.m')
        if metric == 'vifvec':
            paths += [Path(spec['paths'][1])/'buildSpyr.m',
                      Path(spec['paths'][-1])/'vifsub_est_m.m']
    elif profile == 'nr':
        if metric == 'brisque': paths.append(PYLIB/'piq/brisque.py')
        elif metric == 'piqe': paths.append(EXTRAS/'pypiqe/piqe.py')
        else: paths += [EXTRAS/'basicsr/metrics/niqe.py',
                        EXTRAS/'basicsr/utils/color_util.py',
                        EXTRAS/'basicsr/utils/matlab_functions.py']
    elif profile == 'ssim2':
        paths += [REPO/BENCH/'ssimulacra2_primary_compute.cpp',
                  Path('/home/lilith/tmp/portcheck-primary-rgb-build-20261019T0948/src-ssim2-extract/src/lib/threads/thread_parallel_runner_internal.cc')]
    elif profile == 'butter':
        paths += [REPO/BENCH/'libjxl_compute.cpp',
                  Path('/home/lilith/tmp/portcheck/sources/libjxl/libjxl-0.12.0/tools/thread_pool_internal.h'),
                  Path('/home/lilith/tmp/portcheck/sources/libjxl/libjxl-0.12.0/lib/threads/thread_parallel_runner_internal.cc')]
    elif profile == 'ssim1':
        receipt = ROOT/'ssim1-rgb-build/build.json'
        cfg = read(receipt)
        return [source(receipt, bindings),
                source(receipt.parent/'reference.cpp', bindings, cfg['generated_source_sha256']),
                source(cfg['source'], bindings, cfg['source_sha256'])]
    elif profile == 'dssim':
        paths += [REPO/BENCH/'dssim_reference.c', REPO/BENCH/'build_dssim_reference.sh',
                  ROOT/'dssim-author-reference-build-v2/build.sha256']
    elif profile == 'flip':
        paths.append(PYLIB/'flip_evaluator/flip_python_api.py')
    elif profile == 'wshv':
        paths.append(Path(argv[argv.index('--source')+1]))
    elif profile == 'hdr-aic':
        official = Path(argv[2])
        if metric == 'hdrvdp-aic2026':
            paths += [official/'hdrvdp.m', official/'load_spectral_resp.m']
        else:
            paths += [official/'hdrvdp3.m', official/'hdrvdp_visual_pathway.m',
                      official/'utils/load_spectral_resp.m']
    elif profile == 'hdr-base':
        config = Path(argv[3])
        cfg = read(config)
        route = cfg['routes'][metric]
        parent = Path(cfg['parent_plan']).parent
        official = parent/('official-v2' if metric == 'hdrvdp' else 'official-v3')
        paths += [config, *map(Path, route['parts'].values()),
                  parent/'helpers/first_boundary_hdrvdp_resident.py',
                  official/('hdrvdp.m' if metric == 'hdrvdp' else 'hdrvdp3.m'),
                  official/('load_spectral_resp.m' if metric == 'hdrvdp' else 'utils/load_spectral_resp.m')]
    return [source(p, bindings) for p in dict.fromkeys(paths)]


def main(args):
    index = read(args.selections/'index.json')
    comparative = read(args.comparative)
    verified = read(args.verified_build)
    assert verified['source_revision'] == EA
    assert verified['binary_sha256'] == '19898e546917c9b62badcf1733ec74aacb3e02b4e14d88af1e31923a88a54005'
    assert sha(args.comparative) == index['comparative_sha256']
    assert len(comparative['cases']) == 80 and len(index['records']) == 40
    snapshot = args.verified_build.parent/'work/zenmetrics'
    native_cache = {}

    def native_source(relative):
        if relative not in native_cache:
            expected = verified['files'][relative]
            actual = sha(snapshot/relative)
            assert actual == expected, f'native source drift: {relative}'
            native_cache[relative] = dict(repo_path=relative,
                frozen_path=str(snapshot/relative), sha256=actual,
                verified_build_matches=True, revision=EA)
        return native_cache[relative]

    args.output.mkdir()
    rows = []
    for entry in index['records']:
        selected = read(args.selections/entry['selection'])
        metric = selected['metric']
        plan = read(selected['plan'])
        assert sha(selected['plan']) == selected['plan_sha256']
        cases = [c for c in comparative['cases'] if c['metric'] == metric]
        assert len(cases) == 2 and {c['threads'] for c in cases} == {1,8}
        chosen = next(c for c in plan['cases'] if c['id'] == selected['selected_case_id'])
        profile = profile_for(metric)
        notes = PROFILES[profile]
        argv = selected['canonical']['command']
        assert '{threads}' in argv
        input_records = []
        for key in ('reference','distorted'):
            filename = cases[0][key]
            # Source audit records prior binding; it does not re-read image data.
            input_records.append(dict(role=key, path=filename,
                selected_plan_sha256=plan['bindings'].get(filename),
                byte_rehashed_in_this_audit=False))
        row = dict(metric=metric, detailed_source_review=True, profile=profile,
            selected_plan=dict(path=selected['plan'], sha256=selected['plan_sha256'],
                case_id=selected['selected_case_id'], source_thread=selected['selected_thread']),
            comparative_cases=selected['comparative_cases'],
            geometry=dict(width=selected['width'],height=selected['height']),
            inputs=input_records,
            command=argv, command_sha256=selected['canonical_command_sha256'],
            command_digest_encoding='SHA256 of compact JSON argv; templates preserved',
            command_executable_bound_sha256=plan['bindings'].get(argv[0]),
            canonical_label=selected['canonical'].get('label'),
            canonical_source_records=canonical_sources(metric, selected, plan),
            native_source_records=[native_source(p) for p in notes['native_sources']],
            common_source_registry='../source-registry.json',
            shared_warm_boundary=COMMON,
            route_review={k:v for k,v in notes.items() if k not in ('native_sources','entry_names')},
            numerical_contract=dict(absolute_bound=selected['absolute_bound'],
                relative_bound=selected['relative_bound'],
                native_parity_qualified=chosen.get('native_parity_qualified'),
                assigned_new_bound=False),
            original_scope_claim=dict(matched_scope=chosen['matched_scope'],scope=selected['scope']),
            audit_disposition=('preparation asymmetry; numerical observations retained; no strict timing qualification by this audit'
                if profile in ('vmaf-model','hdr-aic','hdr-base') else
                'reviewed declared boundary with explicit route/provenance/thread limits; no runtime qualification by this audit'))
        target = args.output/(metric+'.json')
        write(target,row)
        rows.append(dict(metric=metric, row=target.name, sha256=sha(target),profile=profile))
    common_native = ['benchmarks/heaptrack/drivers/cpu_profile/src/bin/iqa_compete.rs',
                     'crates/zenmetrics-cli/src/metrics/mod.rs']
    common_reference = [REPO/BENCH/n for n in ('prepare_ordered_canonical.py','compare_canonical.py','compare_canonical_budgeted.py')]
    registry = dict(native_revision=EA, verified_build=str(args.verified_build),
        verified_build_sha256=sha(args.verified_build), binary_sha256=verified['binary_sha256'],
        binary_path=str(args.verified_build.parent/'target/release/iqa-compete'),
        native_common=[native_source(p) for p in common_native],
        consumer_sources=[source(p,{}) for p in common_reference],
        prototype_disposition='first_boundary_hdrvdp.rs is uncompiled/superseded; no prototype or optional native probe supplies evidence for current40. Exact unchanged production bodies are compiled in the retained full worker.',
        limits='This registry verifies the inspected native snapshot files against retained receipt hashes. It is not a fresh exhaustive source verification, rebuild, transitive dependency closure or runtime snapshot. Canonical rows explicitly identify unbound observed sources.')
    write(args.output.parent/'source-registry.json',registry)
    write(args.output/'index.json',dict(detailed_reviewed_routes=40,total_routes=40,
        comparative_cases=80,manifest_sha256=index['manifest_sha256'],
        comparative_sha256=index['comparative_sha256'],rows=rows,
        inference_build_ci_or_gpu_maps_run=False, new_timing_or_numerical_qualification=False))
    assert len(rows)==40 and len({r['metric'] for r in rows})==40
    assert {r['metric'] for r in rows}=={c['metric'] for c in comparative['cases']}
    print(json.dumps(dict(detailed_reviewed_routes=40,total_routes=40,comparative_cases=80,
                         native_source_files=len(native_cache),output=str(args.output))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('selections','comparative','verified-build','output'):
        p.add_argument('--'+name,required=True,type=Path)
    main(p.parse_args())
