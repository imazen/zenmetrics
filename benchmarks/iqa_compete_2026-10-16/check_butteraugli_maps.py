#!/usr/bin/env python3
"""Build accuracy-only probes against frozen libraries and compare raw maps."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

import numpy as np

from compare_canonical import sha
from haarpsi_primary import execute


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('native_build', 'reference_build', 'pairs', 'output'):
        p.add_argument('--'+name.replace('_', '-'), type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir()
    root = a.output.resolve()
    here = Path(__file__).resolve().parent
    deps = a.native_build.resolve()/'target/release/deps'
    libraries = {}
    for name in ('butteraugli', 'rayon'):
        candidates = list(deps.glob('lib'+name+'-*.rlib'))
        if len(candidates) != 1:
            raise ValueError('Expected one frozen library: '+name)
        libraries[name] = candidates[0]
    rust = root/'butteraugli-native-map'
    rust_command = ['rustc', '--edition=2024', '-C', 'opt-level=3', '-L', 'dependency='+str(deps),
                    *[v for name,path in libraries.items() for v in ('--extern', name+'='+str(path))],
                    str(here/'butteraugli_primary_dump.rs'), '-o', str(rust)]
    subprocess.run(rust_command, check=True)
    reference = a.reference_build.resolve()
    commands = json.loads((reference/'compile_commands.json').read_text())
    entry = next(c for c in commands if Path(c['file']).name == 'libjxl_compute.cpp')
    argv = shlex.split(entry['command'])
    original_object = argv[argv.index('-o')+1]
    obj = root/'libjxl-map.o'
    argv[argv.index('-o')+1] = str(obj)
    argv[argv.index('-c')+1] = str(here/'libjxl_map_dump.cpp')
    subprocess.run(argv, cwd=entry['directory'], check=True)
    link_file = reference/'CMakeFiles/iqa-jxl.dir/link.txt'
    if link_file.is_file():
        link = shlex.split(link_file.read_text())
    else:
        if 'CMAKE_GENERATOR:INTERNAL=Ninja' not in (reference/'CMakeCache.txt').read_text():
            raise ValueError('Expected an explicit Makefiles or Ninja reference build')
        commands = subprocess.check_output(['ninja','-C',str(reference),'-t','commands','iqa-jxl'], text=True).splitlines()
        links = [line for line in commands if ' -o iqa-jxl ' in line]
        if len(links) != 1:
            raise ValueError('Expected one captured Ninja link command')
        link = shlex.split(links[0])
        if link[:2] != [':','&&'] or link[-2:] != ['&&',':']:
            raise ValueError('Unexpected Ninja command wrapper')
        link = link[2:-2]
        if any(token in ('&&',';','|') for token in link):
            raise ValueError('Unexpected shell operation in captured link command')
    if link.count(original_object) != 1:
        raise ValueError('Reference link command differs from captured compilation')
    link[link.index(original_object)] = str(obj)
    cpp = root/'butteraugli-reference-map'
    link[link.index('-o')+1] = str(cpp)
    subprocess.run(link, cwd=reference, check=True)
    policy = a.native_build/'butteraugli/butteraugli/tests/reference_parity.rs'
    features = list((a.native_build/'target/release/.fingerprint').glob('butteraugli-*/lib-butteraugli.json'))
    if len(features) != 1 or 'iir-blur' in json.loads(features[0].read_text())['features']:
        raise ValueError('Existing scalar policy requires the frozen FIR build')
    policy_text = policy.read_text()
    if 'const SCORE_TOLERANCE: f64 = 0.001;' not in policy_text or 'case.expected_score > 0.001' not in policy_text:
        raise ValueError('Existing scalar policy changed; review before using it')
    link_inputs = {}
    for item in link:
        candidate = reference/item
        if not item.startswith('-') and candidate.is_file():
            link_inputs[str(candidate.resolve())] = sha(candidate)
    receipt = dict(native_library_sha256={name:sha(path) for name,path in libraries.items()},
        rust_command=rust_command, cpp_compile=argv, cpp_link=link, cpp_link_inputs_sha256=link_inputs,
        native_probe_sha256=sha(rust), reference_probe_sha256=sha(cpp),
        probe_sources={path.name:sha(path) for path in [here/'butteraugli_primary_dump.rs', here/'libjxl_map_dump.cpp', here/'libjxl_compute.cpp']},
        policy_source_sha256=sha(policy), native_features=json.loads(features[0].read_text())['features'],
        input_manifest_sha256=sha(a.pairs), worker_sha256=sha(__file__))
    (root/'build-receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    rows = []
    for pair in json.loads(a.pairs.read_text())['pairs']:
        w, h = pair['width'], pair['height']
        paths = {role:(a.pairs.parent/pair[role]).resolve(strict=True) for role in ('reference','distorted')}
        for role,path in paths.items():
            if sha(path) != pair[role+'_sha256']:
                raise ValueError('Input changed: '+str(path))
        for threads in (1, 28):
            directory = root/f'{pair["pair"]}-{pair["size_name"]}-t{threads}'
            directory.mkdir()
            env = dict(os.environ, TMPDIR=str(directory))
            for key in ('RAYON_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
                env[key] = str(threads)
            scores = {}
            maps = {}
            for role,binary in [('native',rust),('canonical',cpp)]:
                destination = directory/(role+'.f32')
                command = [str(binary),str(w),str(h),str(paths['reference']),str(paths['distorted']),str(threads),str(destination)]
                rc,_ = execute(command, env, directory, role, 1100)
                if rc:
                    raise ValueError('Map probe failed; retained '+str(directory))
                line = next(line for line in (directory/(role+'-stdout.log')).read_text().splitlines() if line.startswith('BUTTER_MAP ')).split()
                if list(map(int,line[2:])) != [w,h]:
                    raise ValueError('Output geometry differs')
                scores[role] = float(line[1])
                maps[role] = np.fromfile(destination, dtype='<f4').reshape(h,w)
            finite = np.isfinite(maps['native']) & np.isfinite(maps['canonical'])
            matching = (np.isnan(maps['native']) & np.isnan(maps['canonical'])) | (np.isinf(maps['native']) & (maps['native']==maps['canonical']))
            delta = np.abs(maps['native'][finite].astype(np.float64)-maps['canonical'][finite])
            scalar_delta = abs(scores['native']-scores['canonical'])
            normalized = scalar_delta/scores['canonical'] if scores['canonical'] > .001 else scalar_delta
            row = dict(pair=pair['pair'],size=pair['size_name'],width=w,height=h,threads=threads,
                scores=scores,scalar_delta=scalar_delta,scalar_policy='relative <= 0.001 above canonical 0.001, absolute <= 0.001 otherwise; existing FIR reference_parity.rs',
                scalar_pass=bool(np.isfinite(normalized) and normalized<=.001), map_bound=None,
                finite_pixels=int(finite.sum()),nonfinite_mismatches=int((~finite & ~matching).sum()),
                max_map_abs_delta=float(delta.max()) if delta.size else None,
                mean_map_abs_delta=float(delta.mean()) if delta.size else None,
                files={path.name:sha(path) for path in directory.iterdir() if path.is_file()})
            (directory/'results.json').write_text(json.dumps(row,indent=2)+'\n')
            rows.append(row)
            print('MAP_CASE',directory.name,row['scalar_pass'],row['max_map_abs_delta'],row['nonfinite_mismatches'],flush=True)
    (root/'completion.json').write_text(json.dumps(dict(cases=rows,build_receipt_sha256=sha(root/'build-receipt.json'),qualification='Accuracy-only scalar and map observations; no timing ratio or new map bound'),indent=2)+'\n')


if __name__ == '__main__':
    main()
