#!/usr/bin/env python3
"""Freeze caller-supplied NN exports and runtime files for the existing durable runner.

Preparation requires ONNX to discover external tensors; running the bundle is stdlib-only.
The inventory is captured on the binary's native host and bound by the build receipt.
"""
import argparse
import json
from pathlib import Path
import shutil
from mac_service import verify
from prepare_durable import HERE, sha
from run import LEARNED

NEURAL = LEARNED | {'ahiq', 'dmm'}


def model_files(directory, prefix):
    import onnx
    from onnx.external_data_helper import _get_all_tensors
    files = {p for p in directory.iterdir() if p.is_file() and p.name.startswith(prefix)
             and p.suffix in ('.onnx', '.meta', '.f32')}
    if directory / (prefix + '.meta') not in files:
        raise ValueError('missing model metadata: ' + prefix)
    for graph in sorted(files):
        if graph.suffix != '.onnx':
            continue
        model = onnx.load(graph, load_external_data=False)
        for tensor in _get_all_tensors(model):
            for item in tensor.external_data:
                if item.key == 'location':
                    name = Path(item.value)
                    if name.is_absolute() or '..' in name.parts:
                        raise ValueError('external tensor escapes model directory')
                    external = directory / name
                    external.resolve(strict=True)  # Source caches may use symlinks; copy and hash their bytes.
                    files.add(external)
    return files


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('template', type=Path)
    p.add_argument('binary', type=Path)
    p.add_argument('build', type=Path)
    p.add_argument('inventory', type=Path)
    p.add_argument('models', type=Path, help='JSON metric -> caller-supplied model directory')
    p.add_argument('output', type=Path)
    p.add_argument('--engine', choices=['rten', 'ort-webgpu'], required=True)
    p.add_argument('--ort-library', type=Path)
    p.add_argument('--webgpu-library', type=Path)
    p.add_argument('--accept-weights', default='', help='Explicit accepted weight terms, passed to the existing scorer')
    p.add_argument('--metrics', default='all')
    a = p.parse_args()
    verify(a.template.resolve())
    build = json.loads(a.build.read_bytes())
    if build['binary_sha256'] != sha(a.binary) or build['inventory_sha256'] != sha(a.inventory):
        raise ValueError('build receipt differs from binary/inventory')
    inventory = json.loads(a.inventory.read_bytes())
    eligible = {r['metric_name'] for r in inventory if r['compiled'] and r['metric_name'] in NEURAL}
    metrics = sorted(eligible) if a.metrics == 'all' else a.metrics.split(',')
    if not metrics or set(metrics) - eligible:
        raise ValueError('requested NN routes are not compiled')
    sources = json.loads(a.models.read_bytes())
    if set(metrics) - sources.keys():
        raise ValueError('missing caller-supplied model directories: ' + str(sorted(set(metrics) - sources.keys())))
    runtime = {}
    if a.engine == 'ort-webgpu':
        if not a.ort_library or not a.webgpu_library:
            raise ValueError('Metal requires explicit ORT core and WebGPU provider libraries')
        runtime = {'ORT_DYLIB_PATH': a.ort_library, 'ZENMETRICS_NN_WEBGPU_EP_LIB': a.webgpu_library}
    # Resolve every graph dependency before creating an output that could be mistaken for complete.
    exports = {m: model_files(Path(sources[m]).resolve(), m.replace('-', '_')) for m in metrics}
    shutil.copytree(a.template, a.output)
    shutil.copy2(a.binary, a.output / 'bin/iqa-compete')
    shutil.copy2(a.build, a.output / 'build.json')
    shutil.copy2(a.inventory, a.output / 'inventory.json')
    for name in ('durable.py', 'mac_service.py', 'select_durable.py', 'admission_recovery.py'):
        shutil.copy2(HERE / name, a.output / name)
    shutil.copy2(HERE / 'bundle.just', a.output / 'justfile')
    manifest = json.loads((a.template / 'manifest.json').read_bytes())
    inputs = dict(manifest['input_sha256'])
    runtime_env = {}
    runtime_assets = []
    for key, source in runtime.items():
        name = 'runtime/' + source.name
        target = a.output / name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(source, target)
        inputs[name] = sha(target)
        runtime_env[key] = name
        runtime_assets.append(name)
    per_model = {}
    for metric, files in exports.items():
        source_dir = Path(sources[metric]).resolve()
        names = []
        for source in sorted(files):
            name = 'models/' + metric + '/' + source.relative_to(source_dir).as_posix()
            target = a.output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            before = sha(source)
            shutil.copy2(source, target)
            if sha(target) != before or sha(source) != before:
                raise ValueError('model changed during freeze: ' + str(source))
            inputs[name] = before
            names.append(name)
        per_model[metric] = names
    templates = list({(c['size_name'], c['threads'], c['timing_phase'], c['cold_repeat']): c
                      for c in manifest['cells'] if c['only_mode'] == 'score'}.values())
    cells = []
    for template in templates:
        for metric in metrics:
            worker_env = dict(runtime_env, ZENMETRICS_NN_MODEL_DIR='models/' + metric,
                              ZENMETRICS_NN_ENGINE=a.engine, ZENMETRICS_NN_ACCEPT_WEIGHTS=a.accept_weights)
            if a.engine == 'ort-webgpu':
                worker_env['WGPU_BACKEND'] = 'metal'
            cells.append(dict(template, metric_name=metric, device='cpu' if a.engine == 'rten' else 'metal',
                              worker_environment=worker_env, asset_paths=per_model[metric] + runtime_assets))
    manifest.update(binary_sha256=sha(a.binary), input_sha256=inputs, cells=cells)
    manifest['metadata'].update(metrics=metrics, engine=a.engine, features=build['features'],
                                source_commit=build['source_commit'], build_receipt_sha256=sha(a.build),
                                maps='NN routes have no qualified native-map adapter; scalar-only schedule',
                                runtime='ORT WebGPU on Metal with CPU fallback disabled' if runtime else 'RTEN CPU')
    (a.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (a.output / 'SHA256SUMS.json').unlink()
    hashes = {path.relative_to(a.output).as_posix(): sha(path) for path in sorted(a.output.rglob('*')) if path.is_file()}
    (a.output / 'SHA256SUMS.json').write_text(json.dumps(hashes, indent=2) + '\n')
    verify(a.output.resolve())
    print(json.dumps(dict(bundle=str(a.output.resolve()), metrics=len(metrics), cells=len(cells), engine=a.engine)))


if __name__ == '__main__':
    main()
