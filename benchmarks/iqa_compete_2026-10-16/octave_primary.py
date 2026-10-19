#!/usr/bin/env python3
"""Run a verified primary FSIM/VSI fixed-factor copy on resident RGB8 inputs.

This extends the IQA harness's native-reference adapters. Callers provide the
host heavy locks and resource wrapper. Output is diagnostic until matched
source/runtime, Rust parity and paired acquisition qualification is recorded.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess

from check_fixed_factor_oracle import check_manifest
from supervision import scoring_command
import image_resize_source


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def quoted(value):
    text = str(value)
    if any(c in text for c in ['\n', '\r', '\0']):
        raise ValueError('control character in Octave argument')
    return "'" + text.replace("'", "''") + "'"


def process_max_rss(path):
    rows = [line.split(':', 1)[1].strip() for line in path.read_text().splitlines()
            if line.strip().startswith('Maximum resident set size (kbytes):')]
    if len(rows) != 1 or int(rows[0]) <= 0:
        raise ValueError('missing or invalid GNU time peak-RSS receipt')
    return int(rows[0])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--reference', required=True, type=Path)
    p.add_argument('--distorted', required=True, type=Path)
    p.add_argument('--width', required=True, type=int)
    p.add_argument('--height', required=True, type=int)
    p.add_argument('--threads', required=True, type=int)
    p.add_argument('--samples', required=True, type=int)
    p.add_argument('--evidence', required=True, type=Path)
    p.add_argument('--octave', default='octave-cli')
    p.add_argument('--timeout', type=float, default=300)
    p.add_argument('--measure-memory', action='store_true', help='retain /usr/bin/time -v process peak RSS')
    p.add_argument('--image-archive', type=Path, help='select the explicit hash-bound upstream resize M-file overlay')
    a = p.parse_args()
    if min(a.width, a.height, a.threads, a.samples) <= 0 or a.timeout <= 0:
        p.error('dimensions, threads, samples and timeout must be positive')
    check_manifest(a.manifest)
    manifest = json.loads(a.manifest.read_text())
    names = {'vsi-matlab': ('VSI', 1), 'fsim-matlab': ('FeatureSIM', 2)}
    if manifest['source']['name'] not in names:
        p.error('only primary MATLAB FSIM/VSI source copies are supported')
    function, columns = names[manifest['source']['name']]
    source = Path(manifest['output']['path']).resolve(strict=True)
    if source.name != function + '.m':
        p.error(f'Octave requires the function filename {function}.m')
    inputs = [a.reference.resolve(strict=True), a.distorted.resolve(strict=True)]
    if any(path.stat().st_size != a.width * a.height * 3 for path in inputs):
        p.error('exact RGB8 input length required')
    before = [sha(path) for path in inputs]
    source_before = sha(source)
    executable = Path(shutil.which(a.octave) or a.octave).resolve(strict=True)
    if a.image_archive:
        a.image_archive = a.image_archive.resolve(strict=True)
        image_resize_source.validate_archive(a.image_archive)
    a.evidence.mkdir()  # Fresh evidence only; no overwrite of earlier attempts.
    evidence = a.evidence.resolve()
    scratch = evidence / 'scratch'
    scratch.mkdir()
    resize_receipt = None
    resize_setup = ''
    if a.image_archive:
        overlay = evidence / 'image-resize-overlay'
        resize_receipt = image_resize_source.prepare(a.image_archive, overlay)
        resize_setup = (f"addpath({quoted(overlay)},'-begin'); clear imresize;\n"
                        f"assert(strcmp(which('imresize'),{quoted(overlay / 'imresize.m')}));")
    output = f'[score1,score2] = {function}(reference,distorted);' if columns == 2 else f'score1 = {function}(reference,distorted);'
    release = 'clear reference distorted;'
    emit = "fprintf('%s %.0f %.17g %.17g\\n',label,elapsed,score1,score2);" if columns == 2 else "fprintf('%s %.0f %.17g\\n',label,elapsed,score1);"
    script = f"""more off;
pkg load image;
{resize_setup}
addpath({quoted(source.parent)},'-begin');
assert(strcmp(which('{function}'),{quoted(source)}));
fprintf('VERSION octave-%s primary-{function}\\n',version());
fprintf('SCOPE rgb-input\\n');
fprintf('SOURCE %s\\n',which('{function}'));
for name = {{'imresize','fspecial','conv2','fft2','ifft2'}}
  fprintf('BIND %s %s\\n',name{{1}},which(name{{1}}));
end
fftw('threads',{a.threads});
assert(fftw('threads') == {a.threads});
fprintf('FFTW_THREADS %d\\n',fftw('threads'));
f = fopen({quoted(inputs[0])},'rb'); assert(f >= 0);
[raw_ref,count] = fread(f,Inf,'*uint8'); assert(count == {a.width*a.height*3}); assert(fclose(f) == 0);
f = fopen({quoted(inputs[1])},'rb'); assert(f >= 0);
[raw_dist,count] = fread(f,Inf,'*uint8'); assert(count == {a.width*a.height*3}); assert(fclose(f) == 0);
for call = 0:{a.samples}
  clock = tic();
  reference = permute(reshape(raw_ref,[3,{a.width},{a.height}]),[3,2,1]);
  distorted = permute(reshape(raw_dist,[3,{a.width},{a.height}]),[3,2,1]);
  {output}
  {release}
  elapsed = toc(clock)*1e9;
  label = 'SAMPLE'; if call == 0; label = 'COLD'; end
  {emit}
  fflush(stdout);
end
"""
    entry = evidence / 'oracle_entry.m'
    entry.write_text(script)
    env = os.environ.copy()
    env['TMPDIR'] = str(scratch)
    # A reference scorer has no display work. Keep inherited desktop sockets
    # outside this explicitly headless cohort.
    for key in ['DISPLAY', 'WAYLAND_DISPLAY', 'XAUTHORITY']:
        env.pop(key, None)
    for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS']:
        env[key] = str(a.threads)
    if a.measure_memory:
        env['LC_ALL'] = 'C'
    command = [str(executable), '--no-gui', '--no-window-system', '--no-history', '--no-init-file', '--quiet', str(entry)]
    command = scoring_command(command, system=platform.system(), seconds=a.timeout,
                              memory_log=evidence / 'process-memory.txt' if a.measure_memory else None)
    record = {'command': command, 'manifest_sha256': sha(a.manifest),
              'source_sha256': source_before, 'input_sha256': before,
              'width': a.width, 'height': a.height, 'threads_requested': a.threads,
              'executable_sha256': sha(executable), 'adapter_sha256': sha(Path(__file__)),
              'image_resize_overlay': resize_receipt,
              'headless': {'no_window_system': True, 'display_environment_removed': True},
              'environment': {k: env[k] for k in env if k.endswith('NUM_THREADS')},
              'memory_tool_locale': env.get('LC_ALL') if a.measure_memory else None,
              'qualification': 'diagnostic canonical execution; no Rust comparison or qualified ratio'}
    (evidence / 'attempt.json').write_text(json.dumps(record, indent=2) + '\n')
    with (evidence / 'stdout.log').open('w') as stdout, (evidence / 'stderr.log').open('w') as stderr:
        result = subprocess.run(command, cwd=evidence, env=env, stdout=stdout, stderr=stderr)
    record['returncode'] = result.returncode
    (evidence / 'completion.json').write_text(json.dumps(record, indent=2) + '\n')
    if result.returncode:
        raise RuntimeError(f'Octave failed rc={result.returncode}; see {evidence}/stderr.log')
    if [sha(path) for path in inputs] != before or sha(source) != source_before:
        raise RuntimeError('input or source changed during oracle execution')
    if a.image_archive:
        image_resize_source.verify(a.image_archive, overlay, resize_receipt)
    check_manifest(a.manifest)
    text = (evidence / 'stdout.log').read_text()
    rows = [line.split() for line in text.splitlines() if line.startswith(('COLD ', 'SAMPLE '))]
    if len(rows) != a.samples + 1 or rows[0][0] != 'COLD' or any(r[0] != 'SAMPLE' for r in rows[1:]):
        raise RuntimeError('missing or unexpected canonical call records')
    if any(len(row) != columns + 2 or float(row[1]) <= 0 for row in rows):
        raise RuntimeError('invalid canonical call record')
    bindings = []
    for line in text.splitlines():
        if line.startswith('BIND '):
            _, name, path = line.split(' ', 2)
            resolved = Path(path)
            bindings.append({'name': name, 'path': path,
                             'sha256': sha(resolved) if resolved.is_absolute() else None,
                             'kind': 'file' if resolved.is_absolute() else 'compiled builtin'})
    record['direct_function_bindings'] = bindings
    record['stdout_sha256'] = sha(evidence / 'stdout.log')
    record['stderr_sha256'] = sha(evidence / 'stderr.log')
    record['entry_sha256'] = sha(entry)
    if a.measure_memory:
        record['process_memory_sha256'] = sha(evidence / 'process-memory.txt')
        record['process_max_rss_kbytes'] = process_max_rss(evidence / 'process-memory.txt')
    record['score_rows'] = [{'phase': row[0], 'score_strings': row[2:],
                             'all_finite': all(math.isfinite(float(v)) for v in row[2:])}
                            for row in rows]
    # Preserve nonfinite canonical results as findings, never successful parity.
    record['executable_sha256_after'] = sha(executable)
    if record['executable_sha256_after'] != record['executable_sha256']:
        raise RuntimeError('Octave executable changed during oracle execution')
    (evidence / 'results.json').write_text(json.dumps(record, indent=2) + '\n')
    print(text, end='')


if __name__ == '__main__':
    main()
