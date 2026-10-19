#!/usr/bin/env python3
"""Build unchanged pinned libgmsd plus raw-RGB oracle/map probes, test-only."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

ZIP_SHA = '0da2b55bf4ca87a8951985e777d16ef796e091e9ffb7958badc9664793c37cc4'
REV = 'de646c9a957a892e4b9d78ff94601b83b363c158'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['archive', 'rust-build', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    a.archive = a.archive.resolve(strict=True)
    a.rust_build = a.rust_build.resolve(strict=True)
    assert sha(a.archive) == ZIP_SHA
    original = json.loads((a.rust_build / 'build-receipt.json').read_text())
    assert original['source_revision'] == 'e2d414c33a6d3c7da50a818fa36e56654d3ebeef'
    crate = a.rust_build / 'work/zenmetrics/crates/gmsd'
    repo = Path(__file__).resolve().parents[2]
    for path in crate.rglob('*'):
        if path.is_file():
            relative = path.relative_to(a.rust_build / 'work/zenmetrics')
            assert subprocess.check_output(['git', '-C', str(repo), 'show', original['source_revision'] + ':' + relative.as_posix()]) == path.read_bytes()
    a.output.mkdir()
    out = a.output.resolve()
    with zipfile.ZipFile(a.archive) as archive:
        for item in archive.infolist():
            path = Path(item.filename)
            assert path.parts[0] == 'libgmsd-' + REV and '..' not in path.parts and not path.is_absolute()
            if item.is_dir():
                (out / path).mkdir(parents=True, exist_ok=True)
            else:
                (out / path).parent.mkdir(parents=True, exist_ok=True)
                (out / path).write_bytes(archive.read(item))
    source = out / ('libgmsd-' + REV)
    base = Path(__file__).resolve().parent
    commands = []
    cc = Path(shutil.which('cc')).resolve(strict=True)
    rustc = Path(subprocess.check_output(['rustup', 'which', 'rustc'], text=True).strip()).resolve(strict=True)
    common = [str(cc), '-std=gnu11', '-DNDEBUG', '-ffp-contract=off', '-I' + str(source / 'include'),
              '-I' + str(source / '3rdparty'), '-I' + str(source / 'bin'),
              str(base / 'gmsd_primary_compute.c'), *map(str, sorted((source / 'src').glob('*.c'))),
              str(source / '3rdparty/bmpread.c'), '-lm']
    for name, flags in [('gmsd-primary', ['-O3']), ('gmsd-primary-asan', ['-O1', '-g', '-fsanitize=address,undefined', '-fno-omit-frame-pointer'])]:
        commands.append([*common, *flags, '-o', str(out / name)])
    deps = a.rust_build / 'target/release/deps'
    libs = {}
    for name in ['gmsd', 'rayon']:
        matches = list(deps.glob('lib' + name + '-*.rlib'))
        assert len(matches) == 1
        libs[name] = matches[0]
    commands.append([str(rustc), '--edition=2024', '-C', 'opt-level=3', '-L', 'dependency=' + str(deps),
                     *[arg for name, path in libs.items() for arg in ['--extern', name + '=' + str(path)]],
                     str(base / 'gmsd_primary_dump.rs'), '-o', str(out / 'gmsd-rust-map')])
    for command in commands:
        print('COMMAND', json.dumps(command), flush=True)
        subprocess.run(command, check=True)
    with zipfile.ZipFile(a.archive) as archive:
        assert all((out / item.filename).read_bytes() == archive.read(item) for item in archive.infolist() if not item.is_dir())
    identity = dict(source_revision=REV, archive_sha256=ZIP_SHA, rust_source_revision=original['source_revision'],
                    original_build_receipt_sha256=sha(a.rust_build / 'build-receipt.json'), commands=commands,
                    helpers={name: sha(base / name) for name in ['build_gmsd_primary.py', 'gmsd_primary_compute.c', 'gmsd_primary_dump.rs']},
                    compilers={str(path): sha(path) for path in [cc, rustc]},
                    rust_rlibs={str(path): sha(path) for path in libs.values()},
                    binaries={name: sha(out / name) for name in ['gmsd-primary', 'gmsd-primary-asan', 'gmsd-rust-map']},
                    rust_crate_files={str(path.relative_to(crate)): sha(path) for path in crate.rglob('*') if path.is_file()},
                    native_cpu_flags='none; exact compiler arguments retained', qualified_timing_cells=0)
    (out / 'build-receipt.json').write_text(json.dumps(identity, indent=2) + '\n')
    print(json.dumps(identity), flush=True)


if __name__ == '__main__':
    main()
