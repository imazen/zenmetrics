"""Synthetic retained-file controls. No executable/library/model is ever run."""
import ast
import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import prepare_vmaf_fullcall_plan as verifier


class Fixture:
    def __init__(self, root):
        self.root = root
        self.bundle = root/'bundle'
        self.build = root/'build'
        self.controls = root/'controls'
        self.native_root = root/'native-build'
        for directory in (self.bundle, self.build, self.controls, self.native_root):
            directory.mkdir()
        self.output = root/'new-plan.json'
        self.native = self.file('native-build/target/release/iqa-compete', b'\x7fELFnative')
        self.old_binary = self.file('old-reference', b'\x7fELFold')
        self.binary = self.file('build/libvmaf-rgb-fullcall', b'\x7fELFfullcall')
        archive = self.file('original.a', b'!<arch>\nsynthetic')
        copied = self.file('build/inputs/libvmaf.a', archive.read_bytes())
        self.reference = self.file('reference.rgb', b'\x01'*(512*512*3))
        self.distorted = self.file('distorted.rgb', b'\x02'*(512*512*3))
        compiler = self.file('compiler', b'\x7fELFcompiler')
        dependency = self.file('libfixture.so', b'\x7fELFdependency')
        tools = {name: self.identity(self.file(name, b'\x7fELF'+name.encode()))
                 for name in ('cc1plus', 'collect2', 'as', 'ld')}
        runtime = {name: self.identity(self.file(name, b'\x7fELF'+name.encode()))
                   for name in ('python', 'shell', 'ldd')}
        self.source = root/'original-plan.json'
        original = dict(schema=1, binary=str(self.native), rust_source='b'*40,
                        bindings={}, cases=[])
        for path in (self.old_binary, self.reference, self.distorted):
            original['bindings'][str(path)] = verifier.sha(path)
        selected = []
        for metric in sorted(verifier.ROUTES):
            for threads in (1, 8):
                case = dict(id=f'{metric}-pair-cid22-t{threads}', metric=metric,
                    width=512, height=512, threads=threads, reference=str(self.reference),
                    distorted=str(self.distorted), canonical=dict(label='original', command=
                    [str(self.old_binary), '{metric}', '{width}', '{height}', '{reference}',
                     '{distorted}', '{threads}', '{samples}',
                     '601' if 'studio601' in metric else '709']),
                    absolute_bound=None if metric in ('vmaf-v1', 'vmaf-4k') else 0.02,
                    matched_scope=True, scope='historical model exclusion')
                original['cases'].append(case)
                selected.append(dict(id=case['id'], metric=metric, threads=threads,
                    width=512, height=512, original_plan_case=copy.deepcopy(case),
                    inputs=dict(reference=self.identity(self.reference),
                                distorted=self.identity(self.distorted)),
                    argv=[part.format_map(dict(case, samples=2))
                          for part in case['canonical']['command']]))
        self.write(self.source, original)
        self.write(self.bundle/'cases.json', selected)
        # Real reviewed source bytes are inert fixture data, not imports/evaluation.
        for name in verifier.SOURCE_HASHES:
            self.file('bundle/'+name, (Path(__file__).parent/name).read_bytes())
        headers = [('include-source', 'libvmaf/'+name) for name in
                   ('feature.h', 'model.h', 'picture.h', 'libvmaf.h', 'libvmaf_cuda.h')]
        headers += [('include-build', name) for name in ('vcs_version.h', 'libvmaf/version.h')]
        assets = dict(old_binary=self.identity(self.old_binary), library=self.identity(archive),
                      plan=self.identity(self.source))
        for folder, name in headers:
            header = self.file('bundle/'+folder+'/'+name, b'/* synthetic header */')
            assets[folder+':'+name] = self.identity(header)
        assets['old_source'] = self.identity(self.bundle/'libvmaf_rgb_compare.cpp')
        assets['compile_metadata'] = self.identity(self.file('compile_commands.json', b'[]'))
        assets['old_build_log'] = self.identity(self.file('old-build.log', b'synthetic old link'))
        self.frozen = dict(schema='vmaf-fullcall-freeze/1', source_revision='a'*40,
            files={str(p.relative_to(self.bundle)): verifier.sha(p)
                   for p in self.bundle.rglob('*') if p.is_file()}, assets=assets,
            tools=runtime, compiler=dict(self.identity(compiler), flags=verifier.FLAGS),
            threads=[1, 8], samples=2, child_timeout_seconds=1100)
        self.freeze = self.bundle/'freeze.json'
        self.write(self.freeze, self.frozen)
        expected_assets = {category+':'+key: record for category in ('assets', 'tools')
                           for key, record in self.frozen[category].items()}
        expected_assets['compiler'] = self.identity(compiler)
        for directory in (self.build, self.controls):
            for phase in ('before', 'after'):
                self.write(directory/f'assets-{phase}.json', expected_assets)
        for phase in ('before', 'after'):
            self.write(self.build/f'compiler-tools-{phase}.json', tools)
            for name, record in tools.items():
                self.process(self.build, f'compiler-tools-{phase}-{name}',
                    [str(compiler), '-print-prog-name='+name], self.build/'scratch', 1, 60,
                    record['path']+'\n')
        version = self.process(self.build, 'compiler-version', [str(compiler), '--version'],
                               self.build/'scratch', 1, 60, 'synthetic compiler\n')
        compile_argv = [str(compiler), *verifier.FLAGS, '-I', str(self.bundle/'include-source'),
            '-I', str(self.bundle/'include-build'), str(self.bundle/'libvmaf_rgb_fullcall.cpp'),
            str(copied), '-pthread', '-lm', '-o', str(self.binary)]
        compiled = self.process(self.build, 'compile', compile_argv, self.build/'scratch', 1, 120, '')
        roots = [str(compiler), str(self.old_binary), runtime['python']['path'],
                 runtime['shell']['path'], *[v['path'] for v in tools.values()]]
        self.snapshot(self.build/'runtime-before', roots, dependency)
        closure = self.snapshot(self.build/'runtime-after', roots+[str(self.binary)], dependency)
        for phase in ('before', 'after'):
            self.snapshot(self.controls/f'runtime-{phase}', roots+[str(self.binary)], dependency)
        self.build_path = self.build/'build-receipt.json'
        self.write(self.build_path, dict(schema='vmaf-fullcall-build/1', passed=True, error=None,
            freeze=str(self.freeze), freeze_sha256=verifier.sha(self.freeze),
            source_revision='a'*40, caller_admitted=True, numerical_or_timing_qualification=False,
            compiler_version=version, compile=compiled, binary=self.identity(self.binary),
            new_source_sha256=verifier.SOURCE_HASHES['libvmaf_rgb_fullcall.cpp'],
            old_source_sha256=verifier.SOURCE_HASHES['libvmaf_rgb_compare.cpp'],
            library_sha256=verifier.sha(archive), copied_library=self.identity(copied),
            compiler_tools=tools, runtime_files=closure))
        self.control_path = self.controls/'result.json'
        self.result = dict(schema='vmaf-fullcall-controls/1', passed=True, error=None,
            freeze=str(self.freeze), freeze_sha256=verifier.sha(self.freeze),
            source_revision='a'*40, build_receipt=str(self.build_path),
            build_receipt_sha256=verifier.sha(self.build_path), expected_comparisons=12,
            expected_children=24, expected_observations=72, samples_per_child=2,
            completed_comparisons=12, passed_comparisons=12, caller_admitted=True,
            numerical_bound_changed=False, timing_qualification=False,
            native_parity_qualification=False, effective_library_utilization_verified=False,
            cases=[])
        bits = struct.pack('>d', 0.75).hex()
        for case in selected:
            directory = self.controls/case['id']
            actual = dict(id=case['id'], metric=case['metric'], threads=case['threads'],
                          passed=True, errors=[], arms={}, exact_f64_bits=bits)
            self.write(directory/'inputs-before.json', case['inputs'])
            for arm, binary in [('old', self.old_binary), ('fullcall', self.binary)]:
                scope = 'SCOPE resident-rgb-full-frame'+('-model-lifetime' if arm=='fullcall' else '')
                stdout = 'VERSION 3.2.0\n'+scope+'\nCOLD 10 0.75\nSAMPLE 10 0.75\nSAMPLE 10 0.75\n'
                process = self.process(directory, arm, [str(binary), *case['argv'][1:]],
                    self.controls/'scratch', case['threads'], 1100, stdout,
                    stderr='benign retained diagnostic\n')  # No invented empty-stderr gate.
                actual['arms'][arm] = dict(process=process, binary=self.identity(binary),
                    protocol=dict(version='VERSION 3.2.0', observations=[dict(label=label,
                        elapsed_ns=10, score_text='0.75', score=0.75, f64_bits=bits)
                        for label in ('COLD', 'SAMPLE', 'SAMPLE')]))
                self.write(directory/(arm+'.inputs-after.json'), case['inputs'])
            path = directory/'case.json'
            self.write(path, actual)
            self.result['cases'].append(dict(id=case['id'], passed=True, errors=[],
                exact_f64_bits=bits, case=str(path), case_sha256=verifier.sha(path)))
        self.write(self.control_path, self.result)
        native_files = {}
        for name in verifier.NATIVE_FILES:
            path = self.file('native-build/work/zenmetrics/'+name, ('source '+name).encode())
            native_files[name] = verifier.sha(path)
        native_build = dict(source_revision='c'*40, binary_sha256=verifier.sha(self.native),
            adapters_applied=False, features=['compete', 'metrics-parallel', 'zenmetrics-cli/nr'],
            files=native_files)
        self.write(self.native_root/'build-receipt.json', native_build)
        self.file('native-build/cargo.log', b'synthetic cargo log')
        log = self.file('native-build.log', b'synthetic admitted wrapper')
        self.native_receipt = self.native_root/'verified-build.json'
        self.write(self.native_receipt, dict(source_revision='c'*40, files=native_files,
            binary_sha256=verifier.sha(self.native),
            build_receipt_sha256=verifier.sha(self.native_root/'build-receipt.json'),
            build_log_sha256=verifier.sha(log)))

    def file(self, name, data):
        path = self.root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

    @staticmethod
    def identity(path):
        return dict(path=str(path), resolved=str(path.resolve()), bytes=path.stat().st_size,
                    sha256=verifier.sha(path))

    def process(self, directory, stem, argv, scratch, threads, timeout, stdout, stderr='', rc=0):
        environment = {key: str(threads) for key in verifier.THREAD_VARIABLES}
        environment.update(OMP_DYNAMIC='FALSE', MKL_DYNAMIC='FALSE', TMPDIR=str(scratch),
                           LC_ALL='C', PATH='/usr/bin:/bin')
        self.write(directory/(stem+'.command.json'), dict(argv=argv, cwd=str(directory),
                   timeout_seconds=timeout, environment=environment))
        directory.joinpath(stem+'.stdout').write_text(stdout)
        directory.joinpath(stem+'.stderr').write_text(stderr)
        process = dict(argv=argv, returncode=rc, timed_out=False, launch_error=None,
                       interrupted=None)
        for key, suffix in [('command', '.command.json'), ('stdout', '.stdout'), ('stderr', '.stderr')]:
            process[key] = str(directory/(stem+suffix))
            process[key+'_sha256'] = verifier.sha(process[key])
        self.write(directory/(stem+'.process.json'), process)
        return process

    def snapshot(self, directory, roots, dependency):
        files, logs, queue = {}, [], list(roots)
        while queue:
            path = queue.pop(0)
            if path in files:
                continue
            files[path] = self.identity(Path(path))
            if Path(path).name == 'compiler':
                stdout = f'libfixture.so => {dependency} (0x1234)\n'
                queue.append(str(dependency))
            else:
                stdout = 'statically linked\n'
            logs.append(self.process(directory, f'ldd-{len(logs):03}',
                [str(self.root/'ldd'), str(Path(path).resolve())], directory.parent/'scratch',
                1, 60, stdout))
        self.write(directory/'snapshot.json', dict(roots=roots, files=files, ldd_logs=logs))
        return files

    def resign(self, actual):
        summary = next(r for r in self.result['cases'] if r['id'] == actual['id'])
        self.write(Path(summary['case']), actual)
        summary['case_sha256'] = verifier.sha(summary['case'])
        self.write(self.control_path, self.result)

    def first_case(self):
        return verifier.read(self.result['cases'][0]['case'])

    def prepare(self):
        return verifier.prepare(self.source, self.build_path, self.control_path, self.native,
            self.native_receipt, self.output, freeze_sha=verifier.sha(self.freeze),
            build_sha=verifier.sha(self.build_path), controls_sha=verifier.sha(self.control_path),
            native_receipt_sha=verifier.sha(self.native_receipt))


class RetainedControls(unittest.TestCase):
    def setUp(self):
        scratch = Path.home()/'tmp'
        scratch.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=scratch, prefix='vmaf-plan-fixture-')
        self.addCleanup(self.temporary.cleanup)
        self.f = Fixture(Path(self.temporary.name))

    def rejected(self):
        with self.assertRaises((ValueError, KeyError, FileNotFoundError, TypeError)):
            self.f.prepare()
        self.assertFalse(self.f.output.exists())

    def test_valid_scope_only_and_complete_evidence(self):
        self.f.prepare()
        plan = verifier.read(self.f.output)
        old = verifier.read(self.f.source)
        self.assertEqual(len(plan['cases']), 12)
        for case in plan['cases']:
            original = next(c for c in old['cases'] if (c['metric'], c['threads']) ==
                            (case['metric'], case['threads']))
            self.assertEqual(case['absolute_bound'], original['absolute_bound'])
            if original['absolute_bound'] is None:
                self.assertIsNone(case['absolute_bound'])
            self.assertEqual(case['canonical']['command'][1:], original['canonical']['command'][1:])
            self.assertEqual(case['reference'], original['reference'])
            self.assertEqual(case['distorted'], original['distorted'])
        expected = [p for directory in (self.f.build, self.f.controls) for p in directory.rglob('*')
                    if p.is_file()]
        self.assertTrue(all(str(p) in plan['bindings'] for p in expected))
        self.assertFalse(plan['timing_qualification'])
        self.assertFalse(plan['native_parity_qualification'])

    def test_malformed_native_json_source_is_opaque(self):
        relative = 'crates/vmaf/fixtures/intentionally-malformed.json'
        path = self.f.file('native-build/work/zenmetrics/'+relative, b'{ invalid JSON source fixture')
        native_build_path = self.f.native_root/'build-receipt.json'
        native_build = verifier.read(native_build_path)
        native_build['files'][relative] = verifier.sha(path)
        self.f.write(native_build_path, native_build)
        receipt = verifier.read(self.f.native_receipt)
        receipt['files'][relative] = verifier.sha(path)
        receipt['build_receipt_sha256'] = verifier.sha(native_build_path)
        self.f.write(self.f.native_receipt, receipt)
        self.f.prepare()
        plan = verifier.read(self.f.output)
        self.assertEqual(plan['bindings'][str(path)], verifier.sha(path))
        self.assertEqual(path.read_bytes(), b'{ invalid JSON source fixture')

    def test_forged_arm_names_even_with_resealed_receipt(self):
        case = self.f.first_case()
        case['arms']['pretend'] = case['arms'].pop('old')
        self.f.resign(case)
        self.rejected()

    def test_consistently_forged_argv(self):
        case = self.f.first_case()
        process = case['arms']['old']['process']
        process['argv'][6] = '28'
        command = verifier.read(process['command'])
        command['argv'] = process['argv']
        self.f.write(Path(process['command']), command)
        process['command_sha256'] = verifier.sha(process['command'])
        self.f.write(Path(process['command']).with_name('old.process.json'), process)
        self.f.resign(case)
        self.rejected()

    def test_forged_environment(self):
        case = self.f.first_case()
        process = case['arms']['fullcall']['process']
        command = verifier.read(process['command'])
        command['environment']['OMP_NUM_THREADS'] = '28'
        self.f.write(Path(process['command']), command)
        process['command_sha256'] = verifier.sha(process['command'])
        self.f.write(Path(process['command']).with_name('fullcall.process.json'), process)
        self.f.resign(case)
        self.rejected()

    def test_null_bits(self):
        case = self.f.first_case()
        for arm in case['arms'].values():
            for observation in arm['protocol']['observations']:
                observation['f64_bits'] = None
        self.f.resign(case)
        self.rejected()

    def test_bit_mismatch(self):
        case = self.f.first_case()
        case['arms']['old']['protocol']['observations'][0]['f64_bits'] = '0000000000000000'
        self.f.resign(case)
        self.rejected()

    def test_raw_nonfinite_with_updated_log_hash(self):
        case = self.f.first_case()
        process = case['arms']['old']['process']
        path = Path(process['stdout'])
        path.write_text(path.read_text().replace('0.75', '1e999'))
        process['stdout_sha256'] = verifier.sha(path)
        self.f.write(path.with_name('old.process.json'), process)
        self.f.resign(case)
        self.rejected()

    def test_process_outcome(self):
        case = self.f.first_case()
        process = case['arms']['old']['process']
        process['interrupted'] = 'SIGTERM'
        self.f.write(Path(process['stdout']).with_name('old.process.json'), process)
        self.f.resign(case)
        self.rejected()

    def test_missing_historical_input_snapshot(self):
        path = Path(self.f.result['cases'][0]['case']).parent/'old.inputs-after.json'
        path.unlink()
        self.rejected()

    def test_altered_runtime_transcript(self):
        path = self.f.controls/'runtime-after/ldd-000.stdout'
        path.write_text('libfixture.so => not found\n')
        self.rejected()

    def test_missing_compiler_query_receipt(self):
        (self.f.build/'compiler-tools-after-ld.process.json').unlink()
        self.rejected()

    def test_input_drift(self):
        self.f.reference.write_bytes(b'wrong pixels')
        self.rejected()

    def test_copied_archive_drift(self):
        (self.f.build/'inputs/libvmaf.a').write_bytes(b'changed archive')
        self.rejected()

    def test_original_archive_drift(self):
        (self.f.root/'original.a').write_bytes(b'changed original archive')
        self.rejected()

    def test_asset_snapshot_drift(self):
        path = self.f.controls/'assets-after.json'
        value = verifier.read(path)
        value['assets:library']['sha256'] = '0'*64
        self.f.write(path, value)
        self.rejected()

    def test_raw_scope_mismatch_with_resealed_log(self):
        case = self.f.first_case()
        process = case['arms']['fullcall']['process']
        path = Path(process['stdout'])
        path.write_text(path.read_text().replace('-model-lifetime', ''))
        process['stdout_sha256'] = verifier.sha(path)
        self.f.write(path.with_name('fullcall.process.json'), process)
        self.f.resign(case)
        self.rejected()

    def test_duplicate_and_overflow_json(self):
        original = self.f.control_path.read_text()
        for text in ('{"passed":true,"passed":true}', '{"value":1e999}', '{"value":NaN}'):
            self.f.control_path.write_text(text)
            self.rejected()
        self.f.control_path.write_text(original)

    def test_existing_output_refused_unchanged(self):
        self.f.output.write_bytes(b'keep these bytes')
        with self.assertRaises(ValueError):
            self.f.prepare()
        self.assertEqual(self.f.output.read_bytes(), b'keep these bytes')

    def test_atomic_link_failure_leaves_no_partial_file(self):
        with patch.object(verifier.os, 'link', side_effect=OSError('synthetic link failure')):
            with self.assertRaises(OSError):
                self.f.prepare()
        self.assertFalse(self.f.output.exists())
        self.assertEqual(list(self.f.root.glob('.new-plan.json.*')), [])

    def test_atomic_publication_race_preserves_other_publisher(self):
        original_link = verifier.os.link

        def competing_link(source, target):
            Path(target).write_bytes(b'other publisher')
            original_link(source, target)

        with patch.object(verifier.os, 'link', side_effect=competing_link):
            with self.assertRaises(FileExistsError):
                self.f.prepare()
        self.assertEqual(self.f.output.read_bytes(), b'other publisher')
        self.assertEqual(list(self.f.root.glob('.new-plan.json.*')), [])

    def test_optimized_python_has_no_assertion_bypass(self):
        tree = ast.parse(Path(verifier.__file__).read_text())
        self.assertFalse(any(isinstance(node, ast.Assert) for node in ast.walk(tree)))


if __name__ == '__main__':
    unittest.main()
