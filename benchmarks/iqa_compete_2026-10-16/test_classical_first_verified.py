#!/usr/bin/env python3
"""Inert differential validator controls; no scorer or test worker is executed."""
import ast
from collections import Counter
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import compare_classical_first as old
import compare_classical_first_verified as new
from compare_canonical import sha


class SourceVerificationTests(unittest.TestCase):
    def scratch(self):
        # Caller must explicitly provide the disk-backed scratch directory.
        return tempfile.TemporaryDirectory(prefix='first-verified-', dir=os.environ['TMPDIR'])

    def fixture(self, root):
        native = root / 'native.txt'
        canonical = root / 'canonical.txt'
        image = root / 'pixels.rgb'
        extra = root / 'parameter-source.txt'
        for path, data in [(native, b'inert native artifact'),
                           (canonical, b'inert canonical artifact'),
                           (image, bytes(64 * 64 * 3)), (extra, b'inert source bytes')]:
            path.write_bytes(data)
        sources = []
        cases = []
        bindings = {str(p): sha(p) for p in (native, canonical, image, extra)}
        routes = []
        asset_bindings = bindings.copy()
        for number in range(2):
            metric = 'inert-' + str(number)
            originals = []
            for index in range(3):
                originals.append(dict(id=f'{metric}-{index}', metric=metric,
                    width=64, height=64, threads=1, reference=str(image), distorted=str(image),
                    canonical=dict(label='inert', command=[str(canonical), '{samples}']),
                    absolute_bound=None, relative_bound=0, matched_scope=True, scope='inert'))
            source = root / f'source-{number}.json'
            source.write_text(json.dumps(dict(binary=str(native), rust_source='inert-source',
                cases=originals, bindings=asset_bindings.copy())))
            sources.append(source)
            bindings[str(source)] = sha(source)
            for original in originals:
                cases.append(dict(copy.deepcopy(original), source_plan=str(source),
                    source_case_id=original['id'], binary=str(native), rust_source='inert-source',
                    first_call_scope=dict(matched=True, boundary=old.PHASE,
                        construction='inert', parameter_io='inert', runtime='inert',
                        teardown='inert', evidence=['inert source'])))
            routes.append(dict(metric=metric, classification='first_evaluation_candidate',
                canonical_emits_cold=True, canonical_cold_is_first_pair_evaluation=True,
                adapter_pair_evaluations_before_first_timed_observation=0,
                canonical_command=originals[0]['canonical']['command']))
        review = root / 'review.json'
        review.write_text(json.dumps(dict(routes=routes)))
        required = [review, Path(sys.executable), Path('/usr/bin/time'), Path('/usr/bin/timeout'),
                    Path('/usr/bin/flock'), Path.home() / 'work/claudehints/scripts/run-heavy']
        required += [Path(new.__file__).parent / name for name in new.HELPERS]
        bindings.update({str(p): sha(p) for p in required})
        plan = root / 'plan.json'
        plan.write_text(json.dumps(dict(phase=old.PHASE, seconds=5,
            review=str(review), bindings=bindings, cases=cases)))
        return dict(plan=plan, sources=sources, review=review, native=native,
                    canonical=canonical, image=image, extra=extra)

    def edit_plan(self, fixture, edit):
        plan = json.loads(fixture['plan'].read_text())
        edit(plan)
        fixture['plan'].write_text(json.dumps(plan))

    def edit_source(self, fixture, edit, matching_cases=False):
        path = fixture['sources'][0]
        source = json.loads(path.read_text())
        edit(source)
        path.write_text(json.dumps(source))
        def update(plan):
            plan['bindings'][str(path)] = sha(path)
            if matching_cases:
                for original in source['cases']:
                    for case in plan['cases']:
                        if case['source_plan'] == str(path) and case['source_case_id'] == original['id']:
                            case.update(copy.deepcopy(original))
        self.edit_plan(fixture, update)

    def outcome(self, module, fixture, expected=None, with_scope=False):
        try:
            plan, routes = module.verify(fixture['plan'], expected or sha(fixture['plan']))
            if with_scope and any(module.scope_reasons(c, routes[c['metric']])
                                  for c in plan['cases']):
                return False
            return True
        except Exception:
            return False

    def reject_both(self, fixture):
        self.assertFalse(self.outcome(old, fixture))
        self.assertFalse(self.outcome(new, fixture))

    def test_valid_repeated_sources_and_multiple_routes_agree(self):
        with self.scratch() as tmp:
            fixture = self.fixture(Path(tmp))
            expected = sha(fixture['plan'])
            self.assertEqual(old.verify(fixture['plan'], expected),
                             new.verify(fixture['plan'], expected))
            self.assertTrue(self.outcome(new, fixture, with_scope=True))

    def test_each_invocation_hashes_every_binding_in_the_original_order(self):
        with self.scratch() as tmp:
            fixture = self.fixture(Path(tmp))
            plan = json.loads(fixture['plan'].read_text())
            expected_trace = [str(fixture['plan']), *plan['bindings']]
            for module in (old, new):
                traces = []
                for invocation in range(2):
                    trace = []
                    def recorded(path):
                        trace.append(str(path))
                        return sha(path)
                    with patch.object(module, 'sha', side_effect=recorded):
                        module.verify(fixture['plan'], sha(fixture['plan']))
                    self.assertEqual(trace, expected_trace, (module.__name__, invocation))
                    traces.append(trace)
                self.assertEqual(traces[0], traces[1])

    def test_source_json_read_once_per_path_per_invocation(self):
        with self.scratch() as tmp:
            fixture = self.fixture(Path(tmp))
            real = Path.read_text
            for module, per_invocation in ((old, 3), (new, 1)):
                reads = Counter()
                def recorded(path, *args, **kwargs):
                    reads[str(path)] += 1
                    return real(path, *args, **kwargs)
                for invocation in range(2):
                    with patch.object(Path, 'read_text', new=recorded):
                        module.verify(fixture['plan'], sha(fixture['plan']))
                    for source in fixture['sources']:
                        self.assertEqual(reads[str(source)], per_invocation * (invocation + 1))

    def test_source_binding_contract_map_checked_once_per_source(self):
        with self.scratch() as tmp:
            fixture = self.fixture(Path(tmp))
            real_loads = json.loads
            for module, expected in ((old, 3), (new, 1)):
                iterations = Counter()
                class CountedBindings(dict):
                    def __init__(self, data, identity):
                        super().__init__(data)
                        self.identity = identity
                    def items(self):
                        iterations[self.identity] += 1
                        return super().items()
                def loaded(text, *args, **kwargs):
                    value = real_loads(text, *args, **kwargs)
                    if isinstance(value, dict) and 'binary' in value and 'cases' in value:
                        identity = value['cases'][0]['metric']
                        value['bindings'] = CountedBindings(value['bindings'], identity)
                    return value
                with patch.object(module.json, 'loads', side_effect=loaded):
                    module.verify(fixture['plan'], sha(fixture['plan']))
                self.assertEqual(dict(iterations), {'inert-0': expected, 'inert-1': expected})

    def test_no_cross_invocation_or_mtime_cache(self):
        with self.scratch() as tmp:
            fixture = self.fixture(Path(tmp))
            self.assertTrue(self.outcome(new, fixture))
            source = fixture['sources'][0]
            stat = source.stat()
            # Parent digest is updated, but its case stays unchanged. A cached
            # old parsed source would accept despite preserving final hashes.
            self.edit_source(fixture, lambda s: s['cases'][0].update(scope='changed'))
            os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            self.reject_both(fixture)

    def test_content_tamper_source_native_input_canonical_and_extra(self):
        for key in ('native', 'image', 'canonical', 'extra'):
            with self.subTest(key=key), self.scratch() as tmp:
                fixture = self.fixture(Path(tmp))
                fixture[key].write_bytes(fixture[key].read_bytes() + b'changed')
                self.reject_both(fixture)
        with self.scratch() as tmp:
            fixture = self.fixture(Path(tmp))
            path = fixture['sources'][0]
            path.write_text(path.read_text() + '\n')
            self.reject_both(fixture)

    def test_plan_tamper_rejects_old_digest(self):
        with self.scratch() as tmp:
            fixture = self.fixture(Path(tmp))
            expected = sha(fixture['plan'])
            fixture['plan'].write_text(fixture['plan'].read_text() + '\n')
            self.assertFalse(self.outcome(old, fixture, expected))
            self.assertFalse(self.outcome(new, fixture, expected))

    def test_required_and_source_binding_omissions_reject(self):
        for key in ('native', 'image', 'extra', 'review'):
            with self.subTest(key=key), self.scratch() as tmp:
                fixture = self.fixture(Path(tmp))
                self.edit_plan(fixture, lambda p: p['bindings'].pop(str(fixture[key])))
                self.reject_both(fixture)
        for path in ('/usr/bin/time', '/usr/bin/timeout', '/usr/bin/flock', sys.executable):
            with self.subTest(path=path), self.scratch() as tmp:
                fixture = self.fixture(Path(tmp))
                self.edit_plan(fixture, lambda p: p['bindings'].pop(path))
                self.reject_both(fixture)
        with self.scratch() as tmp:
            fixture = self.fixture(Path(tmp))
            self.edit_plan(fixture, lambda p: p['bindings'].pop(str(fixture['sources'][0])))
            self.reject_both(fixture)

    def test_missing_bound_file_and_binding_digest_change_reject(self):
        with self.scratch() as tmp:
            fixture = self.fixture(Path(tmp))
            fixture['extra'].unlink()
            self.reject_both(fixture)
        with self.scratch() as tmp:
            fixture = self.fixture(Path(tmp))
            self.edit_plan(fixture, lambda p: p['bindings'].update({str(fixture['extra']): '0' * 64}))
            self.reject_both(fixture)

    def test_phase_deadline_duplicate_and_unsafe_parent_ids_reject(self):
        edits = [lambda p: p.pop('phase'), lambda p: p.update(phase='warm'),
                 lambda p: p.update(seconds=451), lambda p: p.update(seconds=True),
                 lambda p: p['cases'].append(copy.deepcopy(p['cases'][0])),
                 lambda p: p['cases'][0].update(id='../escape')]
        for index, edit in enumerate(edits):
            with self.subTest(index=index), self.scratch() as tmp:
                fixture = self.fixture(Path(tmp))
                self.edit_plan(fixture, edit)
                self.reject_both(fixture)

    def test_source_case_worker_and_review_contract_changes_reject(self):
        for key, value in [('scope', 'changed'), ('binary', 'different'),
                           ('rust_source', 'different'), ('source_case_id', 'missing')]:
            with self.subTest(key=key), self.scratch() as tmp:
                fixture = self.fixture(Path(tmp))
                self.edit_plan(fixture, lambda p: p['cases'][0].update({key: value}))
                self.reject_both(fixture)
        with self.scratch() as tmp:
            fixture = self.fixture(Path(tmp))
            self.edit_source(fixture, lambda s: s.update(rust_source='changed'))
            self.reject_both(fixture)
        with self.scratch() as tmp:
            fixture = self.fixture(Path(tmp))
            review = json.loads(fixture['review'].read_text())
            review['routes'][0]['canonical_command'] = ['changed']
            fixture['review'].write_text(json.dumps(review))
            self.edit_plan(fixture, lambda p: p['bindings'].update({
                str(fixture['review']): sha(fixture['review'])}))
            self.reject_both(fixture)

    def test_numerical_bounds_stay_finite_and_nonnegative(self):
        for value in (-1, float('nan'), float('inf')):
            with self.subTest(value=value), self.scratch() as tmp:
                fixture = self.fixture(Path(tmp))
                self.edit_source(fixture, lambda s: s['cases'][0].update(absolute_bound=value),
                                 matching_cases=True)
                self.reject_both(fixture)

    def test_scope_and_review_priming_gates_are_unchanged(self):
        for edit in [lambda p: p['cases'][0]['first_call_scope'].update(matched=False),
                     lambda p: p['cases'][0]['first_call_scope'].pop('parameter_io')]:
            with self.scratch() as tmp:
                fixture = self.fixture(Path(tmp))
                self.edit_plan(fixture, edit)
                self.assertFalse(self.outcome(old, fixture, with_scope=True))
                self.assertFalse(self.outcome(new, fixture, with_scope=True))
        with self.scratch() as tmp:
            fixture = self.fixture(Path(tmp))
            review = json.loads(fixture['review'].read_text())
            review['routes'][0]['adapter_pair_evaluations_before_first_timed_observation'] = 2
            fixture['review'].write_text(json.dumps(review))
            self.edit_plan(fixture, lambda p: p['bindings'].update({
                str(fixture['review']): sha(fixture['review'])}))
            self.assertFalse(self.outcome(old, fixture, with_scope=True))
            self.assertFalse(self.outcome(new, fixture, with_scope=True))

    def test_malformed_source_and_review_reject(self):
        for key in ('source', 'review'):
            with self.subTest(key=key), self.scratch() as tmp:
                fixture = self.fixture(Path(tmp))
                path = fixture['sources'][0] if key == 'source' else fixture['review']
                path.write_text('{malformed')
                self.reject_both(fixture)

    def test_duplicate_source_identity_is_explicitly_stricter(self):
        with self.scratch() as tmp:
            fixture = self.fixture(Path(tmp))
            self.edit_source(fixture, lambda s: s['cases'].append(copy.deepcopy(s['cases'][0])))
            self.assertTrue(self.outcome(old, fixture))
            with self.assertRaisesRegex(ValueError, 'invalid/duplicate source case identity'):
                new.verify(fixture['plan'], sha(fixture['plan']))

    def test_malformed_unused_source_case_is_stricter(self):
        with self.scratch() as tmp:
            fixture = self.fixture(Path(tmp))
            self.edit_source(fixture, lambda s: s['cases'].append({}))
            self.assertTrue(self.outcome(old, fixture))
            self.assertFalse(self.outcome(new, fixture))

    def test_successor_requires_additional_self_binding(self):
        with self.scratch() as tmp:
            fixture = self.fixture(Path(tmp))
            self.edit_plan(fixture, lambda p: p['bindings'].pop(str(Path(new.__file__))))
            self.assertTrue(self.outcome(old, fixture))
            self.assertFalse(self.outcome(new, fixture))

    def test_successful_block_verifies_before_after_each_arm_and_finally(self):
        for module in (old, new):
            for number in (0, 1):
                with self.subTest(module=module.__name__, number=number), self.scratch() as tmp:
                    root = Path(tmp)
                    fixture = self.fixture(root)
                    plan = json.loads(fixture['plan'].read_text())
                    routes = {r['metric']: r for r in json.loads(fixture['review'].read_text())['routes']}
                    events = []
                    def verified(*args):
                        events.append('verify')
                        return plan, routes
                    def arm(case, role, directory, seconds):
                        events.append(role)
                        return dict(ns=10, score=0.5)
                    order = ['canonical', 'native'] if number == 0 else ['native', 'canonical']
                    with patch.object(module, 'verify', side_effect=verified), \
                         patch.object(module, 'run_arm', side_effect=arm):
                        result = module.block(fixture['plan'], sha(fixture['plan']),
                                              plan['cases'][0]['id'], number, root / 'block')
                    self.assertEqual(events, ['verify', 'verify', order[0], 'verify',
                                              'verify', order[1], 'verify', 'verify'])
                    self.assertEqual(result['errors'], {})
                    self.assertIsNone(result['passed'])

    def test_post_arm_tamper_is_rejected_before_next_arm(self):
        for module in (old, new):
            with self.subTest(module=module.__name__), self.scratch() as tmp:
                root = Path(tmp)
                fixture = self.fixture(root)
                plan = json.loads(fixture['plan'].read_text())
                called = []
                def arm(case, role, directory, seconds):
                    called.append(role)
                    fixture['extra'].write_bytes(b'changed between arms')
                    return dict(ns=10, score=0.5)
                with patch.object(module, 'run_arm', side_effect=arm):
                    result = module.block(fixture['plan'], sha(fixture['plan']),
                                          plan['cases'][0]['id'], 0, root / 'block')
                self.assertEqual(called, ['canonical'])
                self.assertIn('bound artifact changed', result['errors']['admission_or_binding'])
                self.assertNotIn('native', result['arms'])

    def test_only_verify_function_and_required_self_binding_changed(self):
        def functions(module):
            tree = ast.parse(Path(module.__file__).read_text())
            return {n.name: ast.dump(n, include_attributes=False) for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name != 'verify'}
        self.assertEqual(functions(old), functions(new))
        self.assertEqual(set(new.HELPERS) - set(old.HELPERS),
                         {'compare_classical_first_verified.py'})


if __name__ == '__main__':
    unittest.main()
