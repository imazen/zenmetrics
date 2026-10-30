"""Family-split declaration of trade view jobs (large views fit the worker's 1 GiB staging limit)."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import declare_evif_trade as dec

FAMILIES = ['ycc444', 'ycc420', 'xyb-extended', 'xyb-capped', 'svt-420']


class Split(unittest.TestCase):
    def test_every_split_is_a_disjoint_cover_in_template_order(self):
        for parts in range(1, len(FAMILIES) + 1):
            groups = dec.split_families(FAMILIES, parts)
            self.assertEqual(len(groups), parts)
            flat = [f for g in groups for f in g]
            self.assertEqual(sorted(flat), sorted(FAMILIES))
            for g in groups:
                self.assertEqual(list(g), sorted(g, key=FAMILIES.index))
            self.assertEqual(groups, dec.split_families(list(FAMILIES), parts))

    def test_three_parts_are_balanced_by_the_pilot_weights(self):
        loads = [sum(dec.WEIGHT[f] for f in g) for g in dec.split_families(FAMILIES, 3)]
        self.assertLessEqual(max(loads) - min(loads), 13)
        self.assertEqual(sum(loads), sum(dec.WEIGHT.values()))

    def test_cannot_split_finer_than_the_family_count(self):
        with self.assertRaises(ValueError):
            dec.split_families(FAMILIES, 6)
        with self.assertRaises(ValueError):
            dec.split_families(FAMILIES, 0)

    def test_part_template_narrows_families_and_byte_controls_only(self):
        base = dict(families=FAMILIES, seed=7, kinds=['ladder'], views=[],
                    byte_controls=[dict(family=f, rung=3) for f in FAMILIES],
                    controls=[dict(x=1)], candidates=[dict(label='f')])
        part = dec.part_template(base, ('ycc420', 'svt-420'))
        self.assertEqual(part['families'], ['ycc420', 'svt-420'])
        self.assertEqual([c['family'] for c in part['byte_controls']], ['ycc420', 'svt-420'])
        for key in ('seed', 'kinds', 'views', 'controls', 'candidates'):
            self.assertEqual(part[key], base[key])
        self.assertEqual(base['families'], FAMILIES)       # the base is not edited

    def test_rules_pick_the_largest_threshold_reached(self):
        rules = [(200_000, 3), (450_000, 5)]
        self.assertEqual([dec.parts_for(p, rules) for p in (1, 199_999, 200_000, 449_999, 450_000, 921_600)],
                         [1, 1, 3, 3, 5, 5])
        self.assertEqual(dec.parts_for(10**9, []), 1)


class Declare(unittest.TestCase):
    def build(self, root, sizes, split):
        template = dict(families=FAMILIES, views=[], seed=1,
                        byte_controls=[dict(family=f, ramp='none', rung=3) for f in FAMILIES])
        template_path = root / 'template.json'
        template_path.write_text(json.dumps(template))
        executor = root / 'exec.py'
        executor.write_text('# executor\n')
        views, sources = [], []
        for i, (w, h) in enumerate(sizes):
            png = root / f'v{i}.png'
            png.write_bytes(b'png%d' % i)
            views.append(dict(id=f'v{i}', reference_group=f'g{i}', mlic_role='fit', content_class='cid22-train',
                              path=str(png), sha256=dec.sha(png.read_bytes()), width=w, height=h))
            sources.append(dict(source=dict(reference_group=f'g{i}', worker_raw=f'/raw{i}.png'), width=w, height=h,
                                source_pixels_sha256='aa' * 32, raw_sha256='bb' * 32))
        (root / 'VIEWS.json').write_text(json.dumps(views))
        (root / 'SOURCES.json').write_text(json.dumps(sources))
        out = root / 'declare.json'
        argv = ['declare', str(root / 'VIEWS.json'), str(root / 'SOURCES.json'), str(template_path), str(out),
                '--executor', str(executor), '--binary-sha256', '11' * 32, '--model', 'm.bin=' + '22' * 32,
                '--partition', 'all']
        if split:
            argv += ['--parts-dir', str(root / 'parts')] + [a for r in split for a in ('--split', r)]
            (root / 'parts').mkdir()
        with patch.object(sys, 'argv', argv):
            dec.main()
        return json.loads(out.read_text())['views'], root / 'parts'

    def test_unsplit_declaration_is_one_job_per_view_on_the_base_template(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as td:
            jobs, _ = self.build(Path(td), [(100, 100), (900, 900)], [])
            self.assertEqual(len(jobs), 2)
            for job in jobs:
                knobs = json.loads(job['cell']['knob_tuple_json'])
                self.assertTrue(knobs['template_path'].endswith('template.json'))
                self.assertNotIn('_work', job)

    def test_large_views_become_disjoint_family_jobs_pinned_by_their_part_template(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as td:
            jobs, parts = self.build(Path(td), [(100, 100), (500, 500), (900, 900)],
                                     ['200000:3', '450000:5'])
            by_view = {}
            for job in jobs:
                knobs = json.loads(job['cell']['knob_tuple_json'])
                template = Path(knobs['template_path'])
                data = template.read_bytes()
                self.assertEqual(dec.sha(data), knobs['template_sha256'])
                self.assertIn(knobs['template_sha256'], job['inputs'])
                self.assertIn(dec.sha(job['cell']['knob_tuple_json'].encode()), job['inputs'])
                families = json.loads(data)['families']
                self.assertEqual([c['family'] for c in json.loads(data)['byte_controls']],
                                 [f for f in FAMILIES if f in families])
                by_view.setdefault(knobs['view']['id'], []).append(families)
            self.assertEqual({k: len(v) for k, v in by_view.items()}, {'v0': 1, 'v1': 3, 'v2': 5})
            for families in by_view.values():
                flat = [f for g in families for f in g]
                self.assertEqual(sorted(flat), sorted(FAMILIES))          # nothing lost, nothing twice
            # The 3-way and 5-way splits share the lone xyb-extended group, which is one template file.
            self.assertEqual(len(list(parts.glob('*.json'))), 3 + 5 - 1)
            self.assertEqual(len({j['cell']['knob_tuple_json'] for j in jobs}), len(jobs))

    def test_heaviest_work_is_declared_first(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as td:
            jobs, _ = self.build(Path(td), [(100, 100), (900, 900), (500, 500)], ['200000:3', '450000:5'])
            views = [json.loads(j['cell']['knob_tuple_json'])['view']['id'] for j in jobs]
            self.assertEqual(views[0], 'v1')          # the 900x900 view's heaviest family job
            self.assertEqual(views[-1], 'v0')         # the 100x100 view

    def test_a_changed_part_template_is_refused(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as td:
            root = Path(td)
            _, parts = self.build(root, [(900, 900)], ['200000:3'])
            first = sorted(parts.glob('*.json'))[0]
            first.write_text('{}')
            with self.assertRaises(SystemExit):
                (root / 'declare.json').unlink()
                self.build_again(root)

    def build_again(self, root):
        argv = ['declare', str(root / 'VIEWS.json'), str(root / 'SOURCES.json'), str(root / 'template.json'),
                str(root / 'declare.json'), '--executor', str(root / 'exec.py'), '--binary-sha256', '11' * 32,
                '--model', 'm.bin=' + '22' * 32, '--partition', 'all', '--parts-dir', str(root / 'parts'),
                '--split', '200000:3']
        with patch.object(sys, 'argv', argv):
            dec.main()


if __name__ == '__main__':
    unittest.main()
