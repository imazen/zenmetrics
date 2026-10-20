import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from compare_canonical import sha
from publish_spotcheck_summary import export


class ObservedSummary(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inventory = self.root / 'inventory.json'
        self.inventory.write_text(json.dumps({'inventory': [{'metric_name': 'a'}, {'metric_name': 'b'}]}))
        self.headlines = self.root / 'results-00.json'
        self.headlines.write_text(json.dumps({'a': dict(ratio=0.45, ratio_cases=2,
            ratio_width=64, ratio_height=64, ratio_threads=1, ratio_source='source',
            ratio_plan_sha256='plan', canonical='original')}))
        (self.root / 'results.json').write_text(json.dumps(dict(schema=2,
            shards=[dict(file=self.headlines.name, sha256=sha(self.headlines))])))
        (self.root / 'unmeasured.json').write_text(json.dumps({'b': {'check': 'self reference'}}))
        columns = ['metric', 'canonical', 'width', 'height', 'threads', 'max_abs_delta',
                   'absolute_bound', 'passed', 'rust_ms', 'canonical_ms', 'ratio',
                   'matched_scope', 'rust_source', 'plan_sha256', 'case']
        stream = io.StringIO()
        writer = csv.DictWriter(stream, columns, delimiter='\t', lineterminator='\n')
        writer.writeheader()
        for case, duration in [('a-first', 3), ('a-second', 6)]:
            writer.writerow(dict(metric='a', canonical='original', width=64, height=64,
                threads=1, max_abs_delta=0, absolute_bound=0.001, passed=True,
                rust_ms=duration, canonical_ms=10, ratio=duration / 10, matched_scope=True,
                rust_source='source', plan_sha256='plan', case=case))
        self.cells = self.root / 'cells.tsv'
        self.cells.write_text(stream.getvalue())
        (self.root / 'cohorts.json').write_text(json.dumps({'cohort': dict(artifacts=[
            dict(file=self.cells.name, sha256=sha(self.cells))])}))

    def test_aggregate_never_supplies_invented_individual_times(self):
        output = self.root / 'published'
        result = export(self.root, self.inventory, output)
        self.assertEqual(result['names'], 2)
        with (output / 'summary.tsv').open() as stream:
            rows = list(csv.DictReader(stream, delimiter='\t'))
        self.assertEqual((rows[0]['ours_ms'], rows[0]['canonical_ms'], rows[0]['ratio']), ('3', '10', '0.3'))
        self.assertEqual(rows[0]['observation'], '0')
        self.assertEqual((rows[1]['status'], rows[1]['ratio']), ('self reference', ''))
        with (output / 'cells-000.tsv').open() as stream:
            cells = list(csv.DictReader(stream, delimiter='\t'))
        self.assertEqual([row['case'] for row in cells], ['a-first', 'a-second'])

    def test_changed_evidence_is_refused_without_creating_output(self):
        self.cells.write_text(self.cells.read_text() + '\n')
        output = self.root / 'refused'
        with self.assertRaises(ValueError):
            export(self.root, self.inventory, output)
        self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
