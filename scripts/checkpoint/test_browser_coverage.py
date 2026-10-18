"""Cheap catalog regression controls; no browser assets or weights required."""
import ast
import json
from pathlib import Path
import re
import unittest

REPO = Path(__file__).resolve().parents[2]


def coverage_helpers():
    # check-pages is an executable browser harness. Load its pure coverage
    # helpers without starting Chromium or parsing this test's command line.
    source = REPO / 'scripts/checkpoint/check-pages.py'
    names = {'card_of', 'learned_blocker', 'unavailable_reason', 'check_coverage'}
    tree = ast.parse(source.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    scope = {'REPO': REPO}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), scope)
    return scope


class BrowserCoverage(unittest.TestCase):
    def test_every_cli_name_has_coverage_in_declaration_order(self):
        source = (REPO / 'crates/zenmetrics-cli/src/metrics/mod.rs').read_text()
        enum = source.split('pub enum MetricKind {', 1)[1].split('\n}', 1)[0]
        names = re.findall(r'#\[value\(name = "([^"]+)"\)\]', enum)
        coverage = json.loads((REPO / 'web/metrics/coverage.json').read_text())
        self.assertEqual(names, [e['name'] for e in coverage['metrics']])

    def test_pyiqa_blockers_use_the_variant_card(self):
        helpers = coverage_helpers()
        for name in ('topiq-fr', 'topiq-fr-pipal', 'wadiqam-pyiqa', 'ahiq-pyiqa', 'pieapp-pyiqa'):
            reason = helpers['learned_blocker'](name, {})
            self.assertIn('zenmetrics-pyiqa-ports/models/', reason)
            self.assertIn('licence class "nc"', reason)
            self.assertIn('does not redistribute', reason)
        self.assertIn('SVD', helpers['learned_blocker']('dmm-pyiqa', {}))
        reason = helpers['unavailable_reason']('cw-ssim-pyiqa', 'disabled')
        self.assertIn('No weights are needed', reason)
        self.assertIn('Noncommercial', reason)

    def test_catalog_guard_rejects_missing_or_misclassified_entries(self):
        check = coverage_helpers()['check_coverage']
        coverage = json.loads((REPO / 'web/metrics/coverage.json').read_text())
        check(coverage, coverage)
        missing = {**coverage, 'metrics': coverage['metrics'][:-1]}
        with self.assertRaises(AssertionError):
            check(missing, coverage)
        changed = json.loads(json.dumps(coverage))
        next(e for e in changed['metrics'] if e['name'] == 'cw-ssim-pyiqa')['status'] = 'score-only'
        with self.assertRaises(AssertionError):
            check(changed, coverage)


if __name__ == '__main__':
    unittest.main()
