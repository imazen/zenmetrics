from pathlib import Path
import tempfile
import unittest

from reference_publication import register_outputs


class ReferencePublication(unittest.TestCase):
    def test_only_explicit_generated_artifacts_are_classified(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            repo=Path(directory); registry=repo/'benchmarks/dashboard/classification.tsv'
            registry.parent.mkdir(parents=True);registry.write_text('# explicit classifications\n')
            output=repo/'benchmarks/reference';output.mkdir()
            cohort=output/'cohort.tsv';cohort.write_text('metric\nexample\n')
            summary=output/'results.json';summary.write_text('{}')
            unrelated=output/'unrelated.tsv';unrelated.write_text('unknown\n')
            register_outputs([(cohort,'cohort'),(summary,'summary')],repo)
            text=registry.read_text()
            self.assertIn('benchmarks/reference/cohort.tsv\tignored\t',text)
            self.assertIn('benchmarks/reference/results.json\tignored\t',text)
            self.assertNotIn('unrelated.tsv',text)
            register_outputs([(cohort,'cohort'),(summary,'summary')],repo)
            self.assertEqual(registry.read_text(),text)

    def test_scratch_and_missing_or_unknown_artifacts_do_not_hide_uploads(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);repo=root/'repo';registry=repo/'benchmarks/dashboard/classification.tsv'
            registry.parent.mkdir(parents=True);registry.write_text('# unchanged\n')
            scratch=root/'scratch.json';scratch.write_text('{}')
            self.assertEqual(register_outputs([(scratch,'interval')],repo),[])
            self.assertEqual(registry.read_text(),'# unchanged\n')
            with self.assertRaises(ValueError):
                register_outputs([(repo/'benchmarks/missing.json','summary')],repo)
            with self.assertRaises(ValueError):
                register_outputs([(scratch,'unknown')],repo)
            self.assertEqual(registry.read_text(),'# unchanged\n')
