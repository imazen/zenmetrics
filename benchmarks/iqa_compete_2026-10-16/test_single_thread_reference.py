import contextlib
import io
import json
import sys
import unittest
from unittest.mock import patch

from canonical_single_thread_reference import main


class SingleThreadReferenceTests(unittest.TestCase):
    def test_budget_is_disclosed_and_pixels_are_unchanged(self):
        for budget in ('1', '28'):
            output = io.StringIO()
            args = ['adapter', '/reference', '64', '65', '/r.rgb', '/d.rgb', budget, '1']
            with patch.object(sys, 'argv', args), patch('os.execv') as execute, contextlib.redirect_stdout(output):
                main()
            execute.assert_called_once_with('/reference', ['/reference', '64', '65', '/r.rgb', '/d.rgb', '1', '1'])
            metadata = json.loads(output.getvalue().removeprefix('CANONICAL '))
            self.assertEqual(metadata['requested_worker_budget'], int(budget))
            self.assertEqual(metadata['effective_threads'], 1)

    def test_nonpositive_budget_or_samples_never_executes(self):
        for budget, samples in [('0', '1'), ('1', '0')]:
            args = ['adapter', '/reference', '64', '65', '/r', '/d', budget, samples]
            with patch.object(sys, 'argv', args), patch('os.execv') as execute, self.assertRaises(ValueError):
                main()
            execute.assert_not_called()


if __name__ == '__main__':
    unittest.main()
