"""Resource-wrapper selection and timeout containment controls."""
import os
from pathlib import Path
import unittest
from unittest.mock import patch
from supervision import resource_wrapper, timeout_command


class Supervision(unittest.TestCase):
    def test_mac_child_stays_in_guarded_group(self):
        self.assertEqual(timeout_command('Darwin', 1100),
                         ['gtimeout', '--foreground', '--kill-after=20s', '1100s'])

    def test_linux_timeout_retains_existing_contract(self):
        self.assertEqual(timeout_command('Linux', 1100),
                         ['timeout', '--kill-after=20s', '1100s'])

    def test_scope_limits_and_unqualified_platform_refuse(self):
        for seconds in [0, 1101]:
            with self.assertRaises(ValueError):
                timeout_command('Darwin', seconds)
        with self.assertRaises(RuntimeError):
            timeout_command('Windows', 10)

    def test_explicit_wrapper_is_resolved(self):
        with patch.dict(os.environ, {'IQA_RUN_HEAVY': '/bin/sh'}):
            self.assertEqual(resource_wrapper(), str(Path('/bin/sh').resolve()))

    def test_missing_wrapper_refuses_before_launch(self):
        with patch.dict(os.environ, {'IQA_RUN_HEAVY': '/nonexistent/iqa-resource-wrapper'}):
            with self.assertRaises(RuntimeError):
                resource_wrapper()


if __name__ == '__main__':
    unittest.main()
