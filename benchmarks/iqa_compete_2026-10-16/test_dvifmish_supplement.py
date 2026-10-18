"""Supplement selection and deployment controls; no benchmark acceptance changes."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import prepare_dvifmish as supplement


class SupplementTests(unittest.TestCase):
    def test_template_dedup_preserves_modes_phases_budgets_and_repeats(self):
        cells = [dict(metric_name=metric, size_name='cid22', width=512, height=512,
            threads=threads, only_mode=mode, timing_phase=phase, cold_repeat=repeat)
            for metric in ['first', 'other'] for threads in [1, 8] for mode in ['score', 'score+map']
            for phase in ['cold', 'warm'] for repeat in range(3 if phase == 'cold' else 1)]
        chosen = supplement.templates(dict(cells=cells))
        self.assertEqual(len(chosen), 16)
        self.assertEqual({c['only_mode'] for c in chosen}, {'score', 'score+map'})
        self.assertEqual({c['threads'] for c in chosen}, {1, 8})
        self.assertEqual(sum(c['timing_phase'] == 'cold' for c in chosen), 12)
        self.assertEqual(sum(c['timing_phase'] == 'warm' for c in chosen), 4)

    def test_mac_install_keeps_external_dependencies_and_resource_policy(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / 'tmp') as directory:
            root = Path(directory)
            prior = dict(bundle=str(root / 'original-nn512-gpu'), output=str(root / 'old-result'))
            own = dict(bundle=str(root / '512-dvifmish'), output=str(root / '512-results'))
            plan = root / 'plan.json'
            plan.write_text(json.dumps(dict(platform='Darwin', lock=str(root / 'shared-heavy.lock'), phases=[
                dict(**own, after=[prior]),
                dict(bundle=str(root / 'larger-dvifmish'), output=str(root / 'larger-results'), after=[prior, own])
            ])))
            with patch.object(supplement.platform, 'system', return_value='Darwin'), \
                 patch.object(supplement.sys, 'platform', 'darwin'), \
                 patch.object(supplement.subprocess, 'run') as run:
                self.assertEqual(supplement.install(plan), 0)
                self.assertEqual(run.call_count, 2)
                commands = [c.args[0] for c in run.call_args_list]
            for command in commands:
                self.assertEqual(command[command.index('--lock') + 1], str(root / 'shared-heavy.lock'))
                self.assertEqual(command[command.index('--memory') + 1], '12G')
                self.assertEqual(command[command.index('--jobs') + 1], '8')
                self.assertIn(prior['bundle'], command)
            self.assertEqual(commands[1].count('--after'), 2)
            self.assertIn(own['output'], commands[1])

    def test_wrong_platform_refuses_before_service_launch(self):
        with tempfile.TemporaryDirectory(dir=Path.home() / 'tmp') as directory:
            plan = Path(directory) / 'plan.json'
            plan.write_text(json.dumps(dict(platform='Darwin', phases=[])))
            with patch.object(supplement.platform, 'system', return_value='Linux'), \
                 patch.object(supplement.subprocess, 'run') as run:
                with self.assertRaisesRegex(ValueError, 'native platform'):
                    supplement.install(plan)
                run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
