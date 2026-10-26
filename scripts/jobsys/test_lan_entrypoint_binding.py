"""Exercise the launcher's real remote entrypoint validation before Docker mutation."""
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class Binding(unittest.TestCase):
    def test_optional_verified_readonly_bind_and_refusals(self):
        source = Path(__file__).with_name('lan_score_launch.sh').read_text()
        block = source.split('# BEGIN verified entrypoint binding\n')[1].split('# END verified entrypoint binding')[0]
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as tmp:
            script = Path(tmp)/'entrypoint.sh'
            script.write_text('#!/bin/bash\nexit 0\n'); script.chmod(0o755)
            digest = hashlib.sha256(script.read_bytes()).hexdigest()
            def run(path='', sha=''):
                return subprocess.run(['bash', '-euc', block+'printf "%s\\n" "${ENTRYPOINT_ARGS[@]}"'],
                    env={**os.environ, 'ZM_ENTRYPOINT_PATH': path, 'ZM_ENTRYPOINT_SHA': sha},
                    capture_output=True, text=True)
            self.assertEqual(run().returncode, 0)
            good = run(str(script), digest)
            self.assertEqual(good.returncode, 0)
            self.assertEqual(good.stdout.splitlines(), ['-v', str(script)+':/usr/local/bin/fleet-entrypoint.sh:ro'])
            alias = Path(tmp)/'alias.sh'; alias.symlink_to(script)
            for path, sha in [(str(script), ''), ('', digest), (str(script), '0'*64),
                              (str(alias), digest), (str(script)+';bad', digest),
                              (str(script.parent/'missing'), digest)]:
                self.assertNotEqual(run(path, sha).returncode, 0)
            script.chmod(0o644)
            self.assertNotEqual(run(str(script), digest).returncode, 0)

    def test_unsafe_binding_is_refused_before_ssh(self):
        source = Path(__file__).with_name('lan_score_launch.sh')
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as tmp:
            fake = Path(tmp)/'ssh'
            fake.write_text('#!/bin/bash\nprintf reached > "$PROBE_MARKER"\nexit 99\n')
            fake.chmod(0o755)
            marker = Path(tmp)/'reached'
            for path, digest in [('/disk/script;exit', 'a'*64), ('/disk/script', ''),
                                 ('', 'a'*64), ('/disk/script', '$(bad)')]:
                result = subprocess.run(['bash', str(source), 'fixture', 'job', 'role', 'cpu'],
                    env={**os.environ, 'PATH': tmp+':'+os.environ['PATH'],
                         'PROBE_MARKER': str(marker), 'ZEN_ENTRYPOINT_HOST_PATH': path,
                         'ZEN_ENTRYPOINT_SHA256': digest}, capture_output=True)
                self.assertEqual(result.returncode, 3)
                self.assertFalse(marker.exists())

    def test_control_cohort_is_forwarded_and_cross_run_or_shell_keys_are_refused(self):
        source = Path(__file__).with_name('lan_score_launch.sh')
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as tmp:
            fake = Path(tmp)/'ssh'
            fake.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$PROBE_MARKER"\nexit 99\n')
            fake.chmod(0o755)
            marker = Path(tmp)/'arguments'
            def run(key):
                return subprocess.run(['bash', str(source), 'fixture', 'job', 'role', 'cpu'],
                    env={**os.environ, 'PATH': tmp+':'+os.environ['PATH'],
                         'PROBE_MARKER': str(marker), 'ZEN_CONTROL_KEY': key,
                         'ZEN_ENTRYPOINT_HOST_PATH': '', 'ZEN_ENTRYPOINT_SHA256': ''},
                    capture_output=True)
            for key in ('', 'jobs/job/control-clock.json'):
                self.assertEqual(run(key).returncode, 99)
                expected = key or 'jobs/job/control.json'
                self.assertIn('ZM_CONTROL_KEY='+expected, marker.read_text().splitlines())
                marker.unlink()
            for key in ('jobs/other/control.json', 'jobs/job/../control.json',
                        'jobs/job/control.json;bad', 'jobs/job/control bad.json'):
                self.assertEqual(run(key).returncode, 3)
                self.assertFalse(marker.exists())


if __name__ == '__main__':
    unittest.main()
