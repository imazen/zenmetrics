"""E33 per-cell wall caps: read from the program's E33 package, enforced by killing the fit, never retried."""

import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

import fit_cell_exec as owner

VARIANT = dict(root="/var/tmp/rev4-featpot/v2e29", spec="sel:c", data_sha="d" * 64, routes={"kadid": []},
               wall_cap_sec=2)


def argv(script="v2_lodo_mlp.py", route=("--heldout", "kadid"), spec="sel:c"):
    return [script, "--spec", spec, "--root", VARIANT["root"], *route]


class WallCap(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path.home() / "tmp")
        self.addCleanup(self.tmp.cleanup)
        self.program = Path(self.tmp.name)
        (self.program / "benchmarks").mkdir()

    def package(self, variants, schema="e33-research-fit-package-v1"):
        (self.program / owner.E33_CONTRACT).write_text(json.dumps(dict(schema=schema, variants=variants)))

    def test_no_e33_package_keeps_the_uncapped_path(self):
        self.assertIsNone(owner.registered_wall_cap(argv(), "d" * 64, self.program))

    def test_cap_follows_the_registered_variant_and_route(self):
        full = {**VARIANT, "routes": {"production": []}, "wall_cap_sec": 7600}
        self.package([VARIANT, full])
        self.assertEqual(owner.registered_wall_cap(argv(), "d" * 64, self.program), 2)
        self.assertEqual(owner.registered_wall_cap(argv("v2_confirm_fit.py", ("--pack-production",)), "d" * 64,
                                                   self.program), 7600)
        for bad in (argv(spec="sel:x"), argv(route=("--heldout", "tid2013"))):
            with self.assertRaises(ValueError):
                owner.registered_wall_cap(bad, "d" * 64, self.program)
        self.package([{**VARIANT, "wall_cap_sec": 0}])
        with self.assertRaises(ValueError):
            owner.registered_wall_cap(argv(), "d" * 64, self.program)

    def test_over_cap_fit_is_killed_with_a_diagnosis_and_not_retried(self):
        dest = self.program / "cell"
        dest.mkdir()
        (dest / "train.log").write_text("epoch 0 | t=1.0s\nepoch 17 | t=9.0s\n")
        child_pid_file = self.program / "child.pid"
        script = (f"import os,time; open({str(child_pid_file)!r},'w').write(str(os.getpid())); time.sleep(60)")
        original = owner.PROGRAM
        owner.PROGRAM = self.program
        self.addCleanup(setattr, owner, "PROGRAM", original)
        started = time.monotonic()
        with self.assertRaises(owner.WallCapExceeded) as raised:
            owner.run_capped([sys.executable, "-c", script], dict(os.environ), 1, dest, Path("spec__N/c0"))
        self.assertLess(time.monotonic() - started, 30)
        message = str(raised.exception)
        self.assertIn("registered wall cap exceeded", message)
        self.assertIn("epoch 17", message)
        pid = int(child_pid_file.read_text())
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)
        self.assertIsNone(owner.error_class_of(raised.exception))

    def test_output_stage_refusal_is_deterministic_and_other_failures_unchanged(self):
        refusal = RuntimeError(f"fit exited 3: ... {owner.E33_OUTPUT_STAGE_REFUSAL}: K1 ...")
        self.assertIsNone(owner.error_class_of(refusal))
        self.assertEqual(owner.error_class_of(RuntimeError("fit exited 1: Traceback ...")), "unknown")
        self.assertEqual(owner.error_class_of(RuntimeError("fit exited 1: OSError: [Errno 5]")), "worker_lost")


if __name__ == "__main__":
    unittest.main()
