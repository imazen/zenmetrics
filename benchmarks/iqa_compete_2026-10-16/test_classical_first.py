#!/usr/bin/env python3
"""Synthetic protocol/transport controls; no metric, model or inference invoked."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import compare_classical_first as first
from compare_canonical import compare, sha
from native_classical_first_cell import parse_native


CASE = dict(id="flip-512-t1", metric="flip", width=512, height=512, threads=1)
ROW = dict(metric_name="flip", width=512, height=512, threads=1, only_mode="score",
           timing_phase="cold", n=1, status="ok", cold_elapsed_ns=100, score=0.5,
           metadata={"retained": True})
SCOPE = dict(matched=True, boundary=first.PHASE, construction="both inside",
             parameter_io="none", runtime="imports excluded", teardown="declared",
             evidence=["synthetic control"])
ROUTE = dict(metric="flip", classification="first_evaluation_candidate",
             canonical_emits_cold=True, canonical_cold_is_first_pair_evaluation=True,
             adapter_pair_evaluations_before_first_timed_observation=0)


class FirstCallTests(unittest.TestCase):
    def test_native_preserves_metadata(self):
        self.assertEqual(parse_native("IQA_ROW " + json.dumps(ROW), CASE), ROW)

    def test_native_rejects_bad_observations(self):
        for key, value in [("status", "failed"), ("status", "unsupported"),
                           ("score", float("nan")), ("score", float("inf")),
                           ("cold_elapsed_ns", 0), ("cold_elapsed_ns", float("inf")),
                           ("cold_elapsed_ns", "10"), ("cold_elapsed_ns", True),
                           ("n", 2), ("n", True), ("timing_phase", "warm"),
                           ("threads", 8), ("width", 64), ("height", 64),
                           ("metric_name", "niqe"), ("only_mode", "score+map")]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                parse_native("IQA_ROW " + json.dumps(dict(ROW, **{key: value})), CASE)
        for text in ("", "IQA_ROW {}", "IQA_ROW broken",
                     "\n".join(["IQA_ROW " + json.dumps(ROW)] * 2)):
            with self.subTest(text=text), self.assertRaises((ValueError, KeyError)):
                parse_native(text, CASE)

    def test_canonical_requires_one_finite_first(self):
        self.assertEqual(first.parse_cold("COLD 12 0.5\nSAMPLE 8 0.5"),
                         dict(ns=12, score=0.5))
        for text in ("SAMPLE 8 0.5", "COLD 0 1", "COLD nan 1", "COLD 1 inf",
                     "COLD 1 1 extra", "COLD 1 1\nCOLD 2 1"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                first.parse_cold(text)

    def test_no_bound_is_not_a_parity_failure(self):
        self.assertEqual(compare(3, 1, None), (2, None))
        self.assertEqual(compare(3, 1, 0.1), (2, False))
        blocks = [dict(order=["canonical", "native"] if i % 2 == 0 else ["native", "canonical"],
                       errors={}, passed=None,
                       arms=dict(native=dict(ns=20, score=3), canonical=dict(ns=10, score=1)))
                  for i in range(3)]
        self.assertEqual(first.summarize(blocks, [], 3), 2)
        blocks[1]["passed"] = False
        self.assertIsNone(first.summarize(blocks, [], 3))

    def test_incompatible_or_primed_scope_withholds(self):
        case = dict(CASE, matched_scope=True, first_call_scope=SCOPE)
        self.assertEqual(first.scope_reasons(case, ROUTE), [])
        for classification in ("resident_setup_excluded", "parameter_io_scope_mismatch",
                               "cold_not_emitted", "pre_evaluated", "precreated_thread_runner"):
            reasons = first.scope_reasons(case, dict(ROUTE, classification=classification))
            self.assertTrue(reasons)
            self.assertIsNone(first.summarize([], reasons, 3))
        self.assertTrue(first.scope_reasons(case, dict(
            ROUTE, adapter_pair_evaluations_before_first_timed_observation=2)))
        self.assertTrue(first.scope_reasons(dict(case, first_call_scope={}), ROUTE))

    def test_missing_error_or_wrong_order_withholds(self):
        good = [dict(order=["canonical", "native"] if i % 2 == 0 else ["native", "canonical"],
                     arms=dict(native=dict(ns=20), canonical=dict(ns=10)), errors={}, passed=True)
                for i in range(3)]
        self.assertEqual(first.summarize(good, [], 3), 2)
        self.assertIsNone(first.summarize(good[:2], [], 3))
        for change in (dict(errors={"native": "exit 2"}), dict(arms={}),
                       dict(order=["native", "canonical"])):
            blocks = copy.deepcopy(good)
            blocks[0].update(change)
            self.assertIsNone(first.summarize(blocks, [], 3))

    def fixture(self, root, classification="first_evaluation_candidate", tamper=False):
        """Bind tiny fake workers and source-plan bytes, exercising real subprocess IO."""
        worker = root / "worker.py"
        worker.write_text("#!" + sys.executable + "\nimport json,sys\n"
                          "c=json.load(open(sys.argv[1])); c.update(status='ok',n=1,"
                          "cold_elapsed_ns=200,score=0.5,metadata={'fixture':True})\n"
                          "print('IQA_ROW '+json.dumps(c))\n")
        worker.chmod(0o700)
        canonical = root / "canonical.py"
        canonical.write_text("import sys\nassert int(sys.argv[1]) == 1, 'positive sample count required'\n"
                             "print('COLD 100 0.5')\nprint('SAMPLE 50 0.75')\n" +
                             ("from pathlib import Path\nPath(__file__).write_text('# changed')\n"
                              if tamper else ""))
        image = root / "input.rgb"
        image.write_bytes(b"rgb")
        source_case = dict(CASE, reference=str(image), distorted=str(image),
                           canonical=dict(label="synthetic",
                                          command=[sys.executable, str(canonical), "{samples}"]),
                           absolute_bound=None, matched_scope=True, scope="synthetic")
        bindings = {str(p): sha(p) for p in (worker, canonical, image)}
        source = root / "source.json"
        first.write(source, dict(binary=str(worker), rust_source="synthetic",
                                 cases=[source_case], bindings=bindings.copy()))
        review = root / "review.json"
        route = dict(ROUTE, classification=classification,
                     canonical_command=source_case["canonical"]["command"])
        first.write(review, dict(routes=[route]))
        case = dict(source_case, source_plan=str(source), source_case_id=source_case["id"],
                    binary=str(worker), rust_source="synthetic", first_call_scope=SCOPE)
        required = [source, review, Path(sys.executable), Path("/usr/bin/time"),
                    Path("/usr/bin/timeout"), Path("/usr/bin/flock"),
                    Path.home() / "work/claudehints/scripts/run-heavy"]
        required += [Path(first.__file__).parent / n for n in first.HELPERS]
        bindings.update({str(p): sha(p) for p in required})
        plan = root / "plan.json"
        first.write(plan, dict(phase=first.PHASE, seconds=5, review=str(review),
                               bindings=bindings, cases=[case]))
        return plan, case

    def scratch(self):
        return tempfile.TemporaryDirectory(prefix="classical-first-test-",
                                           dir=os.environ["TMPDIR"])

    def test_fresh_process_transport_and_metadata(self):
        with self.scratch() as tmp:
            root = Path(tmp)
            plan, case = self.fixture(root)
            results = [first.block(plan, sha(plan), case["id"], n, root / f"block{n}")
                       for n in range(3)]
            self.assertEqual(first.summarize(results, [], 3), 2)
            self.assertIsNone(results[0]["passed"])
            row = results[0]["arms"]["native"]["worker"]
            self.assertEqual(row["metadata"], {"fixture": True})
            self.assertFalse(row["probe_only"])
            self.assertEqual(row["only_mode"], "score")
            self.assertTrue((root / "block0/native/worker/worker-memory.txt").exists())
            self.assertTrue((root / "block0/canonical/canonical-stderr.log").exists())
            self.assertIn("SAMPLE 50 0.75", (
                root / "block0/canonical/canonical-stdout.log").read_text())
            self.assertEqual(results[0]["arms"]["canonical"], dict(ns=100, score=0.5))
            with self.assertRaises(FileExistsError):
                first.block(plan, sha(plan), case["id"], 0, root / "block0")

    def test_hash_tamper_withholds_and_stops_next_arm(self):
        with self.scratch() as tmp:
            root = Path(tmp)
            plan, case = self.fixture(root, tamper=True)
            result = first.block(plan, sha(plan), case["id"], 0, root / "block")
            self.assertIn("admission_or_binding", result["errors"])
            self.assertNotIn("native", result["arms"])

    def test_prewarmed_route_never_launches(self):
        with self.scratch() as tmp:
            root = Path(tmp)
            plan, case = self.fixture(root, classification="pre_evaluated")
            with patch.object(first, "run_arm", side_effect=AssertionError("must not launch")):
                result = first.block(plan, sha(plan), case["id"], 0, root / "block")
            self.assertIn("scope", result["errors"])
            self.assertEqual(result["arms"], {})

    def test_changed_plan_is_retained_error(self):
        with self.scratch() as tmp:
            root = Path(tmp)
            plan, case = self.fixture(root)
            digest = sha(plan)
            plan.write_text(plan.read_text() + "\n")
            result = first.block(plan, digest, case["id"], 0, root / "block")
            self.assertIn("plan changed", result["errors"]["admission_or_binding"])

    def test_wrong_or_missing_phase_refuses_before_execution(self):
        for phase in (None, "resident_setup_first_eval", "warm"):
            with self.subTest(phase=phase), self.scratch() as tmp:
                root = Path(tmp)
                plan, case = self.fixture(root)
                data = json.loads(plan.read_text())
                if phase is None:
                    del data["phase"]
                else:
                    data["phase"] = phase
                plan.write_text(json.dumps(data))
                with patch.object(first, "run_arm", side_effect=AssertionError("must not launch")):
                    result = first.block(plan, sha(plan), case["id"], 0, root / "block")
                self.assertIn("plan phase must equal", result["errors"]["admission_or_binding"])
                self.assertEqual(result["arms"], {})


if __name__ == "__main__":
    unittest.main()
