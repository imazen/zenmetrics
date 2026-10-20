"""Inert protocol controls; synthetic times are never benchmark evidence."""
import json
from pathlib import Path
import tempfile
import unittest

import classical_first_intervals as subject
from compare_canonical import sha


class StageControls(unittest.TestCase):
    def setUp(self):
        scratch = Path.home() / "tmp"
        scratch.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=scratch, prefix="classical-interval-controls-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def stage(self, name, bound=None, count=20, parent=None):
        root = self.root / name
        root.mkdir()
        case = dict(id="inert", metric="fixture", width=1, height=1, threads=1,
                    absolute_bound=bound, rust_source="fixture-source",
                    first_call_scope=dict(matched=True, boundary=subject.PHASE))
        plan = dict(phase=subject.PHASE, cases=[case], bindings={"fixture": "identity"})
        if parent is not None:
            plan["preceding_plan_sha256"] = sha(parent / "plan.json")
        (root / "plan.json").write_text(json.dumps(plan))
        directory = root / "inert"
        directory.mkdir()
        blocks = []
        for number in range(count):
            block = dict(phase=subject.PHASE, errors={}, scope_reasons=[],
                         order=["canonical", "native"] if number % 2 == 0 else ["native", "canonical"],
                         arms={role: dict(ns=1, score=1) for role in ["native", "canonical"]},
                         passed=None if bound is None else True, delta=0.0)
            path = directory / f"block-{number:02d}"
            path.mkdir()
            (path / "result.json").write_text(json.dumps(block))
            blocks.append(block)
        summary = dict(case="inert", phase=subject.PHASE, plan_sha256=sha(root / "plan.json"),
                       scope_reasons=[], ratio=1, absolute_bound=bound,
                       first_call_scope=case["first_call_scope"], blocks=blocks)
        (directory / "summary.json").write_text(json.dumps(summary))
        return root

    def test_null_bound_stays_unqualified(self):
        row = subject.analyze(self.stage("initial"), [])["rows"][0]
        self.assertFalse(row["numerical_qualified"])
        self.assertFalse(row["regression_qualified"])
        self.assertIsNone(row["absolute_bound"])

    def test_bounded_identical_scores_preserve_numerical_gate(self):
        row = subject.analyze(self.stage("initial", bound=0), [])["rows"][0]
        self.assertTrue(row["numerical_qualified"])
        self.assertFalse(row["regression_qualified"])

    def test_incomplete_stage_rejected(self):
        with self.assertRaisesRegex(ValueError, "complete fixed twenty-block"):
            subject.analyze(self.stage("incomplete", count=19), [])

    def test_null_bound_cannot_be_reported_as_passed(self):
        root = self.stage("initial")
        path = root / "inert/block-00/result.json"
        block = json.loads(path.read_text())
        block["passed"] = True
        path.write_text(json.dumps(block))
        path = root / "inert/summary.json"
        summary = json.loads(path.read_text())
        summary["blocks"][0] = block
        path.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "numerical gate failed or recorded gate differs"):
            subject.analyze(root, [])

    def test_summary_cannot_omit_acquired_block(self):
        root = self.stage("initial")
        path = root / "inert/summary.json"
        summary = json.loads(path.read_text())
        summary["blocks"].pop()
        path.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "retain every ordered block"):
            subject.analyze(root, [])

    def test_successor_preserves_contract_and_all_sixty_blocks(self):
        initial = self.stage("initial", bound=0)
        successor = self.stage("successor", bound=0, count=40, parent=initial)
        row = subject.analyze(initial, [successor])["rows"][0]
        self.assertEqual(row["blocks"], 60)
        self.assertEqual(len(row["block_sha256"]), 60)
        path = successor / "plan.json"
        plan = json.loads(path.read_text())
        plan["bindings"]["fixture"] = "changed"
        path.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError, "changes an original artifact binding"):
            subject.analyze(initial, [successor])


if __name__ == "__main__":
    unittest.main()
