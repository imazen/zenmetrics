"""Diagnostic timing transport for zenbench pair and existing IQA adapters.

Usage: probe_pair_arm.py OUTPUT_DIRECTORY canonical|rust SPEC_JSON
Retains raw outputs and internal warm samples. This does not establish matched
workloads, clean resource gates, numeric acceptance, or qualified ratios.
The caller supplies the frozen programs, inputs, and explicit scope in SPEC_JSON.
"""

import json, sys, os, subprocess, hashlib, math
from pathlib import Path
from supervision import scoring_command

root = Path(sys.argv[1])
arm = sys.argv[2]
spec = json.loads(Path(sys.argv[3]).read_text())
env = dict(
    os.environ,
    TMPDIR=str(root),
    ZENBENCH_CPU_POLICY="unrestricted",
    ZENBENCH_NO_SAVE="1",
)
for k in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "RAYON_NUM_THREADS",
]:
    env[k] = "1"
if arm == "rust":
    c = dict(
        width=spec["w"],
        height=spec["h"],
        reference=spec["ref"],
        distorted=spec["dist"],
        metric_name=spec["metric"],
        threads=1,
        only_mode="score",
        timing_phase="warm",
        locked_process_accounting=True,
        sample_target_ns=1000000,
        chunk_budget_ms=20000,
    )
    (root / "config.json").write_text(json.dumps(c))
    cmd = [spec["rust"], str(root / "config.json")]
else:
    cmd = [s.replace("{output}", str(root)) for s in spec["command"]]
cmd = scoring_command(cmd, system="Linux", seconds=22, memory_log=root / "memory.txt")
with (root / "stdout.log").open("w") as out, (root / "stderr.log").open("w") as err:
    r = subprocess.run(cmd, env=env, stdout=out, stderr=err)
assert r.returncode == 0, r.returncode
lines = (root / "stdout.log").read_text().splitlines()
if arm == "rust":
    rows = [json.loads(x[8:]) for x in lines if x.startswith("IQA_ROW ")]
    assert len(rows) == 1 and rows[0]["status"] == "ok", rows
    times = rows[0]["round_means_ns"]
    scores = [rows[0]["score"]]
    extra = rows[0]
else:
    if spec.get("octave"):
        lines = (root / "octave" / "stdout.log").read_text().splitlines()
    rows = [x.split() for x in lines if x.startswith("SAMPLE ")]
    assert rows, lines
    idx = 2 if spec.get("indexed") else 1
    times = [float(x[idx]) for x in rows]
    scores = [float(x[idx + 1]) for x in rows]
    extra = {}
assert times and all(math.isfinite(x) and x > 0 for x in times)
record = dict(
    times_ns=times,
    scores=scores,
    command=cmd,
    scope=spec["scope"],
    extra=extra,
    input_hashes={
        p: hashlib.sha256(Path(spec[p]).read_bytes()).hexdigest()
        for p in ["ref", "dist"]
    },
)
(root / "arm.json").write_text(json.dumps(record, indent=2))
print("PAIR_TIME_NS", sum(times) / len(times), flush=True)
