#!/usr/bin/env python3
"""Freeze and preserve a young FitCell, then TERM its verified Docker allocation.

Run as root on the allocation host. Claim release remains the worker's native
SIGTERM handler; this tool never edits or deletes claim objects or ledger rows.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time


def command(*args):
    return subprocess.check_output(args, text=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("container", "worker", "run", "arm", "fold", "data-sha", "expected-image"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    p.add_argument("--max-epoch", type=int, default=30)
    p.add_argument("--preserve-completed-fit", action="store_true",
                   help="owner-directed arm cancellation may retain and stop a post-fit prediction stage")
    a = p.parse_args()
    if (not re.fullmatch(r"[0-9a-f]{64}", a.data_sha)
            or not re.fullmatch(r"[A-Za-z0-9]+", a.arm)
            or not re.fullmatch(r"[a-z0-9_]+", a.fold)):
        raise ValueError("unsafe cell path inputs")
    if os.geteuid() != 0:
        raise ValueError("root is required to preserve and signal only the verified allocation")
    c = json.loads(command("docker", "inspect", a.container))[0]
    allocation = c["Id"]
    env = dict(s.split("=", 1) for s in c["Config"]["Env"] if "=" in s)
    if (not c["State"]["Running"] or env.get("ZEN_WORKER") != a.worker
            or env.get("ZEN_RUN") != "jobs/" + a.run
            or c["Config"]["Image"] != a.expected_image):
        raise ValueError("allocation identity differs; nothing stopped")
    wanted = (a.arm, a.fold, str(a.seed))
    progress = json.loads(command("docker", "exec", "-e",
        "BYC_CELLS=/var/tmp/rev4-featpot/byc/cells", a.container, "python3",
        "/opt/fleet-fits/program/scripts/rev4_featpot/byc_cells.py", "progress",
        "--arm", a.arm, "--folds", a.fold, "--seeds", str(a.seed)))
    cell = progress["cells"][0]
    if (cell["state"] != "training" or cell["latest_logged_epoch"] is None
            or cell["latest_logged_epoch"] > a.max_epoch):
        raise ValueError("cell is not a young active fit; nothing stopped")
    scratch = Path(next(m["Source"] for m in c["Mounts"] if m["Destination"] == "/scratch"))
    source = scratch / "fit-cell" / a.data_sha / "rev4-featpot/byc/cells" / a.arm / f"without_{a.fold}_s{a.seed}"
    backup = scratch / "rebalanced-partials" / f"{a.worker}-{time.time_ns()}"
    held_output = None
    paused = False
    stopped_pids = []
    try:
        subprocess.run(["docker", "pause", allocation], check=True, stdout=subprocess.DEVNULL)
        paused = True
        rows = command("docker", "top", allocation, "-eo", "pid,comm").splitlines()[1:]
        pids = [int(row.split()[0]) for row in rows]
        workers = [int(row.split()[0]) for row in rows if row.split()[1] == "zenfleet-worker"]
        if len(workers) != 1:
            raise ValueError("expected exactly one native worker in the allocation")
        running = set()
        for pid in pids:
            argv = Path(f"/proc/{pid}/cmdline").read_bytes().decode().strip("\0").split("\0")
            if all(k in argv for k in ("--label", "--folds", "--seeds")):
                running.add(tuple(argv[argv.index(k) + 1] for k in ("--label", "--folds", "--seeds")))
        if (running != {wanted} or not source.is_dir()
                or ((source / "result.json").exists() and not a.preserve_completed_fit)):
            raise ValueError("running cell identity or completion state changed; nothing stopped")
        held_output = open(f"/proc/{workers[0]}/fd/2", "rb")
        shutil.copytree(source, backup / "cell", symlinks=True)
        hashes = {str(f.relative_to(backup / "cell")): hashlib.sha256(f.read_bytes()).hexdigest()
                  for f in (backup / "cell").rglob("*") if f.is_file()}
        # Keep trainers and the entrypoint stopped while the worker releases its
        # lease. Otherwise a killed trainer can race SIGTERM and publish FAILED.
        for pid in pids:
            if pid != workers[0]:
                os.kill(pid, signal.SIGSTOP)
                stopped_pids.append(pid)
        os.kill(workers[0], signal.SIGTERM)
        subprocess.run(["docker", "unpause", allocation], check=True, stdout=subprocess.DEVNULL)
        paused = False
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            try:
                state = Path(f"/proc/{workers[0]}/stat").read_text().split(") ", 1)[1].split()[0]
            except (FileNotFoundError, ProcessLookupError):
                break
            if state == "Z":
                break
            time.sleep(0.05)
        for pid in pids:
            if pid != workers[0]:
                try:
                    os.kill(pid, signal.SIGTERM)
                    os.kill(pid, signal.SIGCONT)
                except ProcessLookupError:
                    pass
        stopped_pids.clear()
        subprocess.run(["docker", "stop", "-t", "20", allocation], check=True, stdout=subprocess.DEVNULL)
        held_output.seek(0)
        log = held_output.read()
        (backup / "worker-stop.log").write_bytes(log)
        stopped = json.loads(command("docker", "inspect", allocation))[0]
        if stopped["State"]["Running"]:
            raise ValueError("allocation did not stop")
        report = dict(observed_at=datetime.now(timezone.utc).isoformat(), worker=a.worker,
                      container=a.container, allocation_id=allocation, run=a.run, cell="/".join(wanted),
                      latest_logged_epoch=cell["latest_logged_epoch"], stopped=True,
                      image=a.expected_image, backup=str(backup), files_sha256=hashes,
                      exit_code=stopped["State"]["ExitCode"], signalled_pids=pids,
                      native_release_reported=b"released chunk claim" in log)
        (backup / "STOP_RECEIPT.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))
        if not report["native_release_reported"]:
            raise ValueError("stopped and preserved, but native claim release needs verification")
    finally:
        if paused:
            subprocess.run(["docker", "unpause", allocation], check=False, stdout=subprocess.DEVNULL)
        if stopped_pids:
            # A failed operator step must not leave a frozen allocation behind.
            for pid in pids:
                try:
                    os.kill(pid, signal.SIGTERM)
                    os.kill(pid, signal.SIGCONT)
                except ProcessLookupError:
                    pass
            subprocess.run(["docker", "stop", "-t", "20", allocation], check=False,
                           stdout=subprocess.DEVNULL)
        if held_output is not None:
            held_output.close()


if __name__ == "__main__":
    main()
