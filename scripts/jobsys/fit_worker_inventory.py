#!/usr/bin/env python3
"""Read live FitCell allocations for coverage and resource-admission receipts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess


def host_workers(host, runs):
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", host):
        raise ValueError("host must be a configured SSH alias")
    names = subprocess.check_output(
        ["ssh", "-o", "BatchMode=yes", host,
         "docker ps --filter name=zen-score- --format '{{.Names}}'"], text=True).split()
    if not names:
        return []
    if any(not re.fullmatch(r"[a-zA-Z0-9_.-]+", n) for n in names):
        raise ValueError("unexpected container name")
    containers = json.loads(subprocess.check_output(
        ["ssh", "-o", "BatchMode=yes", host, "docker inspect " + " ".join(names)], text=True))
    workers = []
    for c in containers:
        env = dict(s.split("=", 1) for s in c["Config"]["Env"] if "=" in s)
        run = env.get("ZEN_RUN", "").removeprefix("jobs/")
        if run not in runs or not c["State"]["Running"]:
            continue
        workers.append(dict(host=host, worker=env["ZEN_WORKER"], run=run, kind="container",
                            container=c["Name"].lstrip("/"), pid=c["State"]["Pid"],
                            started_at=c["State"]["StartedAt"], image=c["Config"]["Image"],
                            memory_max=c["HostConfig"]["Memory"],
                            memory_swap=c["HostConfig"]["MemorySwap"],
                            cpuset=c["HostConfig"]["CpusetCpus"],
                            claim_ttl_secs=env.get("ZEN_CLAIM_TTL_SECS")))
    return workers


def native_workers(runs):
    workers = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            argv = (p / "cmdline").read_bytes().decode().strip("\0").split("\0")
            if (not argv[0].endswith("zenfleet-worker") or "--worker" not in argv
                    or "--claims-prefix" not in argv):
                continue
            prefix = argv[argv.index("--claims-prefix") + 1]
            run = prefix.removeprefix("jobs/").removesuffix("/claims")
            if prefix != f"jobs/{run}/claims" or run not in runs:
                continue
            workers.append(dict(host=Path("/proc/sys/kernel/hostname").read_text().strip(),
                                worker=argv[argv.index("--worker") + 1], run=run,
                                kind="native", pid=int(p.name),
                                cgroup=(p / "cgroup").read_text().strip()))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return workers


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", action="append", required=True)
    p.add_argument("--run", action="append", required=True)
    p.add_argument("--include-native", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--live-workers", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists() or args.live_workers.exists():
        raise ValueError("use fresh receipt paths")
    runs = set(args.run)
    with ThreadPoolExecutor(max_workers=4) as pool:
        groups = list(pool.map(lambda host: host_workers(host, runs), args.host))
    workers = [w for group in groups for w in group]
    if args.include_native:
        workers.extend(native_workers(runs))
    workers.sort(key=lambda w: (w["host"], w["worker"]))
    names = [w["worker"] for w in workers]
    if len(names) != len(set(names)):
        raise ValueError("live worker identity appears more than once")
    counts = {host: sum(w["host"] == host for w in workers)
              for host in sorted({w["host"] for w in workers})}
    report = dict(observed_at=datetime.now(timezone.utc).isoformat(),
                  runs=sorted(runs), counts=counts, workers=workers)
    for path, value in ((args.out, report), (args.live_workers, names)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
    print(json.dumps(dict(observed_at=report["observed_at"], counts=counts, workers=len(workers))))


if __name__ == "__main__":
    main()
