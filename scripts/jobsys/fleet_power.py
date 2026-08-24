#!/usr/bin/env python3
"""fleet_power — queue-driven power management for the Nomad-managed LAN fleet.

Boxes on the SLEEP_ROSTER suspend-to-RAM when the job-system queue is shallow and are
woken (WoL) in measured-throughput order as depth grows, with hysteresis (wake threshold
> sleep threshold) and a minimum awake dwell so a woken box finishes real work before it's
eligible to sleep again. Per HANDOFF-nomad-power-fleet.md: "no parallel system" — this is
a `fleet` subcommand (see scripts/jobsys/fleet's `power` case-arm), not a new script
family, and reuses the SAME signals + thresholds the rest of the job system already uses:

  • Queue depth: `jobs/<run>/ledger_snapshot.parquet` footer row counts (num_rows == distinct
    done), the exact mechanism `pool_progress.py` uses — never a new counter.
  • Idle/awake-but-unproductive: `zenfleet_core::idle`'s thresholds, MIRRORED here (no
    heartbeat 180s / GPU <=10% / <=1 job/hr) — see idle_thresholds_note() below for why this
    file can't literally import the Rust crate and how the mirror is kept honest.
  • Awake/asleep + "is this box currently doing anything": the Nomad HTTP API directly
    (node status + allocations) — the box's OWN client agent already answers "is anything
    scheduled on me right now", which is a cleaner signal than re-deriving worker heartbeats
    once Nomad is the thing placing work.

SAFETY (read before running --apply on anything real):
  - NEVER_SLEEP boxes are hardcoded below and are never touched by this tool, full stop —
    not derived from any live signal, so a bug elsewhere can't accidentally sleep dev/tower/
    r7900x/wsl/mac. If a box hosts a SeaweedFS volume in the future (storage-v2 expansion),
    add it here too, unconditionally, per the "seaweed" principle in
    HANDOFF-nomad-power-fleet.md's resolved sleep-roster note.
  - A box only appears in SLEEP_ROSTER after a recorded G-P1 round-trip
    (wol_roundtrip_test.sh, 3/3) — see each entry's `gate` field. A box with `gate=None`
    is refused for --apply (status still reports it, clearly marked NOT GATED).
  - Hysteresis state (last-action timestamps, per box) persists to the LAN store
    (`jobs/_fleet_power/state.json`) — not local disk — because this runs as a Nomad
    periodic job, which may land on a different box each invocation.
  - `--dry-run` (the default for `apply`) computes and PRINTS the decision without ever
    calling WoL or suspend. Pass `--live` to actually act.

Usage:
  fleet_power.py status [--run RUN ...]
  fleet_power.py apply [--run RUN ...] [--live] [--nomad-addr http://192.168.50.44:4646]

RUN defaults to the pools this file's OWN queue-depth check knows about (see RUNLISTS
below) — override with --run for a specific fleetbench run once one is declared.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "lib"))
from zen_s3env import resolve  # THE resolver — see pool_progress.py's 2026-08-24 fix

import pyarrow.fs as fs
import pyarrow.parquet as pq

# ── Sleep roster + never-sleep set ──────────────────────────────────────────────────────
# Resolved 2026-08-24 (see HANDOFF-nomad-power-fleet.md's sleep-roster note + the ADR):
# "lilith boxes" = every box reached via the lilith SSH account (dev/r7900x/mac) -> r7900x
# is OUT despite being a pure client-eligible box, because it's also a Nomad SERVER and
# excluding it removes the raft-quorum edge case entirely. wsl is the operator seat, never
# in any roster. mac sleeps via its OWN idle gate (launchd + womp), not this mechanism.
NEVER_SLEEP = {"dev", "tower", "r7900x", "wsl", "mac"}

# gate: G-P1 round-trip result. None = never tested (refuse --apply). A float = the
# measured wake round-trip seconds from the LAST passing 3/3 gate run (see wol_roundtrip_test.sh).
SLEEP_ROSTER = {
    "i265": {
        "mac": "60:cf:84:76:20:d2",
        "ssh": "zen@192.168.50.140",
        "nomad_node_name": "i265",
        "gate": 8.0,  # PASSED 3/3, 2026-08-24: 8s, 8s, 8s -- see homefleet NODES.md
    },
    "r3500": {
        "mac": "04:42:1a:09:52:0f",
        "ssh": "zen@192.168.50.55",
        "nomad_node_name": "r3500",
        "gate": 7.0,  # PASSED 3/3, 2026-08-24: 7s, 7s, 7s -- see homefleet NODES.md
    },
    "r5900xt": {
        "mac": "30:c5:99:ef:60:79",
        "ssh": "zen@192.168.50.250",
        "nomad_node_name": "r5900xt",
        "gate": None,  # FAILED to even wake once, 2026-08-24 — see NODES.md. Do NOT set until re-tested.
    },
    "i134": {
        "mac": "04:7c:16:b3:18:51",
        "ssh": "zen@192.168.50.148",
        "nomad_node_name": "i134",
        "power_mode": "poweroff",  # S3 suspend-WoL FAILED 2026-08-24 (arm_wol.sh was skipped --
        # a process error, not a confirmed hardware verdict). User-directed pivot: use S5
        # (real power off/on) instead of chasing S3-arming on a PXE-netboot-every-boot machine --
        # S5-WoL just worked with ZERO extra config via `fleet-pxe worker <mac>` the same
        # session S3 failed. Trade-off accepted: slower wake (full cold boot, tens of seconds+)
        # for a wake mechanism that doesn't depend on OS/driver state surviving a netboot cycle.
        "gate": None,  # Box is currently ASLEEP (stuck from the S3 attempt) and unreachable —
        # needs a physical/keyboard nudge before ANY gate can be attempted. Once reachable:
        # run a poweroff/WoL round-trip gate (NOT wol_roundtrip_test.sh as-is -- it hardcodes
        # `systemctl suspend`; needs a poweroff variant or a manual 3x power-off+WoL round trip)
        # before setting this to a real number.
    },
    "r5600g": {
        "mac": "04:7c:16:8a:b5:b7",
        "ssh": "zen@192.168.50.193",
        "nomad_node_name": "r5600g",
        "power_mode": "poweroff",  # Same story and same S5 pivot as i134 above, same day.
        "gate": None,  # Same story as i134 above -- currently ASLEEP, unreachable, needs a
        # physical/keyboard nudge, then a poweroff/WoL round-trip gate before a real number.
    },
}

# ── Hysteresis knobs (all overridable via env for tuning without a code edit) ───────────
WAKE_GAP_THRESHOLD = int(os.environ.get("ZEN_POWER_WAKE_GAP", "500"))  # cells of undone work
SLEEP_GAP_THRESHOLD = int(os.environ.get("ZEN_POWER_SLEEP_GAP", "50"))  # must be < wake threshold
MIN_AWAKE_DWELL_SECS = int(os.environ.get("ZEN_POWER_MIN_DWELL_SECS", str(30 * 60)))
FLEET_PXE = str(
    pathlib.Path(__file__).resolve().parent.parent.parent.parent
    / "homefleet" / "zenmetrics" / "ubuntu-node" / "pxe" / "fleet-pxe"
)
STATE_KEY = "jobs/_fleet_power/state.json"


def idle_thresholds_note():
    """Why this file mirrors zenfleet_core::idle's thresholds instead of importing them:
    this is a Python script; the canonical values (crates/zenfleet-core/src/idle.rs) are
    stale_heartbeat_secs=180, min_gpu_util_pct=10, min_jobs_per_hr=1.0, grace_secs=120 (see
    CLAUDE.md's Fleet monitoring section: "Every tool uses these same thresholds — do not
    invent new ones"). fleet_power doesn't currently consume per-worker heartbeat reports
    (see the module docstring — it uses Nomad's own allocation state instead), so these
    constants aren't wired in yet; documented here so the NEXT thing that needs them copies
    the number, not a guess."""
    return {"stale_heartbeat_secs": 180, "min_gpu_util_pct": 10, "min_jobs_per_hr": 1.0, "grace_secs": 120}


def s3():
    ep, ak, sk = resolve()
    return fs.S3FileSystem(access_key=ak, secret_key=sk, endpoint_override=ep, region="auto"), ep


def bucket():
    return os.environ.get("ZEN_BUCKET", "zentrain")


def queue_gap(run: str) -> tuple[int, int] | None:
    """(declared, distinct_done) for one run, or None if no manifest is found at all.

    Two manifest path conventions coexist in this codebase and this function must find
    either (found live 2026-08-24 running the first real G-P3 test — a plain declared-
    manifest run has NO `jobs/` prefix and NO ledger_snapshot.parquet at all, so the
    original jobs/-prefix-only version of this function silently returned "no manifest"
    for every non-pool run):
      - POOL-mode runs: `jobs/<run>/manifest.json[.gz]` + a pre-compacted
        `jobs/<run>/ledger_snapshot.parquet` (pool_progress.py's exact mechanism — a fast
        footer-only read, no full scan, built for pool-scale runs with many workers).
      - Plain declared-manifest runs (`zenfleet-ctl declare-encodes` + `launch_fleet.sh` /
        the ad-hoc systemd/Nomad deploys this session used): `<run>/manifest.json`, no
        snapshot file at all — falls back to a real scan of `<run>/ledger/*.parquet`
        (distinct job_id count, latest-wins by ts). Slower than the footer read, but every
        run tested this session (up to ~30k ledger rows) resolved in well under a second;
        revisit with a real snapshot-writer if a run's ledger ever gets large enough for
        this to matter.
    """
    S3, _ = s3()
    B = bucket()
    declared = None
    for prefix in (f"jobs/{run}/", f"{run}/"):
        try:
            with S3.open_input_file(f"{B}/{prefix}manifest.json.gz") as f:
                import gzip, io
                declared = len(json.load(gzip.open(io.BytesIO(f.read()))))
            break
        except Exception:
            pass
        try:
            with S3.open_input_file(f"{B}/{prefix}manifest.json") as f:
                declared = len(json.loads(f.read()))
            break
        except Exception:
            pass
    if declared is None:
        return None
    done = 0
    try:
        with S3.open_input_file(f"{B}/jobs/{run}/ledger_snapshot.parquet") as f:
            done = pq.read_metadata(f).num_rows
    except Exception:
        try:
            import pyarrow.dataset as ds
            d = ds.dataset(f"{B}/{run}/ledger", filesystem=S3, format="parquet")
            latest = {}
            for r in d.to_table(columns=["job_id", "status", "ts"]).to_pylist():
                jid = r["job_id"]
                if jid not in latest or r["ts"] > latest[jid]["ts"]:
                    latest[jid] = r
            done = sum(1 for r in latest.values() if str(r["status"]).lower() == "done")
        except Exception:
            done = 0
    return declared, done


def load_state() -> dict:
    S3, _ = s3()
    B = bucket()
    try:
        with S3.open_input_file(f"{B}/{STATE_KEY}") as f:
            return json.loads(f.read())
    except Exception:
        return {}


def save_state(state: dict):
    S3, _ = s3()
    B = bucket()
    with S3.open_output_stream(f"{B}/{STATE_KEY}") as f:
        f.write(json.dumps(state, indent=2).encode())


def nomad(addr: str, *args: str) -> str:
    env = dict(os.environ, NOMAD_ADDR=addr)
    out = subprocess.run(["nomad", *args], env=env, capture_output=True, text=True, timeout=15)
    return out.stdout


def node_alloc_count(addr: str, node_name: str) -> int | None:
    """How many non-terminal allocations are currently placed on this Nomad node. None if
    the node isn't visible at all (down, or never joined) -- distinct from 0 (up + idle).

    BUG FOUND + FIXED 2026-08-24 (first real G-P3 test): `nomad node status -json
    -verbose <id>` does NOT have an "Allocs" key at all (verified directly: `'Allocs' in
    json.loads(...)` is False) -- the CLI's `-verbose` flag only adds an Events/Drivers
    table to the human-readable text output, not a JSON allocations list. The original
    `.get("Allocs", []) or []` silently returned empty every time, so this function ALWAYS
    returned 0 regardless of real running allocations -- a serious latent bug for the
    "sleep" decision (`gap<=50 AND alloc_n==0 AND dwell_ok`), which could have suspended a
    box while it genuinely still held work. Caught live: two real allocations were
    `running` per `nomad node status` at the exact moment this returned 0. Fixed by
    hitting the correct endpoint, `GET /v1/node/:id/allocations` (a flat list with
    `ClientStatus`, confirmed via curl), instead of guessing at the CLI's JSON shape.
    """
    try:
        nodes = json.loads(nomad(addr, "node", "status", "-json"))
    except Exception:
        return None
    node = next((n for n in nodes if n.get("Name") == node_name), None)
    if node is None:
        return None
    try:
        out = subprocess.run(
            ["curl", "-sf", f"{addr}/v1/node/{node['ID']}/allocations"],
            capture_output=True, text=True, timeout=10,
        )
        allocs = json.loads(out.stdout) if out.returncode == 0 else []
    except Exception:
        return 0
    return sum(1 for a in allocs if a.get("ClientStatus") == "running")


def is_reachable(ssh_target: str) -> bool:
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=4",
         "-o", "StrictHostKeyChecking=accept-new", ssh_target, "true"],
        capture_output=True, timeout=8,
    ).returncode == 0


def resolve_node_id(addr: str, node_name: str) -> str | None:
    """The Nomad node ID for a friendly node name (drain/eligibility need the ID, not the
    name). None if the node isn't visible (down, or never joined)."""
    try:
        nodes = json.loads(nomad(addr, "node", "status", "-json"))
    except Exception:
        return None
    node = next((n for n in nodes if n.get("Name") == node_name), None)
    return node["ID"] if node else None


def re_enable_eligibility(addr: str, node_id: str):
    """A drained node stays `Eligibility = ineligible` forever after the drain completes
    (Nomad's own semantics -- draining and eligibility are separate flags) -- confirmed
    live 2026-08-24 (gp2-suspend-test): i265 came back `ready` but `ineligible` after its
    drain finished and never received another allocation. Without this call, a box that
    goes through one sleep cycle would wake up, rejoin the cluster, and then sit idle
    forever as far as Nomad scheduling is concerned -- the single most important bug a
    real wake-cycle test could have caught. Called right after WoL: harmless to call
    before the node has actually rejoined (Nomad just applies it once the node reconnects).
    """
    subprocess.run(["nomad", "node", "eligibility", "-enable", node_id],
                    env=dict(os.environ, NOMAD_ADDR=addr), check=False)


def wol(mac: str):
    if os.path.isfile(FLEET_PXE) and os.access(FLEET_PXE, os.X_OK):
        subprocess.run([FLEET_PXE, "wol", mac], check=False)
    else:
        subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "root@tower",
             f"/usr/sbin/etherwake -i br0 '{mac}' 2>/dev/null || etherwake -i br0 '{mac}'"],
            check=False,
        )


def suspend(ssh_target: str, addr: str, node_id: str | None, power_mode: str = "suspend"):
    # -force is load-bearing, not optional: plain `-enable` (no -force) lets a running
    # allocation drain VOLUNTARILY -- it keeps running (and was observed, live, to even
    # pick up a NEW chunk claim after the drain command was issued) until it finishes on
    # its own or the deadline hits. Measured 2026-08-24 (gp2-suspend-test): a real
    # in-flight allocation took ~70s to stop under plain `-enable -deadline 2m`, not the
    # "seconds, not claim-TTL" the G-P2 gate requires -- SIGTERM-under-real-load itself is
    # fast (~3s, verified 7/7 via `nomad job stop` / `systemctl stop`), but plain drain
    # never delivers that signal promptly in the first place. `-force` makes Nomad
    # immediately move to stop the allocation (still via the task's normal kill_timeout
    # graceful-SIGTERM-then-SIGKILL sequence, not a raw bypass), which is what actually
    # gets the fast release. Suspending right after a slow, non-forced drain would freeze
    # the box mid-execution with the claim never released at all.
    if node_id:
        # -force and -deadline are mutually exclusive Nomad CLI flags ("can't be combined") --
        # -force alone needs no deadline (it means immediately, not "wait up to N then force").
        # Caught live 2026-08-24: the first version of this fix added -force NEXT TO the
        # existing -deadline 2m instead of replacing it, which errors at runtime -- and with
        # check=False here that failure is SILENT, so this line shipped broken once already.
        # Re-verify with a real `nomad node drain` call (not just reading the diff) after ANY
        # future edit to this argument list.
        subprocess.run(["nomad", "node", "drain", "-enable", "-force", "-self=false", "-yes",
                         node_id], env=dict(os.environ, NOMAD_ADDR=addr), check=False)
    # power_mode "suspend" (S3, default): fast resume (i265/r3500 measured 7-8s) but needs the
    # NIC armed for wake-from-S3 (arm_wol.sh) -- a real, repeat gap on PXE-netboot-every-boot
    # machines (r3500 needed it; i134/r5600g's G-P1 failed 2026-08-24 because it was skipped).
    # power_mode "poweroff" (S5): slower resume (full cold boot, tens of seconds+) but WoL-from-off
    # just worked with ZERO extra config on every box tried this fleet (i134, r5600g both cold-booted
    # cleanly via `fleet-pxe worker <mac>` the same session their S3 gate failed) -- S5-WoL is a
    # firmware-level feature that doesn't depend on OS/driver state surviving a netboot cycle the
    # way S3-arming does. Use poweroff for boxes where wake latency doesn't matter (opportunistic/
    # borrowed compute, not a dedicated low-latency worker) and where S3-arming is a recurring risk.
    cmd = ["sudo", "systemctl", "poweroff"] if power_mode == "poweroff" else ["sudo", "systemctl", "suspend"]
    subprocess.run(["ssh", "-o", "BatchMode=yes", ssh_target, *cmd], check=False, timeout=10)


def cmd_status(args):
    print(f"NEVER_SLEEP (untouchable): {sorted(NEVER_SLEEP)}")
    print()
    runs = args.run or []
    total_gap = 0
    for run in runs:
        g = queue_gap(run)
        if g is None:
            print(f"  run {run}: no manifest/snapshot yet")
            continue
        declared, done = g
        gap = max(declared - done, 0)
        total_gap += gap
        print(f"  run {run}: declared={declared} done={done} gap={gap}")
    if runs:
        print(f"  TOTAL gap across {len(runs)} run(s): {total_gap}")
    print()
    print("Sleep roster:")
    for name, box in SLEEP_ROSTER.items():
        up = is_reachable(box["ssh"])
        alloc_n = node_alloc_count(args.nomad_addr, box["nomad_node_name"]) if up else None
        gate = box["gate"]
        gate_str = f"PASS ({gate}s)" if gate is not None else "NOT GATED — refused for --apply"
        print(f"  {name:10s} up={up!s:5s} allocs={alloc_n} gate={gate_str}")


def decide(name: str, box: dict, up: bool, alloc_n: int | None, total_gap: int, state: dict) -> str | None:
    """Return 'wake' | 'sleep' | None. Pure function of current signals + persisted state —
    keep it that way so the hysteresis logic is unit-testable without touching hardware."""
    now = time.time()
    last_action = state.get(name, {}).get("last_action_ts", 0)
    dwell_ok = (now - last_action) >= MIN_AWAKE_DWELL_SECS
    if not up:
        if total_gap >= WAKE_GAP_THRESHOLD:
            return "wake"
        return None
    # up:
    if total_gap <= SLEEP_GAP_THRESHOLD and (alloc_n == 0) and dwell_ok:
        return "sleep"
    return None


def cmd_apply(args):
    runs = args.run or []
    total_gap = 0
    for run in runs:
        g = queue_gap(run)
        if g:
            declared, done = g
            total_gap += max(declared - done, 0)
    state = load_state()
    # Wake in measured-throughput order (highest first) — TODO: pull real numbers from
    # fleet/handicaps.toml once it carries a machine-readable cells_per_hour field (see
    # docs/status/fleet-orchestration-2026-08.md's Preconditions section for that plan);
    # for now, a fixed priority order recorded from the crossnode_2026-08-04 bench
    # (benchmarks/crossnode_2026-08-04/VERDICT.md): i265 > r3500 (r7900x/i134 excluded/
    # not-gated). Sleep order is just "whoever qualifies", order doesn't matter there.
    wake_priority = ["i265", "r3500", "r5900xt", "i134", "r5600g"]
    acted = []
    for name in wake_priority:
        box = SLEEP_ROSTER[name]
        if box["gate"] is None:
            continue  # refuse to touch an ungated box, per the module docstring
        up = is_reachable(box["ssh"])
        node_id = resolve_node_id(args.nomad_addr, box["nomad_node_name"]) if up else None
        alloc_n = node_alloc_count(args.nomad_addr, box["nomad_node_name"]) if up else None
        # Self-heal every tick, independent of the wake/sleep decision below: a drained
        # node stays `ineligible` FOREVER after the drain completes (Nomad's own
        # semantics — draining and eligibility are separate flags), so a box that just
        # woke from a sleep cycle would otherwise sit reachable-but-never-scheduled
        # indefinitely. Confirmed live 2026-08-24 (gp2-suspend-test) — see
        # re_enable_eligibility's docstring. Idempotent and cheap; safe to call on an
        # already-eligible node every tick.
        if up and node_id and args.live:
            re_enable_eligibility(args.nomad_addr, node_id)
        action = decide(name, box, up, alloc_n, total_gap, state)
        if action is None:
            continue
        acted.append((name, action))
        if args.live:
            if action == "wake":
                wol(box["mac"])
            elif action == "sleep":
                suspend(box["ssh"], args.nomad_addr, node_id, box.get("power_mode", "suspend"))
            state.setdefault(name, {})["last_action_ts"] = time.time()
            state[name]["last_action"] = action
        print(f"{'[LIVE]' if args.live else '[DRY-RUN]'} {name}: {action} (gap={total_gap}, up={up}, allocs={alloc_n}, node_id={node_id})")
    if not acted:
        print(f"no action needed (gap={total_gap})")
    if args.live:
        save_state(state)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("status", "apply"):
        sp = sub.add_parser(name)
        sp.add_argument("--run", action="append", help="job-system run id to read queue depth from (repeatable)")
        sp.add_argument("--nomad-addr", default=os.environ.get("NOMAD_ADDR", "http://192.168.50.44:4646"))
        if name == "apply":
            sp.add_argument("--live", action="store_true", help="actually call WoL/suspend (default: dry-run)")
    args = p.parse_args()
    {"status": cmd_status, "apply": cmd_apply}[args.cmd](args)


if __name__ == "__main__":
    main()
