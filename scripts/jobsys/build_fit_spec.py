#!/usr/bin/env python3
"""Write the registered P0 MLP fit grid as a zenfleet-ctl declare-fits spec.

This is a manifest adapter only. zenfleet-ctl owns job IDs and declaration;
the worker/ledger own claims and duplicate prevention.
"""

import argparse
import json
from pathlib import Path


SETS = ("kadid_train", "tid2013", "konfig_train", "konjnd_bpg_train",
        "cid22_a25", "aic3", "kadid_select", "konfig_val")
ARMS = ("r0", "minus_basic")
HIDDEN = (32, 128)
OUTERS = (0, 1, 2, 3, 4, None)
REPS = range(5)


D2_SOURCES = ("kadid_train", "tid2013", "konfig_train", "konjnd_bpg_train",
              "cid22_a25", "aic3", "kadid_select")  # lodo_bvls.SOURCES
D2_ARMS = ("p2", "p2_perm", "r0")
EXISTING = {"p0": "/var/tmp/rev4-featpot/fits", "p2": "/var/tmp/rev4-featpot/p2/mlp",
            "d2": "/var/tmp/rev4-featpot/p2/d2_mlp", "v2": "/var/tmp/rev4-featpot/v2/cells"}
# Rev4 potential Instrument v2 (zensim benchmarks/rev4_featpot_v2_amendment_2026-09-30.md, revision R1).
# The spec list comes from the same keep_lists.json the cells read (packed into the v2 data archive).
V2_KEEP_LISTS = Path("/var/tmp/rev4-featpot/v2/wide/keep_lists.json")
V2_SOURCES = ("kadid", "tid2013", "konfig", "cid22_a25", "aic3")
V2_HEADS = ("N", "F")
V2_SEEDS = range(10)
V2_CALIBRATION = ("r0", "oracle_lo", "oracle_hi", "minus_basic", "oracle_lo~p1", "oracle_lo~p2", "oracle_lo~p3")


def grid_p2():
    """accelerate_mlp.tasks('p2'): same shape as P0 with the GMSD-peer arms and p2_mlp.py."""
    for name in SETS:
        for arm in ("p2", "p2_perm"):
            for hidden in HIDDEN:
                for outer in OUTERS:
                    for rep in REPS:
                        tag = f"o{outer}" if outer is not None else "full"
                        argv = ["p2_mlp.py", "--set", name, "--arm", arm, "--hidden", str(hidden)]
                        if outer is not None:
                            argv += ["--outer", str(outer)]
                        argv += ["--rep", str(rep)]
                        yield {"name": f"POT_{name}_{arm}_mlp{hidden}/{tag}_r{rep}", "argv": argv}


def grid_d2():
    """accelerate_lodo_mlp.tasks(arm) for every arm: 3 arms x 2 hidden x 7 held-out x 5 reps."""
    for arm in D2_ARMS:
        for hidden in HIDDEN:
            for heldout in D2_SOURCES:
                for rep in REPS:
                    yield {"name": f"LODO_{arm}_mlp{hidden}/without_{heldout}_r{rep}",
                           "argv": ["p2_lodo_mlp.py", "--arm", arm, "--hidden", str(hidden),
                                    "--heldout", heldout, "--rep", str(rep)]}


def grid_v2(phase: str):
    """v2_lodo_mlp.py: spec x head x held-out source x seed; calibration specs are their own phase."""
    specs = list(json.loads(V2_KEEP_LISTS.read_text())["specs"])
    if not set(V2_CALIBRATION) <= set(specs):
        raise ValueError("keep lists lack the calibration specs")
    chosen = {"calibration": [x for x in specs if x in V2_CALIBRATION],
              "arms": [x for x in specs if x not in V2_CALIBRATION],
              "all": specs}[phase]
    for spec in chosen:
        for head in V2_HEADS:
            for heldout in V2_SOURCES:
                for seed in V2_SEEDS:
                    yield {"name": f"{spec}__{head}/without_{heldout}_s{seed}",
                           "argv": ["v2_lodo_mlp.py", "--spec", spec, "--head", head,
                                    "--heldout", heldout, "--seed-index", str(seed)]}


def grid():
    for name in SETS:
        for arm in ARMS:
            for hidden in HIDDEN:
                for outer in OUTERS:
                    for rep in REPS:
                        tag = f"o{outer}" if outer is not None else "full"
                        dest = f"POT_{name}_{arm}_mlp{hidden}/{tag}_r{rep}"
                        argv = ["mlp_probe.py", "--set", name, "--arm", arm,
                                "--hidden", str(hidden)]
                        if outer is not None:
                            argv += ["--outer", str(outer)]
                        argv += ["--rep", str(rep)]
                        yield {"name": dest, "argv": argv}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--program-sha", required=True)
    p.add_argument("--data-sha", required=True)
    p.add_argument("--grid", choices=("p0", "p2", "d2", "v2"), default="p0")
    p.add_argument("--v2-phase", choices=("calibration", "arms", "all"), default="calibration")
    p.add_argument("--existing-root", type=Path, default=None)
    p.add_argument("--include-complete", action="store_true")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    for label, value in (("program", args.program_sha), ("data", args.data_sha)):
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            p.error(f"{label} SHA-256 must be lowercase hex")
    if args.grid == "v2":
        all_cells = list(grid_v2(args.v2_phase))
        assert len(all_cells) == {"calibration": 700, "arms": 6800, "all": 7500}[args.v2_phase]
    else:
        all_cells = list({"p0": grid, "p2": grid_p2, "d2": grid_d2}[args.grid]())
        assert len(all_cells) == {"p0": 960, "p2": 960, "d2": 210}[args.grid]
    if args.existing_root is None:
        args.existing_root = Path(EXISTING[args.grid])
    complete = {cell["name"] for cell in all_cells
                if (args.existing_root / cell["name"] / "result.json").is_file()}
    cells = all_cells if args.include_complete else [cell for cell in all_cells if cell["name"] not in complete]
    spec = {"program_sha": args.program_sha, "data_sha": args.data_sha, "cells": cells}
    args.out.write_text(json.dumps(spec, indent=2) + "\n")
    print(json.dumps({"total": len(all_cells), "completed_local": len(complete),
                      "declared": len(cells), "out": str(args.out)}))


if __name__ == "__main__":
    main()
