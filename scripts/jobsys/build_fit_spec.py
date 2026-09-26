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
            "d2": "/var/tmp/rev4-featpot/p2/d2_mlp"}


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
    p.add_argument("--grid", choices=("p0", "p2", "d2"), default="p0")
    p.add_argument("--existing-root", type=Path, default=None)
    p.add_argument("--include-complete", action="store_true")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    for label, value in (("program", args.program_sha), ("data", args.data_sha)):
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            p.error(f"{label} SHA-256 must be lowercase hex")
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
