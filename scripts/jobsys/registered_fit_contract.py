"""Bind an external private cell contract to opaque program/data identities.

Private contracts travel in the data archive or the authorized harvester's
filesystem. Images carry only their SHA-256 and the data SHA, never contents.
"""
import hashlib
import json
import os
from pathlib import Path


def arg(argv,key):
    if argv.count(key)!=1:
        raise ValueError(f"registered cell needs exactly one {key}")
    return argv[argv.index(key)+1]


def load(argv,data_sha,meta):
    pin=meta.get("registered_fit",{})
    if (argv[0]!="subjective_cell.py" or pin.get("data_sha256")!=data_sha
            or pin.get("contract_sha256")!=arg(argv,"--contract-sha256")
            or "--strict-admission" not in argv or "--train-only" not in argv):
        raise ValueError("registered program/data/contract binding differs")
    path=Path(os.environ.get("ZEN_REGISTERED_FIT_CONTRACT",arg(argv,"--contract")))
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=pin["contract_sha256"]:
        raise ValueError("registered private contract changed")
    packet=json.loads(raw)
    if packet.get("schema")!="registered-training-cells-v1":
        raise ValueError("registered private contract schema")
    name=arg(argv,"--cell-id")
    cell=packet["cells"][name]
    cap=cell.get("wall_cap_sec")
    if (type(cap) is not int or cap<=0 or cell["selected_epoch"]!=119
            or cell["epochs"]!=120 or cell["pairs_per_epoch"]!=50000
            or cell["tier"]!="v3" or cell["formula_revision"]!=5
            or bool("--pack-production" in argv)!=(cell["route"]=="full")):
        raise ValueError("registered full-budget recipe/cap differs")
    expected=cell["qualification"]
    if (expected.get("schema")!="registered-fit-contract-v1"
            or expected.get("init_seed")!=cell["init_seed"] or expected.get("sample_seed")!=cell["sample_seed"]
            or expected.get("seed_index")!=cell["seed_index"]
            or expected.get("selected_epoch")!=119 or expected.get("epochs")!=120
            or expected.get("pairs_per_epoch")!=50000 or expected.get("execution_contract")!="registered-fit"):
        raise ValueError("registered decoded-checkpoint contract differs")
    return expected|dict(wall_cap_sec=cap,registered_cell=name)
