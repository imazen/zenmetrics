"""Bindings for the canonical byc_cells runner; zenfleet owns scheduling."""

import json
from pathlib import Path
import re


def identity(argv):
    if argv[:2] != ["byc_cells.py", "run"]:
        raise ValueError("one canonical byc run required")
    flags = ("--arm", "--label", "--folds", "--seeds", "--jobs")
    if len(argv) != 2 + 2 * len(flags) or any(argv.count(k) != 1 for k in flags):
        raise ValueError("byc requires exactly one arm, label, fold, seed and job")
    values = {}
    for i in range(2, len(argv), 2):
        if argv[i] not in flags:
            raise ValueError("unknown byc argument")
        values[argv[i]] = argv[i + 1]
    label, fold, seed = (values[k] for k in ("--label", "--folds", "--seeds"))
    if (not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", label)
            or fold not in ("kadid", "tid2013", "konfig", "cid22_a25", "pipal", "pairs", "mos7", "full")
            or seed not in ("0", "1", "2") or values["--jobs"] != "1"
            or values["--arm"] not in ("mono:thr:mp32", "mono:thr:mp64")):
        raise ValueError("unsupported byc scientific cell")
    return label, fold, int(seed)


def environment(argv, data, program, fit_root, ensure_link):
    label, fold, seed = identity(argv)
    inventory = json.loads((data / "input_inventory.json").read_text())
    binding = inventory["byc"]
    if binding["schema"] != "byc-fit-input-v1" or binding["label"] != label:
        raise ValueError("byc input arm differs")
    for alias, member in binding["bindings"].items():
        link, rel = Path(alias), Path(member)
        target = data / rel
        if (not link.is_absolute() or ".." in link.parts
                or not (link.is_relative_to("/var/tmp") or link.is_relative_to("/mnt/v/output"))
                or rel.is_absolute() or ".." in rel.parts
                or not rel.is_relative_to("rev4-featpot/byc/inputs")
                or not target.is_dir()):
            raise ValueError("unsafe byc input binding")
        link.parent.mkdir(parents=True, exist_ok=True)
        ensure_link(link, target, "byc input")
    if any(binding[k] not in binding["bindings"] for k in ("root", "score_root")):
        raise ValueError("unbound byc root")
    return dict(BYC_ROOT=binding["root"], BYC_DATASET_SCORE_ROOT=binding["score_root"],
                BYC_CELLS=str(fit_root / "byc/cells"), BYC_COLS=str(program / "byc_columns.txt"),
                BYC_SCRATCH=str(Path("/scratch/tmp/byc") / label / f"{fold}_s{seed}"),
                REV4_V2_BIN_DIR=str(program / "bin"), REV4_V2_ROOT=binding["root"],
                RAYON_NUM_THREADS="1", ZENSIM_MAX_TIER="v3")


def verify_output(dest, argv, digest):
    label, fold, seed = identity(argv)
    rec = json.loads((dest / "byc_record.json").read_text())
    result = json.loads((dest / "result.json").read_text())
    if (dest.parts[-2:] != (label, f"without_{fold}_s{seed}")
            or (rec["fold"], rec["seed"], rec["tokens"]) != (fold, seed, argv[argv.index("--arm") + 1])
            or rec["last_bin_sha256"] != digest(dest / "refit/last.bin")
            or result["epochs"] != 120 or result["pairs_per_epoch"] != 50000
            or result["selection"]["selected_epoch"] != 119 or result["seed_index"] != seed):
        raise ValueError("byc output scientific binding differs")
    if fold in ("kadid", "tid2013", "konfig", "cid22_a25"):
        expected = {f"pred_{fold}.tsv": rec["pred_sha256"]}
    else:
        expected = {f"pred_dataset_{c}.tsv": v["prediction_sha256"]
                    for c, v in rec["dataset_train_statistics"].items()}
    if not expected or any(digest(dest / name) != pin for name, pin in expected.items()):
        raise ValueError("byc prediction inventory differs")
    for name in ("LABEL_READS.json", "DATASET_GROUP_SELECTION.json"):
        if not (dest / name).is_file():
            raise ValueError("byc missing pre-read/exclusion ledger")
