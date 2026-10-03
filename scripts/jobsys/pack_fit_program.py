#!/usr/bin/env python3
"""Pack the preregistered P0 program and current-main Rust binaries for the image."""

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile


SCRIPTS = {
    "scripts/rev4_featpot/mlp_probe.py": "4e7b18d0c5cd28f633af249111f2f06cb988db376b3429b3783a25f8e0e0617a",
    "scripts/rev4_featpot/data.py": "2906e383aa3891a94154cafdaec24a770da72baa64e5a0cbf689845455a7916c",
    "scripts/rev4_featpot/admit_bank.py": "e80ec845dd4d2c186d43e1cf76a23168ee240ab33ac137ffb31571bfa07cd4cf",
    "scripts/rev4_featpot/linear_probe.py": "1deef7b7d49b375b2ee51d32ca586b82a14b3c14ba5472451e4c0a7de86820b7",
    "scripts/rev4_featpot/mlp_importance.py": "6d28cdd9ecb1784be7194947e16287531354372436411a4511af0255bfafb559",
    "scripts/rev4_featpot/stability_lasso.py": "8dd081008595780ca49a40fd3e560d286bc9d783cef4c8b6a0e4ab2ab5fc62f2",
    "scripts/lib/zen_stats.py": "6e2bed69441195674e3a397f11a8eeb4b40e51dd271ebbd124e6702b68eb3d35",
}
# P2 (GMSD-peer MLP) and D2 (source-held-out MLP) program era: the potential lane's committed tip
# b168bba26ad0 (quarantine/codex/featbank-potential), files pinned by sha.
P2D2_SCRIPTS = {
    "scripts/rev4_featpot/admit_bank.py": "e80ec845dd4d2c186d43e1cf76a23168ee240ab33ac137ffb31571bfa07cd4cf",
    "scripts/rev4_featpot/data.py": "2906e383aa3891a94154cafdaec24a770da72baa64e5a0cbf689845455a7916c",
    "scripts/rev4_featpot/linear_probe.py": "24d7ac8b1e90170358f749d7fe12d09c7f911b4d4ae8775a179e6ed135e03a1d",
    "scripts/rev4_featpot/lodo_bvls.py": "43d2f445afe7658c14177870718629fcd5a3390082ae27ee22d5ca07e09f9081",
    "scripts/rev4_featpot/mlp_probe.py": "df6e6aebbced4debcb3e297707ec17b4bdb78786ecee70d0f57180271031d787",
    "scripts/rev4_featpot/p2_data.py": "d30149c9f322095d98e124388d441c7258176ba20ee8b9e5c0c70ec7d58111a8",
    "scripts/rev4_featpot/p2_lodo_mlp.py": "8bbca5b0f937979222df8ff21ff10d6fa4a897c2dce4cb39bf44c7c4b7c14745",
    "scripts/rev4_featpot/p2_mlp_importance.py": "aad42677c9fe93f79ab735c2b218755af0f7957aa3f2902b6a947fc993f4d1b5",
    "scripts/rev4_featpot/p2_mlp.py": "7fba1eb80e5345db0b05dab6bd4f7b842e95239871613d2e56ea3ff14be5b0b3",
    "scripts/rev4_featpot/stability_lasso.py": "8dd081008595780ca49a40fd3e560d286bc9d783cef4c8b6a0e4ab2ab5fc62f2",
    "scripts/lib/zen_stats.py": "6e2bed69441195674e3a397f11a8eeb4b40e51dd271ebbd124e6702b68eb3d35",
}
# Rev4 potential Instrument v2 (R915 sampling): zensim commit 1ee08bae (v2 scripts), e7248c65 (zen_stats)
# (benchmarks/rev4_featpot_v2_amendment_2026-09-30.md, revision R1 + erratum R1.1 two-family layout + revision R3:
# distance-oriented oracles, `<spec>@h<w>` human-leg weight), files pinned by sha. Its bin-dir is the R1.1 mixed set:
# v8 trainer + panel, bake_dial_refit from zensim 86fc02bb (admitted by
# benchmarks/rev4_featpot_v2_predictor_parity_2026-10-01.json). Previous pins (zensim cce9ab19): v2_common 1e22400d…,
# v2_lodo_mlp 6d7d2d60….
V2_SCRIPTS = {
    # zensim ecc69a1f: amendment R4 (EPOCH_RULE = "last", train_and_select), CANONTAB v2c --root support, R2.3 screen specs.
    "scripts/rev4_featpot/v2_common.py": "d3f610fb11e17cbbabeb7e1996ab0f2e05d0668dd8c40bd36447680bf33ffbbb",
    # zensim 538d3549 (TRAINEROPT): dev panels every 17th epoch under EPOCH_RULE == "last", sparse read_curve; the
    # trajectory and final weights do not depend on --log-every (gate: zensim benchmarks/traineropt_WORKLOG.md).
    "scripts/rev4_featpot/v2_lodo_mlp.py": "2d4b6781660db8a47a50548c9e7627e1fdbe5df208e2d585b35e77c67acdc643",
    # zensim 6ffcb814 + c482fddf2a9c (design log E13): curated SafeSyn teacher legs, recipe token ts<rule>; uuid-named scratch copy.
    "scripts/rev4_featpot/v2_teacher.py": "878357a8f343af5ef7974acf7cc7fc91f16ef915a687162fa4cb5ad4e7ae99b1",
    # Revision R2 confirmatory full-data fits (features only).
    "scripts/rev4_featpot/v2_confirm_fit.py": "96543c227318bfcfbdcf5ce8b494fb25d4149264c65761e4c6f7ab41c49205e0",
    # zensim e7248c65 (EFFAUDIT D8: render_indexed_jobs + rendered_jobs=; panel_batch unchanged).
    "scripts/lib/zen_stats.py": "68532bad3b3482bc734508183872d9253893cbacd9d3299d523894fc38158a6a",
    # Design log E5/E5b epoch-selection cells.
    "scripts/rev4_featpot/e5_epochs.py": "059e5eeb4d2efc167b9e99ab95e4f0b689f1f8935eb7c62cc78ad640ebce4f70",
}
# Program v20 (image zenfleet-worker:fit-v2-v20, 2026-10-03): v19 with the compliant E14 table (20 KADIS types; 6/9/10/15 are
# third-party generated and excluded, zensim 6c5d38fa) and the matching v2_teacher pin. v19 was retired before any E14 cell finished.
# Program v19 (image zenfleet-worker:fit-v2-v19, 2026-10-03): v18 + zensim d1587e29 scripts (design log E14: ko<w> KADIS
# ordinal ladder leg; e13 scoring shared) + the E14 ordinal table and manifest (V2_DATA). Specs without ko/ts run as v17.
# Program v18 (image zenfleet-worker:fit-v2-v18, 2026-10-03): v17 binaries + zensim 6ffcb814 scripts (design log E13: ts<rule>
# SafeSyn teacher curation, v2_teacher.py) + the E13 strata data file (V2_DATA). Specs without a ts token run exactly as v17.
# Program v17 (image zenfleet-worker:fit-v2-v17, 2026-10-02): v16 + zensim_mlp_train 605d20e0 from zensim e176cc1e (TRAINEROPT3:
# pair forward in one w1 walk + the next pair's forward fused into the Adam row walk; 25/25 cells byte-identical to f9d076c6).
# Program v16 (image zenfleet-worker:fit-v2-v16, 2026-10-02): v15 + zensim_mlp_train f9d076c6 from zensim f7270995 (TRAINEROPT2:
# group-lasso prox fused into the w1 Adam rows; 12/12 cells byte-identical to a5f40576) + zensim bfd22c24 sel:<id> subset specs.
# Program v15 (image zenfleet-worker:fit-v2-v15, 2026-10-02): v14 + zensim 324078f6 (set:/core+ specs resolve candidate arms
# from the pinned keep lists; v14 imported restore_data, which the program does not pack, so every set: cell failed).
# Program v14 (image zenfleet-worker:fit-v2-v14, 2026-10-01): v13 + zensim 34c41507 set:<groups> specs (E9″, no exempt core).
# Program v13 (image zenfleet-worker:fit-v2-v13, 2026-10-01): v12 + zensim a14a30d4 multi-group core specs (core+X+Y, E9′).
# Program v12 (image zenfleet-worker:fit-v2-v12, 2026-10-01): v11 + zensim 30c9ca7d block specs (r0-<block>, core,
# core+<x>; design log E9).
# Program v11 (image zenfleet-worker:fit-v2-v11, 2026-10-01): v10 binaries + zensim fb24f308 scripts (recipe tokens
# @h<w>:H<n>:gl<lambda>, design log E8; specs without tokens run exactly as before).
# Program v10 (image zenfleet-worker:fit-v2-v10, 2026-10-01): v9 + zensim_mlp_train from zensim 538d3549 (TRAINEROPT: fused
# Adam visits only kept rows; byte-identical final weights; sha a5f40576) + the v2_lodo_mlp pin below.
# Program v9 (93dc93d0, image zenfleet-worker:fit-v2-v9, 2026-10-01): these V2_SCRIPTS + v8 trainer/panel (zensim cafed5ca)
# + bake_dial_refit from a local merge of zensim main b926258e and pr/signedfeat a659715e, so bakes reading the
# SIGNEDFEAT columns f1825-f1852 load (zensim benchmarks/rev4_featpot_effaudit/v9_predictor_gate_2026-10-01.json).
# Data files a v2 program carries (--data NAME=PATH), pinned by sha. E13 strata: per-row codec/quality of the v2-canon SafeSyn
# fit table (zensim benchmarks/e13_safesyn_strata_2026-10-03.pointer.md; rebuilt by e13_teacher.py strata).
V2_DATA = {
    "data/e13/safesyn_fit_strata.npz": "6baf5d1b2cb012963cdfa19d94ea2717bca17066a49cf9660979669d5662929c",
    # E14 KADIS ordinal ladder table + its trainer manifest (zensim benchmarks/e14_kadis_ordinal_2026-10-03.pointer.md).
    "data/e14/kadis_ordinal.parquet": "ffc245a0e39bd85d7527a08fd96bdd93b9fb266ad0334d4401eca06c95812012",
    "data/e14/kadis_ordinal.parquet.manifest.json": "5b47e46181dfba7ab0995ec684cf2b9fcc632754f76bbf9471939398adcf5e31",
}
BINARIES = ("zensim_mlp_train", "bake_dial_refit", "panel")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def add_file(tar: tarfile.TarFile, path: Path, name: str) -> None:
    member = tar.gettarinfo(str(path), arcname=name)
    member.mtime = 0
    member.uid = member.gid = 0
    member.uname = member.gname = ""
    with path.open("rb") as stream:
        tar.addfile(member, stream)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--executor", type=Path, required=True)
    p.add_argument("--bin-dir", type=Path, required=True)
    p.add_argument("--build-meta", type=Path, required=True)
    p.add_argument("--profile", choices=("p0", "p2d2", "v2"), default="p0")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--data", action="append", default=[], metavar="NAME=PATH", help="a pinned V2_DATA file (profile v2)")
    args = p.parse_args()
    files = {}
    for name, expected in {"p0": SCRIPTS, "p2d2": P2D2_SCRIPTS, "v2": V2_SCRIPTS}[args.profile].items():
        path = args.source / name
        if digest(path) != expected:
            raise ValueError(f"preregistered fit source changed: {name}")
        files[name] = path
    given = dict(item.split("=", 1) for item in args.data)
    wanted = V2_DATA if args.profile == "v2" else {}
    if sorted(given) != sorted(wanted):
        raise ValueError(f"--data must name exactly {sorted(wanted)}")
    for name, expected in wanted.items():
        path = Path(given[name])
        if digest(path) != expected:
            raise ValueError(f"pinned data file changed: {name}")
        files[name] = path
    files["fit_cell_exec.py"] = args.executor
    for name in BINARIES:
        path = args.bin_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        files[f"bin/{name}"] = path
    metadata = json.loads(args.build_meta.read_text())
    if metadata.get("schema") != "fleet-fits-build-v1":
        raise ValueError("build metadata missing or wrong schema")
    inventory = {name: digest(path) for name, path in sorted(files.items())}
    metadata = {**metadata, "files": inventory}
    meta_bytes = json.dumps(metadata, sort_keys=True, indent=2).encode() + b"\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w") as tar:
            info = tarfile.TarInfo("build_meta.json")
            info.size = len(meta_bytes)
            info.mtime = 0
            tar.addfile(info, io.BytesIO(meta_bytes))
            for name, path in sorted(files.items()):
                add_file(tar, path, name)
    print(json.dumps({"sha256": digest(args.out), "bytes": args.out.stat().st_size,
                      "files": len(files), "out": str(args.out)}))


if __name__ == "__main__":
    main()
