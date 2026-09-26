#!/usr/bin/env python3
"""ZEN_EXEC for content-addressed Rev4 fit cells.

The worker owns claims, ledger writes and blob addressing. This process only
verifies one declared input bundle, runs one baked fit program, and emits its
complete output directory plus a file-hash receipt as deterministic tar bytes.
"""

import fcntl
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile


ROOT = Path("/scratch/fit-cell")
PROGRAM = Path("/opt/fleet-fits/program")
FIT_ROOT = Path("/var/tmp/rev4-featpot")
# Approved fit programs: (output root under FIT_ROOT, outer-fold importance script or None).
# The importance step is the local runners' second step (run_mlp_baseline.sh / run_p2_mlp.sh).
SCRIPTS = {
    "mlp_probe.py": ("fits", "mlp_importance.py"),
    "p2_mlp.py": ("p2/mlp", "p2_mlp_importance.py"),
    "p2_lodo_mlp.py": ("p2/d2_mlp", None),
}


# Program shas whose binaries carry tier parity (AVX-512 and scalar kernels reproduce AVX2 bit for bit,
# zensim main 322514dc and later). Empty: every fit program built so far (v6-v8) predates it, so on an
# AVX-512 host the executor refuses to run unless ZENSIM_MAX_TIER=v3 caps the trainer to the AVX2
# kernels. Keep in step with harvest_fit_cells.TIER_PARITY_PROGRAMS (test_fit_tools.py checks it).
TIER_PARITY_PROGRAMS = frozenset()


def cpu_has_avx512f() -> bool:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("flags"):
                return " avx512f" in line
    except OSError:
        pass
    return False


def effective_tier(program_sha: str, environ=os.environ, avx512f=None) -> dict:
    """Tier this cell will run at, recorded in the receipt; refuses an uncapped AVX-512 run of a
    program without tier parity (its numerics would differ from the AVX2 fleet's)."""
    avx512f = cpu_has_avx512f() if avx512f is None else avx512f
    capped = environ.get("ZENSIM_MAX_TIER") == "v3"
    if avx512f and not capped and program_sha not in TIER_PARITY_PROGRAMS:
        raise ValueError("this CPU has avx512f and ZENSIM_MAX_TIER is not v3: fit program "
                         f"{program_sha[:8]} has no tier parity, so it must run capped to the AVX2 kernels")
    return {"cpu_avx512f": avx512f, "max_tier_env": environ.get("ZENSIM_MAX_TIER", ""),
            "effective": "v3" if (capped or not avx512f) else "v4"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch_data(sha: str) -> Path:
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = ROOT / f"{sha}.lock"
    with lock.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        archive = ROOT / f"{sha}.tar.gz"
        if not archive.is_file() or digest(archive) != sha:
            temp = ROOT / f"{sha}.download-{os.getpid()}"
            local = os.environ.get("ZEN_FIT_DATA_LOCAL")
            if local:
                shutil.copyfile(local, temp)
            else:
                bucket = os.environ["ZEN_BUCKET"]
                run = os.environ["ZEN_RUN"].strip("/")
                endpoint = os.environ["ZEN_R2_ENDPOINT"]
                uri = f"s3://{bucket}/{run}/inputs/{sha}"
                subprocess.run(["s5cmd", "--endpoint-url", endpoint, "cp", uri, str(temp)], check=True,
                           stdout=subprocess.DEVNULL)  # stdout carries the output bytes
            if digest(temp) != sha:
                temp.unlink(missing_ok=True)
                raise ValueError("data archive SHA-256 differs from declared input")
            temp.replace(archive)
        extracted = ROOT / sha
        if not (extracted / ".verified").is_file():
            if extracted.exists():
                shutil.rmtree(extracted)
            staging = ROOT / f"{sha}.extract-{os.getpid()}"
            staging.mkdir()
            with tarfile.open(archive, "r:gz") as bundle:
                for member in bundle.getmembers():
                    path = Path(member.name)
                    if (path.is_absolute() or ".." in path.parts or not
                            (member.isdir() or member.isfile())):
                        raise ValueError("unsafe data archive member")
                bundle.extractall(staging, filter="data")
            inventory = json.loads((staging / "input_inventory.json").read_text())
            if inventory.get("schema") != "zenfleet-fit-data-v1":
                raise ValueError("bad fit data inventory")
            for name, expected in inventory["files"].items():
                if digest(staging / name) != expected:
                    raise ValueError(f"fit input component changed: {name}")
            (staging / ".verified").write_text(sha + "\n")
            staging.rename(extracted)
        return extracted


def bind_expected_paths(data: Path) -> None:
    Path("/var/tmp").mkdir(exist_ok=True)
    for link, target in ((FIT_ROOT, data / "rev4-featpot"),
                         (Path("/var/tmp/rev4-featbank"), data / "rev4-featbank"),
                         (Path("/var/tmp/gmsbank"), data / "gmsbank")):
        if not target.is_dir():
            continue
        if link.is_symlink():
            if link.resolve() != target.resolve():
                raise ValueError(f"unexpected link at {link}")
        elif link.exists():
            raise ValueError(f"existing non-link at {link}")
        else:
            link.symlink_to(target, target_is_directory=True)
    binary_dir = FIT_ROOT / "target/debug"
    binary_dir.mkdir(parents=True, exist_ok=True)
    for name in ("zensim_mlp_train", "bake_dial_refit", "panel"):
        link = binary_dir / name
        target = PROGRAM / "bin" / name
        if link.is_symlink():
            if link.resolve() != target.resolve():
                raise ValueError(f"unexpected binary link at {link}")
        elif link.exists():
            raise ValueError(f"existing binary at {link}")
        else:
            link.symlink_to(target)


def output_bytes(dest: Path, receipt: dict) -> bytes:
    receipt_path = dest / "fleet_receipt.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w") as tar:
            for path in sorted(dest.rglob("*")):
                if not path.is_file():
                    continue
                member = tar.gettarinfo(str(path), arcname=str(path.relative_to(FIT_ROOT)))
                member.mtime = 0
                member.uid = member.gid = 0
                member.uname = member.gname = ""
                with path.open("rb") as stream:
                    tar.addfile(member, stream)
    output = buf.getvalue()
    # The worker owns persistence of these bytes in the content-addressed
    # store. Keep only the shared verified inputs on scratch between cells;
    # a failed upload can safely rerun the deterministic fit.
    shutil.rmtree(dest)
    return output


def run_fit(job: dict) -> bytes:
    kind = job["kind"]
    if kind.get("kind") != "fit_cell":
        raise ValueError("fit executor accepts fit_cell only")
    program_sha = os.environ["ZEN_FIT_PROGRAM_SHA"]
    if kind["program_sha"] != program_sha:
        raise ValueError("fit program image SHA-256 differs from declared input")
    tier = effective_tier(program_sha)
    argv = kind["argv"]
    if (not isinstance(argv, list) or not argv or argv[0] not in SCRIPTS or
            not all(isinstance(arg, str) and arg for arg in argv)):
        raise ValueError("fit argv is invalid or names an unapproved program")
    argv_sha = hashlib.sha256(json.dumps(argv, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    if argv_sha != kind["argv_sha"]:
        raise ValueError("argv SHA-256 differs from declared input")
    if sorted(job["inputs"]) != sorted([program_sha, kind["data_sha"], argv_sha]):
        raise ValueError("fit input hash list differs from declaration")
    data = fetch_data(kind["data_sha"])
    bind_expected_paths(data)
    name = Path(job["cell"]["image_path"])
    if name.is_absolute() or ".." in name.parts or len(name.parts) != 2:
        raise ValueError("unsafe fit destination")
    root, importance_script = SCRIPTS[argv[0]]
    dest = FIT_ROOT / root / name
    if dest.exists():
        receipt_path = dest / "fleet_receipt.json"
        if receipt_path.is_file():
            old = json.loads(receipt_path.read_text())
            if (old.get("program_sha") == program_sha and old.get("data_sha") == kind["data_sha"]
                    and old.get("argv_sha") == argv_sha and
                    all(digest(dest / filename) == expected for filename, expected in old["files"].items())):
                return output_bytes(dest, old)
        shutil.rmtree(dest)
    env = dict(os.environ)
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONHASHSEED="0", RAYON_NUM_THREADS="4",
               PYTHONPYCACHEPREFIX="/scratch/pycache", ZEN_PANEL_BIN=str(PROGRAM / "bin/panel"),
               TMPDIR="/scratch/tmp")
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    result = subprocess.run([sys.executable, str(PROGRAM / "scripts/rev4_featpot" / argv[0]), *argv[1:]],
                            cwd=PROGRAM, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        raise RuntimeError(f"fit exited {result.returncode}: {result.stdout[-4000:]}")
    stdout = result.stdout
    if importance_script and "--outer" in argv:
        # The local runner (run_mlp_baseline.sh) follows every outer-fold fit with the
        # outer-fold permutation-importance step, which check_core_gates requires
        # (importance.json + perm_* files). Same flags, inference-only.
        importance = subprocess.run(
            [sys.executable, str(PROGRAM / "scripts/rev4_featpot" / importance_script), *argv[1:]],
            cwd=PROGRAM, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if importance.returncode:
            raise RuntimeError(f"importance exited {importance.returncode}: {importance.stdout[-4000:]}")
        if not (dest / "importance.json").is_file():
            raise ValueError("importance returned successfully without importance.json")
        stdout += importance.stdout
    (dest / "fleet_stdout.log").write_text(stdout)
    result_path = dest / "result.json"
    if not result_path.is_file():
        raise ValueError("fit returned successfully without result.json")
    value = json.loads(result_path.read_text())
    selected = Path(value["selected_bake"])
    if selected != dest / "refit" / selected.name or digest(selected) != value["selected_bake_sha256"]:
        raise ValueError("selected checkpoint receipt mismatch")
    files = {str(path.relative_to(dest)): digest(path) for path in sorted(dest.rglob("*")) if path.is_file()}
    receipt = {"schema": "zenfleet-fit-cell-receipt-v1",
               "program_sha": program_sha, "data_sha": kind["data_sha"], "argv_sha": argv_sha,
               "cell": str(name), "result_sha": files["result.json"],
               "selected_bake_sha": value["selected_bake_sha256"], "tier": tier, "files": files}
    return output_bytes(dest, receipt)


def main() -> None:
    if len(sys.argv) == 2 and sys.argv[1] == "capabilities":
        print("fit-cell-v1")
        return
    if len(sys.argv) != 1:
        raise ValueError("fit executor takes no arguments")
    job = json.load(sys.stdin)
    sys.stdout.buffer.write(run_fit(job))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"fit_cell_exec: {exc}", file=sys.stderr)
        raise
