#!/usr/bin/env python3
"""git_iterations.py — recount optimization iterations per metric family
from public git history (agy-effort.md definitions; CPU and GPU paths
counted as SEPARATE families).

Deterministic on origin/master at REV (recorded below). Read-only.

An optimization commit touches a family's source paths AND either has a
`perf`-typed subject or its message states a measured speed change.
An iteration is an optimization commit that cites verification evidence
(touches benchmarks/** or reports a measurement plus a parity/tolerance
check in the message). Exclusions per the brief: codecs, docs-only,
benchmark-data-only, CI-only, style/fmt and pure-rename commits.

Usage: git_iterations.py <git-repo-dir> [--tsv OUT]
"""
import json, re, subprocess, sys
from collections import defaultdict
from datetime import date

REV = "27979b23aff0f6e11be8866f6429c67d48beac89"
NN = "crates/zenmetrics-nn"
CLI = "crates/zenmetrics-cli/src/metrics"

# family -> (cpu source paths, gpu source paths).  Each -gpu crate is its
# own family row; everything else is the CPU row.  Paths prefixed "<repo>:"
# name a sibling repository (its own git log); plain paths are in zenmetrics.
FAMILIES = {
    "butteraugli": (["sibling:/home/lilith/work/butteraugli:butteraugli"],
                    ["crates/butteraugli-gpu"]),
    "butteraugli-original": ([f"{CLI}/butteraugli_original.rs"], []),
    "ssim2": (["sibling:/home/lilith/work/zen/fast-ssim2:src"], ["crates/ssim2-gpu"]),
    "dssim": (["sibling:/home/lilith/work/dssim:dssim-core"], ["crates/dssim-gpu"]),
    "cvvdp": (["crates/cvvdp", "crates/cvvdp-conformance"], ["crates/cvvdp-gpu"]),
    "iwssim": (["crates/iwssim"], ["crates/iwssim-gpu"]),
    "hdrvdp": (["crates/hdrvdp"], []),
    "gmsd": (["crates/gmsd"], []),
    "vmaf": (["crates/vmaf"], []),
    "psnrhvs": (["crates/psnrhvs"], []),
    "haarpsi": (["crates/haarpsi"], []),
    "fsim": (["crates/fsim"], []),
    "vsi": (["crates/vsi"], []),
    "ssim": ([f"{CLI}/classical.rs", "crates/msssim/src/float_ssim.rs"], []),
    "msssim": (["crates/msssim"], []),
    "vif": (["crates/vif"], []),
    "mad": (["crates/mad-iqa"], []),
    "mdctpsnr": (["crates/mdctpsnr"], []),
    "psnr": ([f"{CLI}/classical.rs"], []),
    "nlpd": (["crates/nlpd"], []),
    "ciede2000": ([f"{CLI}/classical.rs"], []),
    "cw-ssim": ([f"{CLI}/cw_ssim.rs", f"{CLI}/cw_fft.rs"], []),
    "ssimulacra1": ([f"{CLI}/ssimulacra1.rs"], []),
    "cambi": (["crates/vmaf/src/cambi.rs"], []),
    "ws-hv": ([f"{CLI}/ws_hv.rs"], []),
    "flip": ([f"{CLI}/flip.rs"], []),
    "zensim": (["sibling:/home/lilith/work/zen/zensimx:"], ["crates/zensim-gpu"]),
    "wd": ([f"{NN}/src/wd.rs", f"{NN}/models/wd"], []),
    "pieapp": ([f"{NN}/src/pieapp.rs", f"{NN}/models/pieapp"], []),
    "lpips": ([f"{NN}/src/lpips.rs", f"{NN}/models/lpips", f"{NN}/models/stlpips"], []),
    "dists": ([f"{NN}/src/dists.rs", f"{NN}/models/dists"], []),
    "deepdc": ([f"{NN}/src/deepdc.rs", f"{NN}/models/deepdc"], []),
    "ckdn": ([f"{NN}/src/ckdn.rs", f"{NN}/models/ckdn"], []),
    "dreamsim": ([f"{NN}/src/dreamsim.rs", f"{NN}/models/dreamsim"], []),
    "afine": ([f"{NN}/src/afine.rs", f"{NN}/models/afine"], []),
    "ahiq": ([f"{NN}/src/ahiq.rs", f"{NN}/models/ahiq"], []),
    "dmm": ([f"{NN}/src/dmm.rs", f"{NN}/models/dmm"], []),
    "wadiqam": ([f"{NN}/src/wadiqam.rs", f"{NN}/models/wadiqam"], []),
    "brisque": ([f"{NN}/src/brisque.rs", f"{NN}/models/brisque"], []),
    "niqe": ([f"{NN}/src/niqe.rs", f"{NN}/models/niqe"], []),
    "piqe": ([f"{NN}/src/piqe.rs", f"{NN}/models/piqe"], []),
    "arniqa": ([f"{NN}/src/arniqa.rs", f"{NN}/models/arniqa"], []),
    "musiq": ([f"{NN}/src/musiq.rs", f"{NN}/models/musiq"], []),
    "maniqa": ([f"{NN}/src/maniqa.rs", f"{NN}/models/maniqa"], []),
    "liqe": ([f"{NN}/src/liqe.rs", f"{NN}/models/liqe"], []),
    "hyperiqa": ([f"{NN}/src/hyperiqa.rs", f"{NN}/models/hyperiqa"], []),
    "paq2piq": ([f"{NN}/src/paq2piq.rs", f"{NN}/models/paq2piq"], []),
    # shared NN infra — every NN family shares these; tracked as its own row
    "nn-infra": ([f"{NN}/src/engine.rs", f"{NN}/src/preprocess.rs",
                  f"{NN}/src/resize.rs", f"{NN}/src/tfresize.rs",
                  f"{NN}/src/common.rs"], []),
}

CLI_NAMES = {
    "butteraugli": "butteraugli", "butteraugli-original": "butteraugli-original",
    "ssim2": "ssim2", "dssim": "dssim", "cvvdp": "cvvdp",
    "iwssim": "iwssim,iwssim-piq", "hdrvdp": "hdrvdp,hdrvdp3", "gmsd": "gmsd,mdsi,ms-gmsd,ms-gmsdc",
    "vmaf": "vmaf,vmaf-neg", "psnrhvs": "psnrhvs-y,psnrhvs,psnrhvs-daala",
    "haarpsi": "haarpsi-y,haarpsi", "fsim": "fsim-y,fsim,fsim-k4",
    "vsi": "vsi,vsi-k4", "ssim": "ssim,ssimc,ssim-libvmaf",
    "msssim": "msssim,msssim-libvmaf", "vif": "vif,vifvec",
    "mad": "mad", "mdctpsnr": "mdctpsnr", "psnr": "psnr,psnr-y,psnr-ycbcr611,psnr-cb,psnr-cr",
    "nlpd": "nlpd,nlpd-iqa", "ciede2000": "ciede2000", "cw-ssim": "cw-ssim",
    "ssimulacra1": "ssimulacra1", "cambi": "cambi", "ws-hv": "ws-hv",
    "flip": "flip,flip-v17", "zensim": "zensim",
    "wd": "wd", "pieapp": "pieapp", "lpips": "lpips-alex,lpips-vgg,lpips-squeeze,stlpips-alex,stlpips-vgg",
    "dists": "dists", "deepdc": "deepdc", "ckdn": "ckdn", "dreamsim": "dreamsim",
    "afine": "afine", "ahiq": "ahiq", "dmm": "dmm",
    "wadiqam": "wadiqam-fr-live,wadiqam-fr-tid,wadiqam-nr-live,wadiqam-nr-tid",
    "brisque": "brisque", "niqe": "niqe", "piqe": "piqe", "arniqa": "arniqa",
    "musiq": "musiq", "maniqa": "maniqa", "liqe": "liqe", "hyperiqa": "hyperiqa",
    "paq2piq": "paq2piq", "nn-infra": "(shared)",
}

# exclusions (the brief): codecs, docs/bench-data/CI/style/rename-only
EXCLUDE_PATH = re.compile(
    r"(^docs/|^\.github/|^benchmarks/.*\.(png|jpg|jpeg|mp4|csv|tsv|json|jsonl)$"
    r"|codec|encoder|decoder|/sweep/)")
EXCLUDE_SUBJECT = re.compile(r"^(style:|docs:|chore\(ci|ci:|build:|test:|fix\(ci)", re.I)
OPT_SUBJECT = re.compile(r"^perf(\(|:|\s)", re.I)
OPT_MEASURE = re.compile(r"(\d+(\.\d+)?\s*[×x]\b|x faster|speedup|faster by|\b\d+(\.\d+)?\s*ms\b|"
                         r"\bfaster\b|\bslower\b|gb/s|mpx/s|mpix|throughput|cycles)", re.I)
EVIDENCE_MSG = re.compile(r"(parity|bit[- ]ident|identical|tolerance|oracle|within \d|golden)", re.I)


def is_optimization(subj, body):
    return bool(OPT_SUBJECT.search(subj)) or bool(
        OPT_MEASURE.search(subj) or OPT_MEASURE.search(body))


def main():
    repo = sys.argv[1]
    out = None
    if "--tsv" in sys.argv:
        out = sys.argv[sys.argv.index("--tsv") + 1]

    # collect logs from zenmetrics + any sibling repos named in FAMILIES
    repos = {"": repo}
    for cpu, gpu in FAMILIES.values():
        for pth in cpu + gpu:
            if pth.startswith("sibling:"):
                _, rdir, sub = pth.split(":", 2)
                repos[rdir] = rdir
    logs = {}
    for rkey, rpath in repos.items():
        r = rpath or repo
        br = subprocess.run(["git", "-C", r, "rev-parse", "--verify",
                             "origin/master"], capture_output=True, text=True)
        ref = "origin/master" if br.returncode == 0 else "HEAD"
        logs[rkey] = subprocess.run(
            ["git", "-C", r, "log", "--format=%H%x1f%aI%x1f%s%x1f%B%x1e",
             "--name-only", ref], capture_output=True, text=True).stdout
        print(f"# log {r} ref={ref} bytes={len(logs[rkey])}", file=sys.stderr)

    # git prints <format>\x1e then a blank line then --name-only files, so the
    # \x1e-delimited chunk is "<prev files>\n\n<header>". Split each chunk on
    # the 40-hex sha to recover both parts.
    commits = []
    prev_files = []
    chunks = []
    for rdir, lg in logs.items():
        for chunk in lg.split("\x1e"):
            chunks.append((rdir, chunk))
    for rdir, chunk in chunks:
        m = re.search(r"[0-9a-f]{40}\x1f", chunk)
        if m is None:
            prev_files += [f for f in chunk.splitlines() if f]
            continue
        hdr = chunk[m.start():]
        if commits:
            commits[-1]["files"] = prev_files
        prev_files = [f for f in chunk[:m.start()].splitlines() if f]
        sha, when, subj, body = (hdr.split("\x1f") + ["", "", ""])[:4]
        commits.append({"repo": rdir, "sha": sha, "date": when[:10],
                        "subj": subj.strip(), "body": body, "files": []})
    if commits:
        commits[-1]["files"] = prev_files

    # path -> family rows (cpu row = family, gpu row = family+"-gpu")
    rows = {}  # row -> dict of counts
    def row(fam, gpu):
        key = fam + ("-gpu" if gpu else "")
        return rows.setdefault(key, {
            "family": key, "total": 0, "opt": 0, "iters": 0,
            "days": set(), "first": None, "last": None,
            "opt_shas": set(), "iter_shas": [],
        })

    for fam, (cpu, gpu) in FAMILIES.items():
        row(fam, False)
        if gpu:
            row(fam, True)

    commit_fams = defaultdict(set)
    for c in commits:
        files = c["files"]
        # file-level exclusions first
        code_files = [f for f in files if not EXCLUDE_PATH.search(f)]
        if not code_files:
            continue  # docs/bench-data/CI-only commit
        if EXCLUDE_SUBJECT.match(c["subj"]):
            continue
        matched = set()
        for f in code_files:
            for fam, (cpu, gpu) in FAMILIES.items():
                for p in cpu:
                    if p.startswith("sibling:"):
                        _, rdir, sub = p.split(":", 2)
                        if c["repo"] == rdir and (f.startswith(sub) or not sub):
                            matched.add(fam)
                    elif not c["repo"] and (f.startswith(p) or f == p):
                        matched.add(fam)
                for p in gpu:
                    if not c["repo"] and f.startswith(p):
                        matched.add(fam + "-gpu")
        if not matched:
            continue
        opt = is_optimization(c["subj"], c["body"])
        it = opt and (any(f.startswith("benchmarks/") for f in files)
                      or bool(EVIDENCE_MSG.search(c["body"]) or EVIDENCE_MSG.search(c["subj"])))
        for fam in matched:
            r = rows[fam]
            r["total"] += 1
            r["last"] = r["last"] or c["date"]
            r["first"] = c["date"]
            if opt:
                r["opt"] += 1
                r["days"].add(c["date"])
                r["opt_shas"].add(c["sha"])
            if it:
                r["iters"] += 1
                r["iter_shas"].append(c["sha"])
            commit_fams[c["sha"]].add(fam)

    # shared = optimization commit counted under >1 family row
    shared = {fam: 0 for fam in rows}
    for sha, fams in commit_fams.items():
        opt_fams = [f for f in fams if sha in rows[f]["opt_shas"]]
        if len(opt_fams) > 1:
            for f in opt_fams:
                shared[f] += 1

    lines = [
        "# git iteration tally — swe2 recount, definitions from agy-effort.md",
        f"# origin/master recorded sha: {REV}",
        "# family->paths map: see FAMILIES dict in scripts/complexity/git_iterations.py",
        "# CPU and GPU implementations are separate family rows (-gpu suffix).",
        "# family\tcli_names\tsource_paths\tcommits_total\toptimization_commits\titerations\t"
        "optimization_days\tfirst_commit_date\tlast_commit_date\tshared_optimization_commits\t"
        "example_iteration_commits",
    ]
    for fam, (cpu, gpu) in FAMILIES.items():
        for key, paths in ((fam, cpu), (fam + "-gpu", gpu)):
            if not paths:
                continue
            r = rows[key]
            ex = ",".join(s[:10] for s in r["iter_shas"][-3:])
            lines.append("\t".join([
                key, CLI_NAMES.get(fam, fam), ";".join(paths),
                str(r["total"]), str(r["opt"]), str(r["iters"]),
                str(len(r["days"])), r["first"] or "-", r["last"] or "-",
                str(shared.get(key, 0)), ex or "-",
            ]))
    text = "\n".join(lines) + "\n"
    if out:
        open(out, "w").write(text)
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
