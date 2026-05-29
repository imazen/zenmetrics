#!/usr/bin/env python3
"""Task #139 — assemble the unified CPU metric benchmark table.

Joins:
  - heaptrack process-peak TSV (benchmarks/heaptrack/refresh_2026-05-28/heaptrack_peaks.tsv)
      keyed (metric, mode, size) — ONE process-peak per mode execution.
  - zenbench wall TSV (benchmarks/cpu_wall_2026-05-28.tsv)
      keyed (size, metric, mode, cold_or_warm) — mean per-call ms.

Emits:
  - benchmarks/cpu_metrics_full_table_2026-05-28.tsv
  - docs/CPU_BENCHMARK_TABLE_2026-05-28.md

Row model (per task #139):
  - full / strip            : one row, cold_or_warm=cold.
  - warm_ref / warm_ref_strip: TWO rows — cold (first call incl precompute) +
    warm (amortized). The heaptrack process-peak is the peak of the mode's
    full single-call execution (build ref + score); it is attached to BOTH
    rows. For the warm row this is an upper bound on steady-state working set
    (the warmed reference is already resident; the driver does not isolate a
    warm-only allocation profile). Noted in the row.
  - dssim strip / warm_ref_strip: NOT_SUPPORTED (dssim-core 3.4 has no strip
    walker).
"""
import os
import sys

ROOT = "/home/lilith/work/zen/zenmetrics--cpu-bench-refresh"
HT = f"{ROOT}/benchmarks/heaptrack/refresh_2026-05-28/heaptrack_peaks.tsv"
WALL = f"{ROOT}/benchmarks/cpu_wall_2026-05-28.tsv"
OUT_TSV = f"{ROOT}/benchmarks/cpu_metrics_full_table_2026-05-28.tsv"
OUT_MD = f"{ROOT}/docs/CPU_BENCHMARK_TABLE_2026-05-28.md"

# Provenance — passed in via env or hardcoded from the run.
GIT_COMMIT = os.environ.get("GIT_COMMIT", "UNKNOWN")
HOST = "lilith (AMD Ryzen 9 7950X 16-core, 32-thread; WSL2, ~49 GB usable)"
ZENBENCH_VER = "0.1.8"
HEAPTRACK_VER = "1.3.0"
RUSTC = "1.95.0"
DATE = "2026-05-28"

SIZES = ["512", "1024", "2K", "12MP", "30MP"]
SIZE_MP = {"512": 0.262, "1024": 1.049, "2K": 4.194, "12MP": 12.0, "30MP": 30.0}
SIZE_LABELS = {
    "512": "512^2", "1024": "1024^2", "2K": "2K", "12MP": "12 MP", "30MP": "30 MP",
}
METRICS = ["cvvdp", "ssim2", "dssim", "butter", "iwssim", "zensim"]
# modes a metric offers (driver-verified)
MODES = {
    "cvvdp": ["full", "strip", "warm_ref", "warm_ref_strip"],
    "ssim2": ["full", "strip", "warm_ref", "warm_ref_strip"],
    "dssim": ["full", "warm_ref"],  # strip/warm_ref_strip NOT_SUPPORTED
    "butter": ["full", "strip", "warm_ref", "warm_ref_strip"],
    "iwssim": ["full", "strip", "warm_ref", "warm_ref_strip"],
    "zensim": ["full", "strip", "warm_ref", "warm_ref_strip"],
}
DSSIM_MISSING = [("dssim", "strip"), ("dssim", "warm_ref_strip")]


def load_ht():
    """(metric,mode,size) -> dict(peak_bytes, peak_human, rss_human, score)."""
    d = {}
    with open(HT) as f:
        hdr = f.readline().rstrip("\n").split("\t")
        idx = {k: i for i, k in enumerate(hdr)}
        for line in f:
            p = line.rstrip("\n").split("\t")
            key = (p[idx["metric"]], p[idx["mode"]], p[idx["size_label"]])
            d[key] = {
                "peak_bytes": p[idx["peak_heap_bytes"]],
                "peak_human": p[idx["peak_heap_human"]],
                "rss_human": p[idx["peak_rss_human"]],
                "score": p[idx["score"]],
            }
    return d


def load_wall():
    """(size,metric,mode,cold_or_warm) -> dict(mean_ms, n_rounds, score)."""
    d = {}
    with open(WALL) as f:
        hdr = f.readline().rstrip("\n").split("\t")
        idx = {k: i for i, k in enumerate(hdr)}
        for line in f:
            p = line.rstrip("\n").split("\t")
            key = (
                p[idx["size_label"]], p[idx["metric"]],
                p[idx["mode"]], p[idx["cold_or_warm"]],
            )
            d[key] = {
                "mean_ms": p[idx["mean_ms"]],
                "n_rounds": p[idx["n_rounds"]],
                "score": p[idx["score"]],
            }
    return d


def human_bytes(b):
    try:
        b = int(b)
    except (ValueError, TypeError):
        return "NOT_SUPPORTED"
    for unit, div in (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
        if b >= div:
            return f"{b / div:.2f} {unit}"
    return f"{b} B"


def main():
    ht = load_ht()
    wall = load_wall()
    rows = []  # unified TSV rows

    for size in SIZES:
        w, h = {
            "512": (512, 512), "1024": (1024, 1024), "2K": (2048, 2048),
            "12MP": (4000, 3000), "30MP": (6000, 5000),
        }[size]
        mp = SIZE_MP[size]
        for metric in METRICS:
            for mode in MODES[metric]:
                htk = ht.get((metric, mode, size), {})
                peak_b = htk.get("peak_bytes", "NA")
                peak_h = htk.get("peak_human", "NA")
                score = htk.get("score", "-")
                is_warm_mode = mode in ("warm_ref", "warm_ref_strip")
                if not is_warm_mode:
                    # full / strip → single cold row
                    wk = wall.get((size, metric, mode, "cold"), {})
                    wall_ms = wk.get("mean_ms", "NA")
                    note = ""
                    if metric == "cvvdp" and mode == "strip":
                        note = "Path A walker: only pool stage strips; peak==full (documented)"
                    rows.append([
                        metric, mode, SIZE_LABELS[size], w, h, f"{mp:.3f}",
                        "cold", wall_ms, peak_b, peak_h,
                        wk.get("score", score), note,
                    ])
                else:
                    # warm modes → cold + warm rows; heaptrack peak on both
                    wkc = wall.get((size, metric, mode, "cold"), {})
                    wkw = wall.get((size, metric, mode, "warm"), {})
                    rows.append([
                        metric, mode, SIZE_LABELS[size], w, h, f"{mp:.3f}",
                        "cold", wkc.get("mean_ms", "NA"), peak_b, peak_h,
                        wkc.get("score", score),
                        "first call incl one-time precompute",
                    ])
                    rows.append([
                        metric, mode, SIZE_LABELS[size], w, h, f"{mp:.3f}",
                        "warm", wkw.get("mean_ms", "NA"), peak_b, peak_h,
                        wkw.get("score", score),
                        "amortized per-call (ref cached); heaptrack peak = cold-path process peak (upper bound)",
                    ])
        # dssim NOT_SUPPORTED rows
        for (mm, md) in DSSIM_MISSING:
            rows.append([
                mm, md, SIZE_LABELS[size], w, h, f"{mp:.3f}",
                "-", "NOT_SUPPORTED", "NOT_SUPPORTED", "NOT_SUPPORTED",
                "-", "dssim-core 3.4 has no strip walker",
            ])

    # write unified TSV
    cols = ["metric", "mode", "size_label", "size_w", "size_h", "size_mp",
            "cold_or_warm", "wall_ms", "peak_heap_bytes", "peak_heap_human",
            "score", "notes"]
    with open(OUT_TSV, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    print(f"wrote {OUT_TSV} ({len(rows)} rows)")

    write_md(rows, ht, wall)


def fmt_ms(v):
    if v in ("NA", "NOT_SUPPORTED", ""):
        return v
    try:
        return f"{float(v):.2f}"
    except ValueError:
        return v


def write_md(rows, ht, wall):
    L = []
    L.append("# CPU Metric Benchmark Table — 2026-05-28")
    L.append("")
    L.append("Full CPU wall-time + peak-heap table for the 6 CPU metrics across")
    L.append("5 sizes x every mode each metric offers x cold/warm. Supersedes the")
    L.append("stale `benchmarks/heaptrack/stats.tsv` (which had GAP strip stubs,")
    L.append("pre-0.9.4 butteraugli, and pre-Path-A cvvdp strip).")
    L.append("")
    L.append("## Provenance")
    L.append("")
    L.append(f"- **Date:** {DATE}")
    L.append(f"- **Git commit:** `{GIT_COMMIT}`")
    L.append(f"- **Host:** {HOST}")
    L.append(f"- **Build:** `cargo build --release` (workspace profile: opt-level=3, "
             "thin-LTO, codegen-units=16). **NO `-C target-cpu=native`** — runtime "
             "SIMD dispatch is what users get (per CLAUDE.md).")
    L.append(f"- **Wall times:** zenbench {ZENBENCH_VER} (interleaved round-robin, "
             "paired stats; NOT criterion, NOT the heaptrack-instrumented runtime). "
             "16 rounds/cell at 512/1024, 14 at 2K, 12 at 12MP, 10 at 30MP; "
             "all 32 cells per size interleave in one group.")
    L.append(f"- **Peak heap:** heaptrack {HEAPTRACK_VER}, PROCESS peak "
             "(`peak heap memory consumption`), NOT top-callstack (top-callstack "
             "misreports per task #130). One process per (metric, mode, size).")
    L.append(f"- **rustc:** {RUSTC}")
    L.append("- **butteraugli:** 0.9.4 (via `[patch.crates-io]` local sibling). "
             "**fast-ssim2:** 0.8.1. **dssim-core:** 3.4.0. **cvvdp/iwssim/zensim:** "
             "local workspace/sibling crates at the commit above.")
    L.append("- **Synthetic input:** deterministic per-pixel pattern (ref) + fixed "
             "channel offset (dist), identical between the wall and heaptrack drivers.")
    L.append("- **Strip body height:** 512 rows for all strip modes "
             "(cvvdp/iwssim STRIP_H_BODY_DEFAULT; ssim2/butter take an explicit height).")
    L.append("")
    L.append("### Commands")
    L.append("")
    L.append("```bash")
    L.append("# Driver (real crate APIs for every (metric,mode)):")
    L.append("cargo build --release   # in benchmarks/heaptrack/drivers/cpu_profile")
    L.append("# Peak heap:")
    L.append("bash benchmarks/heaptrack/refresh_2026-05-28/run_heaptrack_sweep.sh \\")
    L.append("     benchmarks/heaptrack/refresh_2026-05-28")
    L.append("# Wall (per size):")
    L.append("target/release/cpu-wall <512|1024|2K|12MP|30MP> benchmarks/cpu_wall_2026-05-28.tsv")
    L.append("# Assemble:")
    L.append("python3 benchmarks/heaptrack/refresh_2026-05-28/assemble_table.py")
    L.append("```")
    L.append("")
    L.append("### cold vs warm")
    L.append("")
    L.append("- `full` / `strip` are inherently **cold**: each call builds the "
             "reference from scratch. One row, `cold_or_warm=cold`, `wall_ms` = "
             "per-call wall.")
    L.append("- `warm_ref` / `warm_ref_strip` are **cached-ref**: two rows.")
    L.append("  - **cold** = first scored pair, including the one-time reference "
             "precompute.")
    L.append("  - **warm** = amortized per-call cost reusing the cached reference "
             "(zenbench loops the score call with the reference warmed once outside "
             "the loop). The cold/warm wall delta is the precompute one-time cost.")
    L.append("  - Peak heap: heaptrack measures the mode's full single-call "
             "execution once; that process peak is the **cold-path** peak and is "
             "listed on both rows (it is an upper bound on the warm steady-state "
             "working set).")
    L.append("- `dssim` strip / warm_ref_strip: **NOT_SUPPORTED** — dssim-core 3.4 "
             "has no strip walker (honest gap, not a stub).")
    L.append("")

    # Headline: 12 MP Full + 30 MP Full across metrics
    L.append("## Headline — Full mode across metrics")
    L.append("")
    L.append("> Note: the task brief mentions a \"16 MP\" headline, but the measured "
             "size grid uses 12 MP (4000x3000) and 30 MP (6000x5000) per the task's "
             "size list. No 16 MP cell was measured; nothing is extrapolated.")
    L.append("")
    for hsize, hlabel in (("12MP", "12 MP (4000x3000)"), ("30MP", "30 MP (6000x5000)")):
        L.append(f"### {hlabel} — `full` mode")
        L.append("")
        L.append("| metric | wall (ms) | peak heap |")
        L.append("|--------|----------:|----------:|")
        for metric in METRICS:
            wk = wall.get((hsize, metric, "full", "cold"), {})
            htk = ht.get((metric, "full", hsize), {})
            L.append(f"| {metric} | {fmt_ms(wk.get('mean_ms', 'NA'))} | "
                     f"{htk.get('peak_human', 'NA')} |")
        L.append("")

    # Per-metric breakdown, sub-table per size
    L.append("## Per-metric breakdown")
    L.append("")
    for metric in METRICS:
        L.append(f"### {metric}")
        L.append("")
        for size in SIZES:
            sl = SIZE_LABELS[size]
            L.append(f"**{sl}** ({SIZE_MP[size]:.2f} MP)")
            L.append("")
            L.append("| mode | cold/warm | wall (ms) | peak heap |")
            L.append("|------|-----------|----------:|----------:|")
            for mode in MODES[metric]:
                if mode in ("warm_ref", "warm_ref_strip"):
                    htk = ht.get((metric, mode, size), {})
                    for cw in ("cold", "warm"):
                        wk = wall.get((size, metric, mode, cw), {})
                        L.append(f"| {mode} | {cw} | {fmt_ms(wk.get('mean_ms','NA'))} | "
                                 f"{htk.get('peak_human','NA')} |")
                else:
                    wk = wall.get((size, metric, mode, "cold"), {})
                    htk = ht.get((metric, mode, size), {})
                    L.append(f"| {mode} | cold | {fmt_ms(wk.get('mean_ms','NA'))} | "
                             f"{htk.get('peak_human','NA')} |")
            if metric == "dssim":
                L.append("| strip | - | NOT_SUPPORTED | NOT_SUPPORTED |")
                L.append("| warm_ref_strip | - | NOT_SUPPORTED | NOT_SUPPORTED |")
            L.append("")
    L.append("---")
    L.append("")
    L.append("Raw data: `benchmarks/cpu_metrics_full_table_2026-05-28.tsv` "
             "(unified), `benchmarks/cpu_wall_2026-05-28.tsv` (wall), "
             "`benchmarks/heaptrack/refresh_2026-05-28/heaptrack_peaks.tsv` (heap) "
             "+ raw `.zst` traces in that dir.")
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
