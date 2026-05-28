#!/usr/bin/env python3
"""Parse every heaptrack file in benchmarks/heaptrack/ and emit a TSV with
the headline stats + the top 3 allocator function names (skipping
libcore / std::rt frames).

Output columns (tab-separated):
  metric  mode  size_label  total_runtime_s  n_alloc  n_temp_alloc
  peak_heap  peak_rss  leaked
  top1_calls  top1_peak  top1_caller
  top2_calls  top2_peak  top2_caller
  top3_calls  top3_peak  top3_caller

Usage:
  parse_heaptracks.py > stats.tsv
"""

import os
import re
import subprocess
import sys
from pathlib import Path

DIR = Path(__file__).resolve().parent

SKIP_FRAME = re.compile(
    r"alloc::raw_vec|library/(?:alloc|core|std)/|^[\s]*at /rustc/|heaptrack|"
    r"__rust_begin_short_backtrace|lang_start|"
    r"^[\s]*at .*?/rt\.rs|^[\s]*main$|^[\s]*main\s|"
    r"^[\s]*in /|"
    r"^[\s]*at /home/.*?/\.rustup/"
)


def pick_caller(bt_lines):
    """Return the first interesting frame from a backtrace block."""
    for line in bt_lines:
        if not line.strip():
            continue
        if SKIP_FRAME.search(line):
            continue
        # Function-name lines start with indent + identifier::namespace
        m = re.match(r"^\s+([a-z][a-zA-Z0-9_:<>{} ]+::[a-zA-Z0-9_:<>{} ]+)", line)
        if not m:
            continue
        sym = m.group(1).strip()
        # Drop trailing ::hHEX (Rust mangled hash)
        sym = re.sub(r"::h[0-9a-f]+$", "", sym)
        # Filter out obvious non-userland namespaces
        if sym.startswith(("std::", "core::", "alloc::", "_R")):
            continue
        return sym
    return ""


def parse_one(path: Path):
    """Run heaptrack_print on `path` and extract the stats row."""
    try:
        out = subprocess.run(
            ["heaptrack_print", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=60,
        ).stdout.decode("utf-8", errors="replace")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"# parse_one({path}) failed: {e}", file=sys.stderr)
        return None

    base = path.stem  # strips .zst
    # base = "<metric>_<mode_with_or_without_underscore>_<label>"
    label = base.rsplit("_", 1)[1]
    rest = base.rsplit("_", 1)[0]
    metric = rest.split("_", 1)[0]
    mode = rest.split("_", 1)[1] if "_" in rest else ""

    # Headline stats
    def grep(pat, line=False):
        m = re.search(pat, out, re.MULTILINE)
        return m.group(1) if m else "NA"

    runtime = grep(r"^total runtime:\s*([\d.]+)s")
    alloc_n = grep(r"^calls to allocation functions:\s*(\d+)")
    temp_n = grep(r"^temporary memory allocations:\s*(\d+)")
    peak_heap = grep(r"^peak heap memory consumption:\s*(\S+)")
    peak_rss = grep(r"^peak RSS \(including heaptrack overhead\):\s*(\S+)")
    leaked = grep(r"^total memory leaked:\s*(\S+)")

    # Backtrace groups under MOST CALLS TO ALLOCATION FUNCTIONS.
    # Each group header: "N calls with P peak consumption from:"
    # Followed by indented backtrace lines until next group or section end.
    sec_m = re.search(
        r"^MOST CALLS TO ALLOCATION FUNCTIONS\n(.*?)^MOST TEMPORARY",
        out,
        re.MULTILINE | re.DOTALL,
    )
    groups = []
    if sec_m:
        body = sec_m.group(1)
        # Split on group headers
        parts = re.split(r"^(\d+) calls with (\S+) peak consumption from:\n", body, flags=re.MULTILINE)
        # parts = [pre, calls1, peak1, body1, calls2, peak2, body2, ...]
        for i in range(1, len(parts), 3):
            calls = parts[i]
            peak = parts[i + 1]
            bt = parts[i + 2].splitlines()
            groups.append((int(calls), peak, bt))

    # Top-3 picks (deduped by caller frame)
    top = []
    seen = set()
    for calls, peak, bt in groups:
        frame = pick_caller(bt)
        if not frame or frame in seen:
            continue
        seen.add(frame)
        top.append((calls, peak, frame))
        if len(top) >= 3:
            break
    while len(top) < 3:
        top.append(("", "", ""))

    fields = [
        metric, mode, label,
        runtime, alloc_n, temp_n,
        peak_heap, peak_rss, leaked,
    ]
    for calls, peak, frame in top[:3]:
        fields.extend([str(calls), str(peak), frame])
    return "\t".join(fields)


def main():
    header = [
        "metric", "mode", "size_label",
        "total_runtime_s", "n_alloc", "n_temp_alloc",
        "peak_heap", "peak_rss", "leaked",
        "top1_calls", "top1_peak", "top1_caller",
        "top2_calls", "top2_peak", "top2_caller",
        "top3_calls", "top3_peak", "top3_caller",
    ]
    print("\t".join(header))
    files = sorted(DIR.glob("*.zst"))
    for f in files:
        row = parse_one(f)
        if row is not None:
            print(row)


if __name__ == "__main__":
    main()
