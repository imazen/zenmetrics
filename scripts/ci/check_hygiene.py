#!/usr/bin/env python3
"""hygiene: address/identifier check over every tracked text file.

WHY THIS EXISTS. Global CLAUDE.md forbids household and home-network
identifiers in a public repo and prescribes grepping the diff before a push.
Nothing enforced it, so on 2026-09-06 an audit found the class live in this
repo and its sibling -- and the mechanism was not carelessness but COPYING:
one documented example in a shared doc spread a LAN URL through a dozen files
in two days. A class that spreads by being read has to be checked by a
machine, not by remembering.

    python3 scripts/ci/check_hygiene.py             # report + exit 1 on a hit
    python3 scripts/ci/check_hygiene.py --list      # report only, always exit 0
    python3 scripts/ci/check_hygiene.py --self-test # pin the patterns, exit 1 on failure

The patterns are NOT here: scripts/lib/hygiene_patterns.txt is their one owner,
shared with scripts/safe_push.sh so the pre-push gate and this scan can never
drift into checking different things. That gate scans the outgoing diff; this
scans the whole tracked tree, which is how an INHERITED hit gets found rather
than only a newly added one.

Site-specific patterns (names, labels) come from `hygiene_patterns` in the
private homefleet config -- HOMEFLEET_NODES, default
~/work/zen/homefleet/zenmetrics/fleet/nodes.toml -- because a pattern list in a
public repo that spelled out the values it protects would leak exactly what it
exists to keep out. CI has no such config, so CI enforces the generic classes;
that is the intended split, not a gap.

Counterpart in the sibling repo: zensim's scripts/lint_scripts.py grows the
same check as one of its own checks. Keep the pattern files in step.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATTERNS_FILE = ROOT / "scripts" / "lib" / "hygiene_patterns.txt"
HOMEFLEET_NODES_DEFAULT = Path.home() / "work" / "zen" / "homefleet" / "zenmetrics" / "fleet" / "nodes.toml"

# Enough bytes to catch a NUL in any real binary header without reading a
# multi-megabyte artifact in full. Matches zensim's linter.
BINARY_SNIFF_BYTES = 8192


def load_patterns() -> list[tuple[str, re.Pattern[str]]]:
    """The shared class patterns, plus the private site-specific ones when the
    homefleet config is readable. A missing config is not an error; an
    unreadable pattern file is, and the caller reports it rather than passing."""
    pats: list[tuple[str, re.Pattern[str]]] = []
    try:
        raw = PATTERNS_FILE.read_text(encoding="utf-8")
    except OSError:
        return []
    for line in raw.splitlines():
        if not line.strip() or line.lstrip().startswith("#") or "\t" not in line:
            continue
        name, rx = line.split("\t", 1)
        try:
            pats.append((name.strip(), re.compile(rx)))
        except re.error:
            continue

    cfg = Path(os.environ.get("HOMEFLEET_NODES") or HOMEFLEET_NODES_DEFAULT).expanduser()
    try:
        import tomllib

        with open(cfg, "rb") as f:
            for i, extra in enumerate(tomllib.load(f).get("hygiene_patterns") or []):
                if isinstance(extra, str) and extra:
                    try:
                        pats.append((f"private-{i}", re.compile(extra)))
                    except re.error:
                        continue
    except (OSError, ImportError, ValueError):
        pass
    return pats


def tracked_files() -> list[str]:
    """Every tracked file, however this checkout reaches its VCS.

    CI (`actions/checkout`) and the colocated primary both answer `git
    ls-files`. A SECONDARY jj workspace has no `.git` at all, so that exits
    nonzero there and `jj file list` answers instead -- every workspace of the
    same repo returns the same list, since they share the operation log."""
    for cmd, sep in (
        (["git", "-c", "core.quotepath=off", "ls-files", "-z"], "\0"),
        (["jj", "file", "list"], "\n"),
    ):
        try:
            out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.SubprocessError):
            continue
        if out.returncode == 0:
            return [f for f in out.stdout.split(sep) if f]
    return []


def scan(pats: list[tuple[str, re.Pattern[str]]], files: list[str]) -> list[str]:
    """Hits as "path:line: [class] text".

    Skips `.orig`/`.rej` (patch tooling writes what it likes into those),
    binaries by a NUL sniff of the first bytes, and the pattern file itself --
    which necessarily contains regexes describing the classes it matches, and a
    guard that refuses its own definition is a guard nobody can edit."""
    rel_patterns = str(PATTERNS_FILE.relative_to(ROOT))
    hits: list[str] = []
    for rel in files:
        if rel.endswith((".orig", ".rej")) or rel == rel_patterns:
            continue
        fp = ROOT / rel
        try:
            with open(fp, "rb") as fh:
                head = fh.read(BINARY_SNIFF_BYTES)
                if b"\0" in head:
                    continue
                raw = head + fh.read()
        except OSError:
            continue
        for lineno, line in enumerate(raw.decode("utf-8", errors="replace").splitlines(), 1):
            for name, rx in pats:
                if rx.search(line):
                    hits.append(f"{rel}:{lineno}: [{name}] {line.strip()[:120]}")
                    break
    return hits


# ---------------------------------------------------------------- self-test


def self_test() -> int:
    """Pin the patterns, positive AND negative. Every flagged literal below is
    ASSEMBLED at runtime from parts, never written out, so this file cannot
    itself carry the shapes it exists to ban."""
    pats = load_patterns()
    names = {n for n, _ in pats}
    fails = 0

    def check(label: str, ok: bool, detail: str = "") -> None:
        nonlocal fails
        print(f"  {label:<58} {'PASS' if ok else 'FAIL'}{'' if ok else '  ' + detail}")
        if not ok:
            fails += 1

    def classes(line: str) -> list[str]:
        return [n for n, rx in pats if rx.search(line)]

    check("pattern file loads all three public classes",
          {"private-network-address", "hardware-address", "household-framing"} <= names, str(names))

    positives = [
        ("private-network-address", "see http://%s.%s.%s.%s:3300/report" % ("192", "168", "50", "44")),
        ("private-network-address", "see http://%s.%s.%s.%s:3300/report" % ("10", "1", "2", "3")),
        ("private-network-address", "see http://%s.%s.%s.%s:3300/report" % ("172", "20", "0", "5")),
        ("hardware-address", '  "hw": "%s",' % ":".join(["04", "7c", "16", "b3", "18", "51"])),
        ("household-framing", "the %s PCs stay on Windows" % ("kid" + "s'")),
        ("household-framing", "borrowed a %s box overnight" % ("child" + "'s")),
    ]
    for cls, line in positives:
        check(f"positive [{cls}]", cls in classes(line), line)

    # Negative controls. Without these, a gate that refused EVERYTHING would
    # pass every positive above.
    negatives = [
        "browse http://localhost:3300/zensim/reports/",
        "curl http://127.0.0.1:3300/health",
        "python3 -m http.server 3400 --bind 0.0.0.0",
        "worker on r7900x, secondary on tower",
        "zenjxl 0.4.0, build 10.0.19045 on the runner",
        "elapsed 01:02:03:04:05 in the log",
        "per_img_front_kids = {}   # img -> set(kid) on front",
        "topk_kids = sorted(keep)",
    ]
    for line in negatives:
        check("negative control", classes(line) == [], f"{line!r} -> {classes(line)}")

    # The public pattern file must describe CLASSES, never the values it
    # protects -- checked by feeding the file to its own patterns.
    self_hits = [(i, ln) for i, ln in enumerate(PATTERNS_FILE.read_text(encoding="utf-8").splitlines(), 1)
                 if classes(ln)]
    check("public patterns carry no specific values", not self_hits, str(self_hits[:2]))

    print(f"\nself-test check_hygiene: {'PASS' if fails == 0 else 'FAIL'} ({fails} failure(s))")
    return 0 if fails == 0 else 1


def main() -> int:
    if "--self-test" in sys.argv:
        return self_test()

    pats = load_patterns()
    if not pats:
        print(f"check_hygiene: cannot read {PATTERNS_FILE} — refusing to report a pass "
              "on an unrun check.", file=sys.stderr)
        return 1

    files = tracked_files()
    if not files:
        print("check_hygiene: no VCS answered for the tracked-file list — refusing to "
              "report a pass on an unrun check.", file=sys.stderr)
        return 1

    hits = scan(pats, files)
    if not hits:
        print(f"check_hygiene: {len(files)} tracked files checked against {len(pats)} "
              "pattern(s), no address/identifier hits")
        return 0

    print("\nhygiene: address/identifier check\n")
    for h in hits:
        print(f"    {h}")
    print(
        f"\ncheck_hygiene: {len(hits)} hit(s). There is no bypass. The two resolutions are:\n"
        "  1. use the documented canonical form — http://localhost:<port> for a served\n"
        "     page, a neutral node id for a box;\n"
        "  2. move the value into the private homefleet repo and read it from there at\n"
        f"     runtime.\nPatterns: {PATTERNS_FILE.relative_to(ROOT)}"
    )
    return 0 if "--list" in sys.argv else 1


if __name__ == "__main__":
    sys.exit(main())
