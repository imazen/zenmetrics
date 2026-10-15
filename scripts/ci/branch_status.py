#!/usr/bin/env python3
"""Every remote branch and open PR, classified so stale work is visible.

Usage: python3 scripts/ci/branch_status.py [--remote origin] [--base master] [--stale-days 7]

Fetches with `jj git fetch` (network goes through jj in this repo), then reads
refs with plain git against jj's git directory, so it works in any jj
workspace. Per remote branch:
commits behind/ahead of the base, last commit date and subject, the open PR
(number, mergeable state, failing checks), and a verdict:

  merged      no commits beyond the base: safe to delete
  archived    name starts with abandoned/ or superseded/: kept on purpose
  conflict    an open PR that does not merge cleanly
  stale       unmerged, no commit for --stale-days days
  behind      unmerged and more than 50 commits behind the base
  active      everything else

Exit status is 0; this reports, it never changes anything.
"""

import argparse
import datetime as dt
import json
import subprocess
import sys


GIT = ["git"]


def run(*args):
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def git(*args):
    return run(*GIT, *args)


def open_prs(repo):
    try:
        out = run(
            "gh", "pr", "list", "-R", repo, "--state", "open", "--limit", "200",
            "--json", "number,headRefName,mergeable,isDraft,statusCheckRollup,title",
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"branch_status: gh pr list failed ({e}); PR columns empty", file=sys.stderr)
        return {}
    prs = {}
    for pr in json.loads(out):
        failing = sum(
            1
            for c in pr.get("statusCheckRollup") or []
            if (c.get("conclusion") or c.get("state") or "").upper()
            in {"FAILURE", "ERROR", "TIMED_OUT", "CANCELLED"}
        )
        prs[pr["headRefName"]] = (pr["number"], pr["mergeable"], failing, pr["isDraft"])
    return prs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", default="origin")
    ap.add_argument("--base", default="master")
    ap.add_argument("--stale-days", type=int, default=7)
    a = ap.parse_args()

    run("jj", "git", "fetch", "--remote", a.remote)
    GIT.append(f"--git-dir={run('jj', 'git', 'root').strip()}")
    url = git("remote", "get-url", a.remote).strip()
    repo = url.split("github.com")[-1].lstrip(":/").removesuffix(".git")
    prs = open_prs(repo)
    base = f"{a.remote}/{a.base}"
    now = dt.datetime.now(dt.timezone.utc)
    rows = []
    for line in git("for-each-ref", "--format=%(refname:short)",
                    f"refs/remotes/{a.remote}/").split():
        name = line.removeprefix(f"{a.remote}/")
        if name in {"HEAD", a.base} or line == a.remote:
            continue
        behind, ahead = map(int, git("rev-list", "--left-right", "--count",
                                     f"{base}...{line}").split())
        when, subject = git("log", "-1", "--format=%cI%x09%s", line).strip().split("\t", 1)
        age = (now - dt.datetime.fromisoformat(when)).days
        pr = prs.get(name)
        if ahead == 0:
            verdict = "merged"
        elif name.startswith(("abandoned/", "superseded/")):
            verdict = "archived"
        elif pr and pr[1] == "CONFLICTING":
            verdict = "conflict"
        elif age > a.stale_days:
            verdict = "stale"
        elif behind > 50:
            verdict = "behind"
        else:
            verdict = "active"
        prcol = ""
        if pr:
            prcol = f"#{pr[0]} {pr[1].lower()}" + (f" {pr[2]} failing" if pr[2] else "") + (" draft" if pr[3] else "")
        rows.append((verdict, name, behind, ahead, age, prcol, subject[:60]))
    order = ["conflict", "stale", "behind", "merged", "active", "archived"]
    rows.sort(key=lambda r: (order.index(r[0]), r[1]))
    print(f"{'verdict':9} {'branch':52} {'behind':>6} {'ahead':>5} {'age':>4}  {'PR':22} last commit")
    for v, n, b, ah, age, pr, s in rows:
        print(f"{v:9} {n[:52]:52} {b:6} {ah:5} {age:3}d  {pr:22} {s}")
    orphans = sorted(set(prs) - {r[1] for r in rows})
    for o in orphans:
        print(f"{'pr-only':9} {o:52} (open PR #{prs[o][0]} from a fork or deleted branch)")


if __name__ == "__main__":
    main()
