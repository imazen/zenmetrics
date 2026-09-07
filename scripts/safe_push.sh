#!/usr/bin/env bash
# safe_push.sh — the ONLY sanctioned way to advance a tracked bookmark in this repo.
#
# WHY THIS EXISTS
# ---------------
# `jj bookmark set master -r @ && jj git push --bookmark master` is a NON-fast-forward
# push whenever `@` does not descend from `<bookmark>@origin`. jj performs it without a
# prompt, and the push succeeds: the bookmark simply MOVES SIDEWAYS and every commit
# that was only reachable from the old tip is silently unreachable from the new one.
# Nothing errors. Nothing warns. The commits still exist as local objects, so
# `jj log` on the pushing lane looks fine — the loss is only visible to whoever
# later asks "is my commit an ancestor of origin/<bookmark>?".
#
# MEASURED in the SIBLING repo, 2026-09-04: imazen/zensim's origin/main moved sideways
# TWICE in one afternoon (16:58:53
# and 17:08:29 UTC-6), and the second move dropped NINE commits from six lanes,
# including `d3a948ca` (the G-ADDR board-coverage feature, +555/-23 across six
# files). The boards on /mnt/v had been generated WITH that code, so the next
# regen would have silently un-drawn 46 NOT-SHIPPABLE badges. Full incident
# record: zensim's benchmarks/push_clobber_2026-09-05.md. Nothing about the
# mechanism is repo-specific — jj moves any bookmark sideways without a prompt —
# so this repo runs the same wrapper. It is a MIRROR of zensim's
# scripts/safe_push.sh, differing only in the default bookmark (master) and in
# what its second gate reads; keep the two in step.
#
# WHAT THIS DOES
# --------------
#   fetch -> assert <bookmark>@origin is an ANCESTOR of the target
#         -> hygiene: address/identifier check on the outgoing diff
#         -> set -> push -> verify
#
# The second gate scans the lines this push would ADD for the identifier classes
# that must not enter a public repo (see scripts/lib/hygiene_patterns.txt for the
# patterns and the reasoning). It exits 7 -- a DIFFERENT code from the sideways
# refusal below, so a caller can tell the two apart -- and, like the first gate,
# it has no bypass flag.
#
# The assertion is not a warning. If any commit is reachable from <bookmark>@origin
# but NOT from the target, this script prints every one of them and exits 3 WITHOUT
# touching the bookmark. There is no --force. Recover with:
#     jj git fetch && jj rebase -d <bookmark>@origin      # then re-run safe_push
#
# USAGE
#   scripts/safe_push.sh                       # push @ (or @- if @ is empty) to master
#   scripts/safe_push.sh -r <rev>              # push an explicit revision
#   scripts/safe_push.sh -b <bookmark>         # a bookmark other than master
#   scripts/safe_push.sh --dry-run             # run every check, push nothing
#   scripts/safe_push.sh --self-test           # build a throwaway repo and prove
#                                              #   a sideways target is REFUSED and
#                                              #   a flagged identifier is REFUSED
#
# EXIT CODES
#   0 pushed (or --dry-run/--self-test passed)   4 fetch failed
#   2 bad usage / unresolvable revision          5 bookmark set or push failed
#   3 REFUSED: not a fast-forward                6 push reported success but did not land
#                                                7 REFUSED: hygiene check on the outgoing diff
set -uo pipefail

BOOKMARK=master
REV=""
DRY_RUN=0
SELF_TEST=0

while [ $# -gt 0 ]; do
  case "$1" in
    -b|--bookmark) BOOKMARK="$2"; shift 2 ;;
    -r|--rev)      REV="$2";      shift 2 ;;
    -n|--dry-run)  DRY_RUN=1;     shift ;;
    --self-test)   SELF_TEST=1;   shift ;;
    -h|--help)     sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "safe_push: unknown argument '$1' (see --help)" >&2; exit 2 ;;
  esac
done


# ---------------------------------------------------------------- hygiene

# Patterns live in ONE file, shared with scripts/ci/check_hygiene.py, so the pre-push
# gate and the CI tracked-file scan can never drift into checking different things.
HYGIENE_PATTERNS_FILE="$(cd "$(dirname "$0")" && pwd)/lib/hygiene_patterns.txt"

# Site-specific additions (names, labels) are NOT in this repo: they come from the
# private homefleet config, whose `hygiene_patterns` array this reads if it is
# present. Absent config = the generic classes only, announced, never silent.
HOMEFLEET_NODES_DEFAULT="$HOME/work/zen/homefleet/zenmetrics/fleet/nodes.toml"

hygiene_extra_patterns() {
  local cfg="${HOMEFLEET_NODES:-$HOMEFLEET_NODES_DEFAULT}"
  [ -r "$cfg" ] || return 0
  command -v python3 >/dev/null 2>&1 || return 0
  python3 - "$cfg" <<'PYEOF' 2>/dev/null || true
import sys, tomllib
try:
    with open(sys.argv[1], "rb") as f:
        cfg = tomllib.load(f)
except Exception:
    raise SystemExit(0)
for i, pat in enumerate(cfg.get("hygiene_patterns") or []):
    if isinstance(pat, str) and pat:
        print(f"private-{i}\t{pat}")
PYEOF
}

# Added lines of the outgoing diff, as "<path>:<text>". `jj diff --git` is used
# rather than a bare diff because only the git format carries the `+++ b/<path>`
# header this needs to attribute a hit to a file. The pattern file itself is
# skipped: it necessarily contains regexes that describe the very classes it
# matches, and a guard that refuses its own definition is a guard nobody can edit.
hygiene_added_lines() {
  local from="$1" to="$2"
  jj diff --ignore-working-copy --git --from "$from" --to "$to" 2>/dev/null |
    awk '
      /^\+\+\+ b\// { path = substr($0, 7); next }
      /^\+\+\+ \/dev\/null/ { path = ""; next }
      /^\+/ && path != "" && path != "scripts/lib/hygiene_patterns.txt" {
        print path ":" substr($0, 2)
      }'
}

# Returns 0 clean, 7 refused. Prints every hit; never truncates to one.
hygiene_check() {
  local from="$1" to="$2"

  if [ ! -r "$HYGIENE_PATTERNS_FILE" ]; then
    echo "safe_push: hygiene: address/identifier check CANNOT RUN — no $HYGIENE_PATTERNS_FILE" >&2
    echo "  Refusing rather than pushing unchecked." >&2
    return 7
  fi

  local added; added=$(hygiene_added_lines "$from" "$to")
  if [ -z "$added" ]; then
    echo "safe_push: hygiene: address/identifier check — no added lines to scan."
    return 0
  fi

  local hits="" name rx n_pat=0
  while IFS=$'\t' read -r name rx; do
    case "$name" in ''|\#*) continue ;; esac
    [ -z "$rx" ] && continue
    n_pat=$((n_pat + 1))
    local found
    found=$(printf '%s\n' "$added" | grep -nE -- "$rx" || true)
    if [ -n "$found" ]; then
      hits="${hits}$(printf '%s\n' "$found" | sed "s/^/    [${name}] /")
"
    fi
  done <<HYGEOF
$(cat "$HYGIENE_PATTERNS_FILE"; hygiene_extra_patterns)
HYGEOF

  if [ -n "$hits" ]; then
    echo "" >&2
    echo "safe_push: REFUSED — hygiene: address/identifier check." >&2
    echo "" >&2
    echo "  These lines would be ADDED to a public repo by this push:" >&2
    echo "" >&2
    printf '%s' "$hits" | cut -c1-200 >&2
    echo "" >&2
    echo "  There is no bypass. The two resolutions are:" >&2
    echo "    1. use the documented canonical form — http://localhost:<port> for a" >&2
    echo "       served page, a neutral node id for a box;" >&2
    echo "    2. move the value into the private homefleet repo and have the public" >&2
    echo "       side read it from there at runtime." >&2
    echo "" >&2
    echo "  Patterns: $HYGIENE_PATTERNS_FILE" >&2
    echo "" >&2
    return 7
  fi

  echo "safe_push: hygiene: address/identifier check OK ($n_pat pattern(s), $(printf '%s\n' "$added" | grep -c .) added line(s))."
  return 0
}

# ---------------------------------------------------------------- core

jjq() { jj log --no-graph --ignore-working-copy -r "$1" -T "${2:-commit_id}"; }

do_push() {
  local bookmark="$1" rev="$2" dry="$3"

  echo "safe_push: fetching..."
  jj git fetch || { echo "safe_push: FETCH FAILED — refusing to push blind." >&2; return 4; }

  # Resolve the target. Default: @, or @- when @ is the usual empty working commit.
  if [ -z "$rev" ]; then
    if [ -z "$(jj log --no-graph --ignore-working-copy -r '@ & ~empty()' -T 'commit_id')" ]; then
      rev='@-'
      echo "safe_push: @ is empty; targeting @- instead."
    else
      rev='@'
    fi
  fi

  local target
  target=$(jjq "$rev") || { echo "safe_push: cannot resolve revision '$rev'." >&2; return 2; }
  if [ -z "$target" ]; then
    echo "safe_push: revision '$rev' resolved to nothing." >&2; return 2
  fi

  local remote_rev="${bookmark}@origin"
  local remote_tip
  remote_tip=$(jjq "$remote_rev" 2>/dev/null || true)

  if [ -z "$remote_tip" ]; then
    echo "safe_push: ${remote_rev} does not exist yet — treating this as a NEW bookmark."
  else
    # THE GATE. `::A ~ ::B` is the set of commits reachable from the remote tip
    # but NOT from the target — exactly what this push would drop. It is empty
    # if and only if the remote tip is an ancestor of the target, so the same
    # expression is both the test and the diagnostic. (`A ~ ::B` would be a
    # correct test too, but names only the tip and hides the rest of the loss.)
    local dropped
    dropped=$(jj log --no-graph --ignore-working-copy \
                -r "::${remote_rev} ~ ::${target}" \
                -T 'commit_id.short() ++ " " ++ description.first_line() ++ "\n"' 2>/dev/null || true)
    if [ -n "$dropped" ]; then
      echo "" >&2
      echo "safe_push: REFUSED — this push is NOT a fast-forward." >&2
      echo "  bookmark : ${bookmark}" >&2
      echo "  target   : ${target}" >&2
      echo "  ${remote_rev} : ${remote_tip}" >&2
      echo "" >&2
      echo "  ${remote_rev} is NOT an ancestor of the target. Pushing would make these" >&2
      echo "  commits unreachable from ${bookmark} (they would NOT be deleted, but nothing" >&2
      echo "  would point at them and the next lane to regenerate from ${bookmark} would" >&2
      echo "  silently lose their content):" >&2
      echo "" >&2
      printf '%s\n' "$dropped" | head -40 | sed 's/^/      /' >&2
      local n_dropped; n_dropped=$(printf '%s\n' "$dropped" | grep -c .)
      [ "$n_dropped" -gt 40 ] && echo "      ... and $((n_dropped - 40)) more ($n_dropped total)" >&2
      echo "" >&2
      echo "  Fix it, do not force it:" >&2
      echo "      jj git fetch && jj rebase -d ${remote_rev}" >&2
      echo "      # resolve any conflicts in @, keeping BOTH lanes' hunks, then re-run:" >&2
      echo "      scripts/safe_push.sh -b ${bookmark}" >&2
      echo "" >&2
      return 3
    fi
    echo "safe_push: OK — ${remote_rev} (${remote_tip:0:12}) is an ancestor of ${target:0:12}."
  fi

  # SECOND GATE. Scan what this push would ADD. For a brand-new bookmark there is
  # no remote tip to diff against, so the whole target tree is the addition.
  local hygiene_from="${remote_tip:-$(jjq 'root()')}"
  hygiene_check "$hygiene_from" "$target" || return 7

  if [ "$dry" = 1 ]; then
    echo "safe_push: --dry-run, stopping before 'bookmark set'."
    return 0
  fi

  jj bookmark set "$bookmark" -r "$target" || return 5
  jj git push --bookmark "$bookmark" || {
    echo "safe_push: PUSH FAILED. The bookmark was moved locally but not published;" >&2
    echo "  re-run after 'jj git fetch && jj rebase -d ${remote_rev}'." >&2
    return 5
  }

  # VERIFY. A push that reports success but does not land is the failure mode the
  # 2026-05-29 orphaned-bookmark incident taught us to check for explicitly.
  jj git fetch >/dev/null 2>&1
  remote_tip=$(jjq "$remote_rev" 2>/dev/null || true)
  local unlanded
  unlanded=$(jj log --no-graph --ignore-working-copy -r "${target} ~ ::${remote_rev}" -T 'commit_id' 2>/dev/null || true)
  if [ -n "$unlanded" ]; then
    echo "safe_push: VERIFY FAILED — ${target:0:12} is NOT an ancestor of ${remote_rev} after the push." >&2
    return 6
  fi
  echo "safe_push: VERIFIED — ${target:0:12} is on ${remote_rev} (now ${remote_tip:0:12})."
  return 0
}

# ---------------------------------------------------------------- self-test

self_test() {
  local rc fails=0
  # SP_TMP is intentionally GLOBAL: the EXIT trap fires after self_test's locals
  # have gone out of scope, so a `local tmp` would be unbound there under `set -u`.
  SP_TMP=$(mktemp -d "${TMPDIR:-$HOME/tmp}/safe_push_selftest.XXXXXX") || return 1
  trap 'rm -rf "$SP_TMP"' EXIT
  local tmp="$SP_TMP"

  local script; script=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")

  git init --bare -q "$tmp/remote.git" || return 1
  jj git clone --quiet "$tmp/remote.git" "$tmp/work" >/dev/null 2>&1 || return 1
  cd "$tmp/work" || return 1
  jj config set --repo user.name "safe-push selftest" >/dev/null 2>&1
  jj config set --repo user.email "selftest@example.invalid" >/dev/null 2>&1

  # base commit, published
  echo base > f.txt
  jj describe -m base >/dev/null 2>&1
  jj bookmark create ${BOOKMARK} -r @ >/dev/null 2>&1 || jj bookmark set ${BOOKMARK} -r @ >/dev/null 2>&1
  jj git push --bookmark ${BOOKMARK} >/dev/null 2>&1 || return 1
  local base; base=$(jj log --no-graph --ignore-working-copy -r "${BOOKMARK}@origin" -T 'commit_id')

  # --- CASE 1: a fast-forward target is ACCEPTED -------------------------
  jj new "${BOOKMARK}@origin" -m "ff-child" >/dev/null 2>&1
  echo ff > g.txt
  # Capture the target BEFORE the push: a successful `jj git push` makes the
  # working-copy commit immutable and creates a fresh empty `@` on top of it,
  # so reading `@` afterwards would read the wrong commit.
  jj status >/dev/null 2>&1   # force a snapshot so g.txt is in the commit
  local ff_tip; ff_tip=$(jj log --no-graph --ignore-working-copy -r '@' -T 'commit_id')
  "$script" -r "$ff_tip" >"$tmp/case1.log" 2>&1; rc=$?
  if [ "$rc" -eq 0 ] && [ -z "$(jj log --no-graph --ignore-working-copy -r "${ff_tip} ~ ::${BOOKMARK}@origin" -T 'commit_id')" ]; then
    echo "  CASE 1 fast-forward ACCEPTED and landed .......... PASS"
  else
    echo "  CASE 1 fast-forward ACCEPTED and landed .......... FAIL (rc=$rc)"; sed 's/^/      /' "$tmp/case1.log"; fails=$((fails+1))
  fi

  # --- CASE 2: a SIDEWAYS target is REFUSED ------------------------------
  # A second lane's commit, built on the OLD base, ignorant of ff-child.
  jj new "$base" -m "sideways-lane" >/dev/null 2>&1
  echo sideways > h.txt
  jj status >/dev/null 2>&1
  local side_tip; side_tip=$(jj log --no-graph --ignore-working-copy -r '@' -T 'commit_id')
  local before; before=$(jj log --no-graph --ignore-working-copy -r "${BOOKMARK}@origin" -T 'commit_id')
  "$script" -r "$side_tip" >"$tmp/case2.log" 2>&1; rc=$?
  local after; after=$(jj log --no-graph --ignore-working-copy -r "${BOOKMARK}@origin" -T 'commit_id')
  if [ "$rc" -eq 3 ] && [ "$before" = "$after" ]; then
    echo "  CASE 2 sideways REFUSED (rc=3), remote UNMOVED ... PASS"
  else
    echo "  CASE 2 sideways REFUSED (rc=3), remote UNMOVED ... FAIL (rc=$rc, before=${before:0:12} after=${after:0:12})"
    sed 's/^/      /' "$tmp/case2.log"; fails=$((fails+1))
  fi

  # --- CASE 3: the refusal NAMES the commit that would be dropped --------
  if grep -q "ff-child" "$tmp/case2.log"; then
    echo "  CASE 3 refusal names the dropped commit .......... PASS"
  else
    echo "  CASE 3 refusal names the dropped commit .......... FAIL"; fails=$((fails+1))
  fi

  # --- CASE 4: after rebasing onto the remote tip, it is ACCEPTED --------
  jj rebase -r "$side_tip" -d "${BOOKMARK}@origin" >/dev/null 2>&1
  local fixed; fixed=$(jj log --no-graph --ignore-working-copy -r 'description(glob:"sideways-lane*")' -T 'commit_id')
  "$script" -r "$fixed" >"$tmp/case4.log" 2>&1; rc=$?
  # Assert by DESCRIPTION, not by the pre-rebase commit id: `jj rebase` rewrites
  # the commit, and jj resolves an obsolete id to its successor, so a stale id
  # would compare the wrong object.
  local landed_side landed_ff
  landed_side=$(jj log --no-graph --ignore-working-copy -r "description(glob:\"sideways-lane*\") & ::${BOOKMARK}@origin" -T 'commit_id')
  landed_ff=$(jj log --no-graph --ignore-working-copy -r "description(glob:\"ff-child*\") & ::${BOOKMARK}@origin" -T 'commit_id')
  local miss_fixed="" miss_ff=""
  [ -z "$landed_side" ] && miss_fixed="sideways-lane"
  [ -z "$landed_ff" ]   && miss_ff="ff-child"
  if [ "$rc" -eq 0 ] && [ -n "$landed_side" ] && [ -n "$landed_ff" ]; then
    echo "  CASE 4 rebase-then-push lands BOTH lanes ......... PASS"
  else
    echo "  CASE 4 rebase-then-push lands BOTH lanes ......... FAIL (rc=$rc not_landed='${miss_fixed} ${miss_ff}')"
    sed 's/^/      /' "$tmp/case4.log"; fails=$((fails+1))
  fi

  # --- CASE 5: a flagged identifier is REFUSED (rc=7), remote UNMOVED ----
  # Built from the pattern file's own private-network-address class, so the case
  # cannot drift from the rule it is testing. The literal is assembled at runtime
  # rather than written out, so this script never itself carries the shape it bans.
  jj new "${BOOKMARK}@origin" -m "hygiene-probe" >/dev/null 2>&1
  printf 'probe http://%s.%s.%s.%s:3300/x\n' 192 168 50 44 > probe.md
  jj status >/dev/null 2>&1
  local hyg_tip; hyg_tip=$(jj log --no-graph --ignore-working-copy -r '@' -T 'commit_id')
  before=$(jj log --no-graph --ignore-working-copy -r "${BOOKMARK}@origin" -T 'commit_id')
  "$script" -r "$hyg_tip" >"$tmp/case5.log" 2>&1; rc=$?
  after=$(jj log --no-graph --ignore-working-copy -r "${BOOKMARK}@origin" -T 'commit_id')
  if [ "$rc" -eq 7 ] && [ "$before" = "$after" ] && grep -q 'probe.md' "$tmp/case5.log"; then
    echo "  CASE 5 flagged line REFUSED (rc=7), remote UNMOVED . PASS"
  else
    echo "  CASE 5 flagged line REFUSED (rc=7), remote UNMOVED . FAIL (rc=$rc, before=${before:0:12} after=${after:0:12})"
    sed 's/^/      /' "$tmp/case5.log"; fails=$((fails+1))
  fi

  # --- CASE 6: the NEGATIVE CONTROL — same commit, canonical form, ACCEPTED --
  # Without this, CASE 5 would also pass if the hygiene gate refused everything.
  echo 'probe http://localhost:3300/x' > probe.md
  jj status >/dev/null 2>&1
  local clean_tip; clean_tip=$(jj log --no-graph --ignore-working-copy -r '@' -T 'commit_id')
  "$script" -r "$clean_tip" >"$tmp/case6.log" 2>&1; rc=$?
  if [ "$rc" -eq 0 ] && [ -z "$(jj log --no-graph --ignore-working-copy -r "${clean_tip} ~ ::${BOOKMARK}@origin" -T 'commit_id')" ]; then
    echo "  CASE 6 canonical form ACCEPTED and landed ....... PASS"
  else
    echo "  CASE 6 canonical form ACCEPTED and landed ....... FAIL (rc=$rc)"
    sed 's/^/      /' "$tmp/case6.log"; fails=$((fails+1))
  fi

  echo ""
  if [ "$fails" -eq 0 ]; then
    echo "self-test safe_push: PASS (0 failure(s))"; return 0
  fi
  echo "self-test safe_push: FAIL ($fails failure(s))"; return 1
}

if [ "$SELF_TEST" = 1 ]; then
  self_test; exit $?
fi

do_push "$BOOKMARK" "$REV" "$DRY_RUN"; exit $?
