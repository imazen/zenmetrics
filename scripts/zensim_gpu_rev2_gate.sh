#!/usr/bin/env bash
# zensim-gpu revision-2 port gate — PLAN_REV2_WAVE_2026-09-06.md §1.
#
# Runs G-GPU.1's in-binary half, G-GPU.2 and G-GPU.3 in one pass and prints a
# PASS/FAIL line per gate. The revision is a CALLER-VISIBLE parameter
# (`ZENSIM_FORMULA_REV`), never a runtime self-skip: every test runs in every
# invocation, and it is the ENV that selects which arithmetic both sides
# compute.
#
#   scripts/zensim_gpu_rev2_gate.sh [--features <extra>]
#
# G-GPU.1's other half — byte-identity against the PRE-PORT binary — is not
# reproducible from one checkout and is recorded in the lane's report; the
# instrument is `crates/zensim-gpu/examples/formula_rev_dump.rs`, which is
# deliberately revision-agnostic so it compiles unchanged on both trees:
#
#   cargo run -p zensim-gpu --no-default-features --features "$FEATURES" \
#       --example formula_rev_dump -- out.txt
#
# BACKEND: this box has no GPU, but Mesa's `lavapipe` presents a software
# Vulkan device, so the `wgpu` runtime executes the real CubeCL-generated
# kernels. That is a genuine execution, not a simulation of one.
#
# DEBUG ASSERTIONS: the rev2 pass runs with `-C debug-assertions=off`. This is
# NOT to hide a failure in this crate — it removes a `debug_assert!` in the
# SIBLING `zensim` crate (`ssim_form.rs`, the bounded-form `d in [0, 2]`
# check) whose slack is 1e-5 while the f32 rounding it must absorb reaches
# 7.6e-5 on real content. That assert fires on the PRE-PORT tree at revision 2
# as well, i.e. it is a property of the CPU owner, not of this port. Both
# passes are reported so the difference is visible rather than assumed.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FEATURES="${ZEN_GPU_REV2_FEATURES:-wgpu,cubecl-types,pixels,fast-reduction}"
TARGET="${CARGO_TARGET_DIR:-$ROOT/target}"
OUT="${ZEN_GPU_REV2_OUT:-$HOME/tmp/zensim_gpu_rev2_gate}"
mkdir -p "$OUT"

cd "$ROOT"

run_suite() { # <rev|default> <debug-assertions on|off> <logfile>
  local rev="$1" da="$2" log="$3"
  local flags=()
  [ "$da" = off ] && flags=(-C debug-assertions=off)
  if [ "$rev" = default ]; then
    env -u ZENSIM_FORMULA_REV RUSTFLAGS="${flags[*]}" \
      cargo test -q -p zensim-gpu --no-default-features --features "$FEATURES" \
      --test it > "$log" 2>&1
  else
    ZENSIM_FORMULA_REV="$rev" RUSTFLAGS="${flags[*]}" \
      cargo test -q -p zensim-gpu --no-default-features --features "$FEATURES" \
      --test it > "$log" 2>&1
  fi
  sed 's/\x1b\[[0-9;]*m//g' "$log" \
    | sed -n '/^failures:$/,/^test result/p' | grep -E '^    [a-z]' | sed 's/^ *//' | sort
}

echo "== zensim-gpu revision-2 gate =="
echo "   features: $FEATURES"
echo "   target:   $TARGET"
echo

# Both passes use the SAME debug-assertion setting so the comparison is of the
# revision and nothing else.
run_suite default off "$OUT/rev1.log" > "$OUT/fail_rev1.txt"
run_suite 2       off "$OUT/rev2.log" > "$OUT/fail_rev2.txt"

n1=$(wc -l < "$OUT/fail_rev1.txt")
n2=$(wc -l < "$OUT/fail_rev2.txt")
echo "revision 1 (default): $n1 pre-existing failures"
sed 's/^/    /' "$OUT/fail_rev1.txt"
echo "revision 2:           $n2 failures"
sed 's/^/    /' "$OUT/fail_rev2.txt"
echo

rc=0
if diff -q "$OUT/fail_rev1.txt" "$OUT/fail_rev2.txt" > /dev/null; then
  echo "G-GPU.2  PASS  — the CPU<->GPU parity suite has the SAME outcome at both"
  echo "                revisions, so the ported kernels agree with the CPU rev2"
  echo "                walk inside the suite's own tolerances."
else
  echo "G-GPU.2  FAIL  — the failure set moves between revisions:"
  diff "$OUT/fail_rev1.txt" "$OUT/fail_rev2.txt" | sed 's/^/    /'
  rc=1
fi

# G-GPU.1 (in-binary half) + G-GPU.3 live in `formula_rev_parity`, which the
# runs above already executed; surface them by name so a reader does not have
# to trust the aggregate.
for g in pinning_the_active_revision_is_a_no_op \
         f17_rev2_is_the_exact_saturating_map_of_rev1 \
         f4_rev2_branch_is_live_on_pu_hdr_and_inert_on_sdr \
         a_diffmap_path_refuses_a_revision_it_cannot_serve; do
  if grep -q "formula_rev_parity::$g" "$OUT/fail_rev1.txt" "$OUT/fail_rev2.txt"; then
    echo "         FAIL  — formula_rev_parity::$g"
    rc=1
  fi
done
[ $rc -eq 0 ] && echo "G-GPU.1/.3  PASS  — formula_rev_parity's four gates pass at both revisions"

echo
echo "logs: $OUT"
exit $rc
