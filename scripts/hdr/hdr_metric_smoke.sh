#!/usr/bin/env bash
# HDR-metric smoke + sanity test for `zenmetrics score --hdr`.
#
# Exercises the three HDR decode paths (EXR, gain-map HEIC, Google-format
# UltraHDR JPEG) through every CPU metric (ssim2/dssim/butteraugli/zensim),
# asserting:
#   - identity scores hit the per-metric best (ssim2/zensim=100, dssim/butter=0)
#   - a low-quality re-encode scores clearly worse (discrimination)
#   - different EXR content scores far from identity
#   - the HEIC decode path reproduces the EXR pixel-for-pixel (butteraugli=0)
#
# Local-corpus harness, not CI: needs the HDR corpus on /mnt/v. Build first
# (`hdr` needs a metric backend so the umbrella has variants to dispatch — the
# bare `--features hdr` build does not compile on its own):
#   cargo build --release -p zenmetrics-cli \
#     --features "png,jpeg,cpu-metrics,gpu-cvvdp,gpu-butteraugli,gpu-cuda,hdr"
#   cargo build --release -p zenhdr-corpus --example make_distorted
#
# Override paths via env (BIN, MK_DISTORTED, UHDR_JPEG, HEIC_SM, HEIC_X,
# EXR_X, EXR_A, EXR_B). Exits nonzero on the first failed assertion.
set -uo pipefail

BIN=${BIN:-./target/release/zenmetrics}
MK_DISTORTED=${MK_DISTORTED:-./target/release/examples/make_distorted}
UHDR_JPEG=${UHDR_JPEG:-/home/lilith/work/zen/ultrahdr/test_ultrahdr.jpg}
HEIC_SM=${HEIC_SM:-/mnt/v/heic/FE0C1B41-813F-425A-BDA8-EDA6F5860FB3.HEIC}
HEIC_X=${HEIC_X:-/mnt/v/heic/IMG_1420.HEIC}
EXR_X=${EXR_X:-/mnt/v/hdr-corpus/refs/IMG_1420.exr}
EXR_SMALL=${EXR_SMALL:-/mnt/v/hdr-corpus/refs/IMG_1509.exr}
EXR_A=${EXR_A:-/mnt/v/hdr-corpus/refs/32F76D88-01ED-42BA-95A2-716FEBABF1E2.exr}
EXR_B=${EXR_B:-/mnt/v/hdr-corpus/refs/94067DD9-550D-4573-A8EA-A462D51888D7.exr}
DIST_JPEG=${DIST_JPEG:-/mnt/v/hdr-corpus/distorted_test_ultrahdr_q15.jpg}

fails=0
# val <metric> <ref> <dist>  -> echoes the last numeric token of the score line
val() {
  local out; out=$("$BIN" score --metric "$1" --reference "$2" --distorted "$3" --hdr 2>&1)
  if [[ $? -ne 0 ]]; then echo "ERR:$out"; return; fi
  # last `name=number` token
  echo "$out" | grep -oE '[a-z0-9_]+=[-0-9.]+' | tail -1 | cut -d= -f2
}
# assert <desc> <actual> <op> <bound>
assert() {
  local desc=$1 act=$2 op=$3 bound=$4
  if [[ "$act" == ERR:* ]]; then printf '  FAIL  %-38s %s\n' "$desc" "$act"; fails=$((fails+1)); return; fi
  if awk "BEGIN{exit !($act $op $bound)}"; then
    printf '  ok    %-38s %s %s %s\n' "$desc" "$act" "$op" "$bound"
  else
    printf '  FAIL  %-38s %s NOT %s %s\n' "$desc" "$act" "$op" "$bound"; fails=$((fails+1))
  fi
}

echo "== generate distorted UltraHDR JPEG (q15 re-encode) =="
"$MK_DISTORTED" "$UHDR_JPEG" "$DIST_JPEG" 15 15 || { echo "make_distorted failed"; exit 2; }

echo "== UltraHDR JPEG: identity = per-metric best =="
assert "jpeg identity ssim2"       "$(val ssim2       "$UHDR_JPEG" "$UHDR_JPEG")" ">=" 99.9
assert "jpeg identity dssim"       "$(val dssim       "$UHDR_JPEG" "$UHDR_JPEG")" "<=" 0.0001
assert "jpeg identity butteraugli" "$(val butteraugli "$UHDR_JPEG" "$UHDR_JPEG")" "<=" 0.0001
assert "jpeg identity zensim"      "$(val zensim      "$UHDR_JPEG" "$UHDR_JPEG")" ">=" 99.9

echo "== UltraHDR JPEG: q15 re-encode scores worse (discrimination) =="
assert "jpeg distorted ssim2"       "$(val ssim2       "$UHDR_JPEG" "$DIST_JPEG")" "<" 95
assert "jpeg distorted dssim"       "$(val dssim       "$UHDR_JPEG" "$DIST_JPEG")" ">" 0
assert "jpeg distorted butteraugli" "$(val butteraugli "$UHDR_JPEG" "$DIST_JPEG")" ">" 0
assert "jpeg distorted zensim"      "$(val zensim      "$UHDR_JPEG" "$DIST_JPEG")" "<" 95

echo "== HEIC: identity = per-metric best =="
assert "heic identity ssim2" "$(val ssim2 "$HEIC_SM" "$HEIC_SM")" ">=" 99.9
assert "heic identity dssim" "$(val dssim "$HEIC_SM" "$HEIC_SM")" "<=" 0.0001

echo "== EXR: identity, and different content discriminates =="
assert "exr identity ssim2"  "$(val ssim2 "$EXR_SMALL" "$EXR_SMALL")" ">=" 99.9
assert "exr different ssim2" "$(val ssim2 "$EXR_A"     "$EXR_B")"     "<" 50

echo "== cross-path: HEIC decode reproduces the EXR pixel-for-pixel =="
assert "heic-vs-exr ssim2"       "$(val ssim2       "$HEIC_X" "$EXR_X")" ">=" 99.9
assert "heic-vs-exr butteraugli" "$(val butteraugli "$HEIC_X" "$EXR_X")" "<=" 0.0001

echo "== fleet worker: batch --hdr over a mixed-input chunk (JPEG+HEIC+EXR) =="
BATCH_TSV=${BATCH_TSV:-/tmp/hdr_batch_pairs.tsv}
BATCH_OUT=${BATCH_OUT:-/tmp/hdr_batch_out.tsv}
{
  printf 'ref_path\tdist_path\tlabel\n'
  printf '%s\t%s\tjpeg_identity\n'  "$UHDR_JPEG" "$UHDR_JPEG"
  printf '%s\t%s\tjpeg_distorted\n' "$UHDR_JPEG" "$DIST_JPEG"
  printf '%s\t%s\theic_vs_exr\n'    "$HEIC_X"    "$EXR_X"
} > "$BATCH_TSV"
if "$BIN" batch --metric ssim2 --hdr --pairs "$BATCH_TSV" --output "$BATCH_OUT" 2>/dev/null; then
  printf '  ok    %-38s %s\n' "batch --hdr ran (3 rows)" "$BATCH_OUT"
  assert "batch row jpeg_identity ssim2" "$(awk -F'\t' '$3=="jpeg_identity"{print $4}'  "$BATCH_OUT")" ">=" 99.9
  assert "batch row jpeg_distorted ssim2" "$(awk -F'\t' '$3=="jpeg_distorted"{print $4}' "$BATCH_OUT")" "<" 95
  assert "batch row heic_vs_exr ssim2"   "$(awk -F'\t' '$3=="heic_vs_exr"{print $4}'    "$BATCH_OUT")" ">=" 99.9
else
  printf '  FAIL  %-38s\n' "batch --hdr exited nonzero"; fails=$((fails+1))
fi

echo
if [[ $fails -eq 0 ]]; then echo "ALL HDR SMOKE ASSERTIONS PASSED"; else echo "$fails ASSERTION(S) FAILED"; fi
exit $fails
