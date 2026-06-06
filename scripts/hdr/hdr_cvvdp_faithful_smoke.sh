#!/usr/bin/env bash
# Faithful HDR-metric smoke (chunk 4) — the native linear-planes paths that
# bypass the u8 PU clamp, for cvvdp (4a) AND butteraugli-gpu (4b). Needs a
# gpu-cvvdp + gpu-butteraugli build + a CUDA GPU:
#   cargo build --release -p zen-metrics-cli --no-default-features \
#     --features "png,jpeg,cpu-metrics,gpu-cvvdp,gpu-butteraugli,gpu-cuda,hdr"
#   cargo build --release -p zenhdr-corpus --example make_distorted
#   cargo build --release -p zenhdr-corpus --example synth_highlight_pair
#
# Asserts: cvvdp identity = 10 (JOD max) + butter-gpu identity = 0; q15 + different
# content discriminate; and — the point of chunk 4 — the faithful paths DETECT a
# highlight-only difference (2000→200 cd/m²) that the u8-PU SDR metrics are blind
# to (ssim2 stays 100, u8 butteraugli stays 0).
set -uo pipefail

BIN=${BIN:-./target/release/zen-metrics}
MK_DISTORTED=${MK_DISTORTED:-./target/release/examples/make_distorted}
SYNTH=${SYNTH:-./target/release/examples/synth_highlight_pair}
GPU=${GPU_RUNTIME:-cuda}
UHDR_JPEG=${UHDR_JPEG:-/home/lilith/work/zen/ultrahdr/test_ultrahdr.jpg}
HEIC_X=${HEIC_X:-/mnt/v/heic/IMG_1420.HEIC}
EXR_X=${EXR_X:-/mnt/v/hdr-corpus/refs/IMG_1420.exr}
EXR_A=${EXR_A:-/mnt/v/hdr-corpus/refs/32F76D88-01ED-42BA-95A2-716FEBABF1E2.exr}
EXR_B=${EXR_B:-/mnt/v/hdr-corpus/refs/94067DD9-550D-4573-A8EA-A462D51888D7.exr}
DIST_JPEG=${DIST_JPEG:-/mnt/v/hdr-corpus/distorted_test_ultrahdr_q15.jpg}
HI_REF=${HI_REF:-/mnt/v/hdr-corpus/synth_hi_ref.exr}
HI_DIST=${HI_DIST:-/mnt/v/hdr-corpus/synth_hi_dist.exr}

fails=0
last() { echo "$1" | grep -oE '[a-z0-9_]+=[-0-9.]+' | tail -1 | cut -d= -f2; }
score() { # metric ref dist [extra args...]
  local m=$1 r=$2 d=$3; shift 3
  "$BIN" score --metric "$m" --reference "$r" --distorted "$d" --hdr "$@" 2>&1
}
assert() { # desc actual op bound
  if [[ "$2" == *error* || -z "$2" ]]; then printf '  FAIL  %-40s %s\n' "$1" "$2"; fails=$((fails+1)); return; fi
  if awk "BEGIN{exit !($2 $3 $4)}"; then printf '  ok    %-40s %s %s %s\n' "$1" "$2" "$3" "$4"
  else printf '  FAIL  %-40s %s NOT %s %s\n' "$1" "$2" "$3" "$4"; fails=$((fails+1)); fi
}

# GPU presence gate — loud, not silent (this is a manual harness).
if ! "$BIN" score --metric cvvdp --hdr --gpu-runtime "$GPU" \
     --reference "$UHDR_JPEG" --distorted "$UHDR_JPEG" >/dev/null 2>&1; then
  echo "SKIPPED: cvvdp --hdr did not run — needs a gpu-cvvdp build + a working $GPU GPU."
  echo "         (build with --features gpu-cvvdp,gpu-cuda,hdr; see the header.)"
  exit 2
fi

echo "== generate synthetic highlight-only pair (2000 vs 200 cd/m²) + distorted JPEG =="
"$SYNTH" "$HI_REF" "$HI_DIST" 2000 200 256 || { echo "synth failed"; exit 3; }
"$MK_DISTORTED" "$UHDR_JPEG" "$DIST_JPEG" 15 15 >/dev/null 2>&1 || true

echo "== faithful cvvdp HDR: identity = JOD max, distortions discriminate =="
assert "cvvdp jpeg identity"   "$(last "$(score cvvdp "$UHDR_JPEG" "$UHDR_JPEG" --gpu-runtime "$GPU")")" ">=" 9.99
assert "cvvdp heic-vs-exr"     "$(last "$(score cvvdp "$HEIC_X" "$EXR_X" --gpu-runtime "$GPU")")"        ">=" 9.99
assert "cvvdp jpeg q15 distort" "$(last "$(score cvvdp "$UHDR_JPEG" "$DIST_JPEG" --gpu-runtime "$GPU")")" "<" 9.5
assert "cvvdp exr different"    "$(last "$(score cvvdp "$EXR_A" "$EXR_B" --gpu-runtime "$GPU")")"         "<" 7

echo "== faithful butteraugli-gpu HDR (intensity_target=peak): identity = 0, discriminates =="
# butter emits two columns; `last` returns pnorm3, fine for these checks.
assert "butter-gpu identity"        "$(last "$(score butteraugli-gpu "$UHDR_JPEG" "$UHDR_JPEG" --gpu-runtime "$GPU")")" "<=" 0.0001
assert "butter-gpu jpeg q15 distort" "$(last "$(score butteraugli-gpu "$UHDR_JPEG" "$DIST_JPEG" --gpu-runtime "$GPU")")" ">" 1

echo "== highlight-only A/B: faithful + pu-rescale DETECT; only the pu-CLAMP bug is BLIND =="
# Faithful linear-planes paths see the 2000→200 cd/m² highlight crush.
assert "cvvdp highlight crush (DETECTS)"      "$(last "$(score cvvdp "$HI_REF" "$HI_DIST" --gpu-runtime "$GPU")")"          "<" 7
assert "butter-gpu highlight crush (DETECTS)" "$(last "$(score butteraugli-gpu "$HI_REF" "$HI_DIST" --gpu-runtime "$GPU")")" ">" 10
# The validated DEFAULT SDR feeding (pu-rescale, no clamp) ALSO sees it —
# that IS the bug fix: the highlight range survives the u8 encode, so the
# SSIM-family kernels are no longer blind. (UPIQ: pu-rescale 0.65 > clamp 0.55.)
assert "ssim2 highlight crush (pu-rescale DETECTS)"  "$(last "$(score ssim2 "$HI_REF" "$HI_DIST")")"       "<"  95
# CPU butter now routes through the umbrella's faithful LINEAR feeding (native
# butteraugli_linear, intensity_target = display peak) — the luminance-aware
# metrics never take a u8 transfer, so butter detects the crush on its own.
assert "butter highlight crush (faithful DETECTS)"  "$(last "$(score butteraugli "$HI_REF" "$HI_DIST")")" ">"  1
# The legacy pu-CLAMP feeding (the bug) collapses everything above ~100 cd/m²
# to u8 255 → ref and dist become byte-identical → the SSIM-FAMILY kernels go
# BLIND (ssim2 stays ~100). This is the regression the fix removed.
assert "ssim2 highlight crush (pu-clamp BLIND)"     "$(last "$(score ssim2 "$HI_REF" "$HI_DIST" --hdr-transfer pu-clamp)")"      ">=" 99.9
# butter is luminance-aware → it uses faithful linear regardless of
# --hdr-transfer (the u8 transfer applies only to the SSIM-family), so the
# pu-clamp bug can't blind it. It still DETECTS the crush.
assert "butter highlight crush (transfer-agnostic, DETECTS)" "$(last "$(score butteraugli "$HI_REF" "$HI_DIST" --hdr-transfer pu-clamp)")" ">" 1

echo
if [[ $fails -eq 0 ]]; then echo "ALL FAITHFUL HDR-METRIC ASSERTIONS PASSED"; else echo "$fails ASSERTION(S) FAILED"; fi
exit $fails
