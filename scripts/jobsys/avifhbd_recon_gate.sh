#!/usr/bin/env bash
# avifhbd_recon_gate.sh — the bd10 x MULTI-TILE x LOW-PRESET recon-parity gate.
#
# WHY THIS EXISTS. imazen/zenav1-svt#18 shipped structurally wrong bd10 pixels
# TWICE, and both times every depth gate and the C-parity axis stayed green:
#   * G3 reads av1C + sequence header + decoder ImageInfo. All three said
#     "10-bit", correctly, while the PIXELS were destroyed. Depth gates are
#     blind to a recon defect by construction.
#   * The C-parity axis ran only presets 6/10/13. Round 2 of the bug was in
#     `dr_predict_hbd`, and DIRECTIONAL modes only enter the candidate set at
#     presets 0-5 — so the whole parity suite sat outside the broken band.
# This gate covers the intersection both bugs hid in: a bd10 encode that AV1
# forces to a multi-tile grid, at a preset low enough for directional intra.
#
# THE TRIGGERS (a (0,0) tile request is CLAMPED UP past either, so the encode
# is multi-tile whether or not tiles were asked for):
#   * width > 4096                        -- area-independent
#   * sb-aligned area > 4096*2304         -- 9,437,184
#
# THE ASSERTION is bd10-vs-8bit on the SAME cell, which is self-calibrating:
# a healthy bd10 encode is at least as good as its 8-bit twin (that is the
# whole point of the arm), so `bd10 >= 8bit - TOL` catches a recon defect
# without hard-coding a quality floor per image or per q.
#
#   usage: avifhbd_recon_gate.sh [SOURCE.png] [Q] [TOL]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ZM="${ZM_BIN:-$ROOT/target/x86_64-unknown-linux-musl/release/zenmetrics}"
[ -x "$ZM" ] || ZM="$ROOT/target/release/zenmetrics"
SRC="${1:-/mnt/v/output/avifsvt-subsample-2026-09-01/sources/1008.scale3000x4000.png}"
Q="${2:-90}"; TOL="${3:-5.0}"
[ -x "$ZM" ] || { echo "no zenmetrics binary (set ZM_BIN)"; exit 3; }
[ -f "$SRC" ] || { echo "no source: $SRC"; exit 3; }

# Refuse a source that does NOT force multi-tile — a gate that silently runs
# single-tile would pass through both bugs, which is how we got here.
python3 - "$SRC" <<'PY' || exit 3
import sys, struct
b=open(sys.argv[1],'rb').read(32)
w,h=struct.unpack('>II', b[16:24])
sb=lambda x: ((x+63)//64)*64
a=sb(w)*sb(h)
if w>4096 or a>4096*2304:
    print(f"  source {w}x{h} sb-area {a} -> FORCED MULTI-TILE (ok)"); sys.exit(0)
print(f"  source {w}x{h} sb-area {a} does NOT force multi-tile — this gate would prove nothing"); sys.exit(1)
PY

WORK="$(mktemp -d "${TMPDIR:-$HOME/tmp}/avifhbd-recon.XXXXXX")"; trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/src"; cp "$SRC" "$WORK/src/"
# Default plan is the SPEED-4 (SVT preset 4) arm: dead centre of the 0-5
# directional band, and the exact configuration both #18 bugs were measured
# on. The full ladder is available via ZM_PLAN=svt_doe_t1_bd10_ladder, but it
# includes preset 0 on a >9 MP source and costs ~10 min, which makes it a poor
# default for a gate meant to run on every image build.
"$ZM" sweep --codec zenavif --plan "${ZM_PLAN:-svt_doe_t1_bd10_transfer}" --sources "$WORK/src" \
    --q-grid "$Q" --max-deviations 1 --metric ssim2 --output "$WORK/bd10.tsv" >/dev/null 2>&1 \
  || { echo "  bd10 sweep FAILED"; exit 1; }
# max-deviations 0, NOT 1: svt_doe_transfer at 1 emits all 17 main-effect
# arms x 2 speeds, which turned a 20-second control into a 10-minute one.
# The control this gate needs is the DEFAULT stratum only.
"$ZM" sweep --codec zenavif --plan svt_doe_transfer --sources "$WORK/src" \
    --q-grid "$Q" --max-deviations 0 --metric ssim2 --output "$WORK/bd8.tsv" >/dev/null 2>&1 \
  || { echo "  8-bit control sweep FAILED"; exit 1; }

python3 - "$WORK/bd10.tsv" "$WORK/bd8.tsv" "$TOL" <<'PYEOF'
import csv, sys, json, re
S2P={1:0,2:1,3:3,4:4,5:6,6:7,7:9}
def load(p):
    out={}
    for r in csv.DictReader(open(p), delimiter="\t"):
        k=[c for c in r if "ssim2" in c.lower()][0]
        kt=r.get("knob_tuple_json","")
        try: cell=json.loads(kt).get("cell","")
        except Exception: cell=kt
        m=re.match(r"s(\d+)", cell or "")
        if m: out[int(m.group(1))]=(float(r[k]), cell)
    return out
ten, eight = load(sys.argv[1]), load(sys.argv[2])
tol=float(sys.argv[3]); bad=[]
if not ten:
    print("  GATE INCONCLUSIVE - no bd10 cell parsed"); sys.exit(2)
print(f"  {'speed':>5} {'preset':>6} {'bd10':>9} {'8-bit':>9} {'delta':>8}  cell")
for sp in sorted(ten):
    v,cell = ten[sp]
    ctl = eight.get(sp) or eight.get(4) or (next(iter(eight.values())) if eight else None)
    if ctl is None: continue
    b8=ctl[0]; d=v-b8
    if d < -tol: bad.append(sp)
    print(f"  {sp:>5} {str(S2P.get(sp,'?')):>6} {v:>9.3f} {b8:>9.3f} {d:>+8.3f}  {cell}" + ("  <-- FAIL" if d < -tol else ""))
low=[s for s in ten if S2P.get(s,99)<=5]
print(f"\n  cells in the directional band (preset<=5): {len(low)}")
if not low:
    print("  GATE INCONCLUSIVE - no low-preset cell measured"); sys.exit(2)
if bad:
    print(f"  GATE FAIL - {len(bad)} cell(s) more than {tol} ssim2 below the 8-bit control"); sys.exit(1)
print(f"  GATE PASS - every bd10 cell within {tol} ssim2 of its 8-bit control")
PYEOF
