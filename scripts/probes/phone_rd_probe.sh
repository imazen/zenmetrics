#!/usr/bin/env bash
# Phase-1 phone-vs-desktop JXL RD probe.
# encode (cjxl-rs) -> decode (zenjxl-decoder-cli) -> dual-display cvvdp.
set -u
CJXL=~/work/zen/jxl-encoder/target/release/cjxl-rs
DJXL=~/work/zen/zenjxl-decoder/target/release/zenjxl-decoder-cli
SCORE=/home/lilith/work/zen/zenmetrics/target/release/examples/score_two_displays
REFDIR=/tmp/cvvdp-display-eval/refs
OUT=/tmp/phone_rd_probe; mkdir -p "$OUT"
TSV="$OUT/results.tsv"
printf 'image\tdistance\tbytes\tbpp\tjod_desktop\tjod_phone\tdelta\n' > "$TSV"

IMAGES="photo_general photo_dark lineart photo_bright screen_text"
# Weighted toward aggressive (low-quality) end per sweep discipline.
DISTANCES="0.7 1.0 1.5 2.0 3.0 4.5 6.0 8.0"
EFFORT=7
PIXELS=$((1024*768))

for img in $IMAGES; do
  ref="$REFDIR/${img}_ref.png"
  [ -f "$ref" ] || { echo "skip $img"; continue; }
  for d in $DISTANCES; do
    jxl="$OUT/${img}_d${d}.jxl"; png="$OUT/${img}_d${d}.png"
    "$CJXL" --effort $EFFORT --distance "$d" "$ref" "$jxl" >/dev/null 2>&1 || { echo "enc fail $img $d"; continue; }
    "$DJXL" "$jxl" "$png" >/dev/null 2>&1 || { echo "dec fail $img $d"; continue; }
    bytes=$(stat -c%s "$jxl")
    bpp=$(python3 -c "print(f'{$bytes*8/$PIXELS:.4f}')")
    line=$("$SCORE" "$ref" "$png" 2>/dev/null)
    jd=$(echo "$line" | cut -f1); jp=$(echo "$line" | cut -f2); dl=$(echo "$line" | cut -f3)
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$img" "$d" "$bytes" "$bpp" "$jd" "$jp" "$dl" | tee -a "$TSV"
  done
done
echo "--- $TSV ---"
