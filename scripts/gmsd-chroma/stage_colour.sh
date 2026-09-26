#!/usr/bin/env bash
# Canonical decoding only; no label read or accuracy calculation.
set -euo pipefail
python3 scripts/gmsd-chroma/preflight.py
root=/var/tmp/gmsd-chroma/colour_v2
decoder=/var/tmp/gmsd-chroma/bin/gmsbank_decode_dump
printf '%s  %s\n' 3306465d56d279a512b02c3a63de701e9b634d6aebda5e12fb164817ba491122 "$decoder" | sha256sum -c -
test ! -e "$root/rgb"
"$decoder" dump "$root/decode_pairs.tsv" "$root/rgb"
python3 scripts/gmsd-chroma/prepare_colour_rgb.py verify
