#!/usr/bin/env bash
# One input-preparation job, invoked through the shared heavy wrapper.
set -euo pipefail
test ! -e /home/lilith/tmp/zensim-paper/rev4/CODEX_QUOTA_STOP.md
python3 scripts/gmsd-chroma/preflight.py
root=/var/tmp/gmsd-chroma
decoder=/var/tmp/gmsd-chroma/bin/gmsbank_decode_dump
printf '%s  %s\n' 3306465d56d279a512b02c3a63de701e9b634d6aebda5e12fb164817ba491122 "$decoder" | sha256sum -c -
"$decoder" dump "$root/kadid_decode_pairs.tsv" "$root/kadid_rgb"
python3 scripts/gmsd-chroma/prepare_pairs.py stage
