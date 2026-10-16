#!/usr/bin/env bash
set -euo pipefail
root=$HOME/tmp/pieapp-oracle
PIEAPP_LEGACY_NUM_PATCHES=1024 just nn-pieapp-oracle "$root/python2/bin/python" "$root/original-py2" "$HOME/tmp/nn-phase0/ladder/1415_128_ref.png" "$HOME/tmp/nn-phase0/ladder/1415_128_dist.png" "$root/sparse128.log" sparse
"$HOME/tmp/nnrt-research/venv/bin/python" -I crates/zenmetrics-nn/models/pieapp/sampling_probe.py --checkpoint "$root/upstream/weights/PieAPPv0.1.pth" --reference "$HOME/tmp/nn-phase0/ladder/1415_128_ref.png" --distorted "$HOME/tmp/nn-phase0/ladder/1415_128_dist.png" --target 1.6797861082006782 --out "$root/sampling128"
