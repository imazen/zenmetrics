#!/usr/bin/env bash
# Historical probe: records the original dated environment, not a portable recipe.
set -euo pipefail
root=$HOME/tmp/pieapp-oracle
for size in 64 128; do
    just nn-pieapp-blackbox-probe "$root/python2/bin/python" "$root/original-py2" "$root/upstream/weights/PieAPPv0.1.pth" "$HOME/tmp/nn-phase0/ladder/1415_${size}_ref.png" "$HOME/tmp/nn-phase0/ladder/1415_${size}_dist.png"
done
