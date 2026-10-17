#!/usr/bin/env bash
# Historical probe: records the original dated environment, not a portable recipe.
set -u
root=$HOME/tmp/pieapp-oracle
for mode in original modern; do
    if [[ $mode == original ]]; then python=$root/python2/bin/python; upstream=$root/original-py2; else python=$root/venv/bin/python; upstream=$root/upstream; fi
    for size in 64 128; do
        PIEAPP_LEGACY_NUM_PATCHES=1024 just nn-pieapp-oracle "$python" "$upstream" "$HOME/tmp/nn-phase0/ladder/1415_${size}_ref.png" "$HOME/tmp/nn-phase0/ladder/1415_${size}_dist.png" "$root/qualified_${mode}_${size}.log"
        echo "qualified_${mode}_${size}_exit=$?"
    done
    for fill in 0 255; do
        PIEAPP_LEGACY_NUM_PATCHES=1024 just nn-pieapp-oracle "$python" "$upstream" "$root/carriers/1415_65_${fill}_ref.png" "$root/carriers/1415_65_${fill}_dist.png" "$root/qualified_${mode}_65_${fill}.log"
        echo "qualified_${mode}_carrier_${fill}_exit=$?"
    done
done
just nn-pieapp-probe "$HOME/tmp/nnrt-research/venv/bin/python" "$root/upstream/weights/PieAPPv0.1.pth" "$HOME/tmp/nn-phase0/ladder/1415_128_ref.png" "$HOME/tmp/nn-phase0/ladder/1415_128_dist.png"
