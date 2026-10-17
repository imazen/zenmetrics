#!/usr/bin/env bash
# Historical probe: records the original dated environment, not a portable recipe.
set -u
root=$HOME/tmp/pieapp-oracle
for mode in original modern; do
    if [[ $mode == original ]]; then python=$root/python2/bin/python; upstream=$root/original-py2; else python=$root/venv/bin/python; upstream=$root/upstream; fi
    for patches in 1 121 1024; do
        PIEAPP_LEGACY_NUM_PATCHES=$patches just nn-pieapp-oracle "$python" "$upstream" "$HOME/tmp/nn-phase0/ladder/1415_128_ref.png" "$HOME/tmp/nn-phase0/ladder/1415_128_dist.png" "$root/runtime_${mode}_128_${patches}.log"
        echo "runtime_${mode}_${patches}_exit=$?"
    done
    for fill in 0 255; do
        PIEAPP_LEGACY_NUM_PATCHES=1 just nn-pieapp-oracle "$python" "$upstream" "$root/carriers/1415_65_${fill}_ref.png" "$root/carriers/1415_65_${fill}_dist.png" "$root/runtime_${mode}_65_${fill}.log"
        echo "runtime_${mode}_carrier_${fill}_exit=$?"
    done
done
