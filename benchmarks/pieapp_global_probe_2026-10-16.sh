#!/usr/bin/env bash
# Probe a missing legacy configuration global, preserving all reference operations.
set -u
root=$HOME/tmp/pieapp-oracle
for patches in 1 121 1024; do
    PIEAPP_LEGACY_NUM_PATCHES=$patches just nn-pieapp-oracle "$root/venv/bin/python" "$root/upstream" "$HOME/tmp/nn-phase0/ladder/1415_128_ref.png" "$HOME/tmp/nn-phase0/ladder/1415_128_dist.png" "$root/global128_${patches}.log"
    echo "configuration_${patches}_exit=$?"
done
PIEAPP_LEGACY_NUM_PATCHES=1 just nn-pieapp-oracle "$root/venv/bin/python" "$root/upstream" "$root/carriers/1415_65_0_ref.png" "$root/carriers/1415_65_0_dist.png" "$root/global65_0.log"
echo "carrier_zero_exit=$?"
PIEAPP_LEGACY_NUM_PATCHES=1 just nn-pieapp-oracle "$root/venv/bin/python" "$root/upstream" "$root/carriers/1415_65_255_ref.png" "$root/carriers/1415_65_255_dist.png" "$root/global65_255.log"
echo "carrier_white_exit=$?"
