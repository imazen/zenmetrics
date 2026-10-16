#!/usr/bin/env bash
# Original source / original torch runtime, CPU redirection only.
set -u
root=$HOME/tmp/pieapp-oracle
just nn-pieapp-oracle "$root/python2/bin/python" "$root/original-py2" "$HOME/tmp/nn-phase0/ladder/1415_128_ref.png" "$HOME/tmp/nn-phase0/ladder/1415_128_dist.png" "$root/original128_noconfig.log"
echo "original_noconfig_exit=$?"
for patches in 1 121 1024; do
    PIEAPP_LEGACY_NUM_PATCHES=$patches just nn-pieapp-oracle "$root/python2/bin/python" "$root/original-py2" "$HOME/tmp/nn-phase0/ladder/1415_128_ref.png" "$HOME/tmp/nn-phase0/ladder/1415_128_dist.png" "$root/original128_${patches}.log"
    echo "original_configuration_${patches}_exit=$?"
done
for fill in 0 255; do
    PIEAPP_LEGACY_NUM_PATCHES=1 just nn-pieapp-oracle "$root/python2/bin/python" "$root/original-py2" "$root/carriers/1415_65_${fill}_ref.png" "$root/carriers/1415_65_${fill}_dist.png" "$root/original65_${fill}.log"
    echo "original_carrier_${fill}_exit=$?"
done
