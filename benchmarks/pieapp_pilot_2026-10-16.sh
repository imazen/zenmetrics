#!/usr/bin/env bash
# Historical probe: records the original dated environment, not a portable recipe.
# Diagnostic probes, not a parity gate. Keep every attempted CLI result.
set -u
oracle=$HOME/tmp/pieapp-oracle
for fill in 0 255; do
    just nn-pieapp-oracle "$oracle/venv/bin/python" "$oracle/upstream" "$oracle/carriers/1415_65_${fill}_ref.png" "$oracle/carriers/1415_65_${fill}_dist.png" "$oracle/carrier65_${fill}.log"
    echo "carrier_${fill}_exit=$?"
done
just nn-pieapp-oracle "$oracle/venv/bin/python" "$oracle/upstream" "$HOME/tmp/nn-phase0/ladder/1415_128_ref.png" "$HOME/tmp/nn-phase0/ladder/1415_128_dist.png" "$oracle/pilot128.log"
echo "oracle128_exit=$?"
just nn-pieapp-probe "$HOME/tmp/nnrt-research/venv/bin/python" "$oracle/upstream/weights/PieAPPv0.1.pth" "$HOME/tmp/nn-phase0/ladder/1415_64_ref.png" "$HOME/tmp/nn-phase0/ladder/1415_64_dist.png"
