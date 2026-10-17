#!/usr/bin/env bash
# Historical probe: records the original dated environment, not a portable recipe.
# Isolated official runtime, executed as a black box. Never read model source.
set -euo pipefail
root=$HOME/tmp/pieapp-oracle
bash "$root/Miniconda2-py27_4.8.3-Linux-x86_64.sh" -b -p "$root/python2"
"$root/python2/bin/pip" install --no-index --find-links "$root" numpy==1.16.6 PyYAML==3.13 opencv-python==4.2.0.32 torch==0.3.1
cd "$root/original-py2"
"$root/python2/bin/python" -c 'import torch,numpy,cv2; print(torch.__version__,numpy.__version__,cv2.__version__)'
# Diagnostic failures are recorded individually; this is not the landing gate.
set +e
for sampling in dense sparse; do
    "$root/python2/bin/python" test_PieAPP_PT.py --ref_path "$HOME/tmp/nn-phase0/ladder/1415_128_ref.png" --A_path "$HOME/tmp/nn-phase0/ladder/1415_128_dist.png" --sampling_mode "$sampling" --gpu_id -1 > "$root/legacy128_${sampling}.log" 2>&1
    echo "legacy128_${sampling}_exit=$?"
done
