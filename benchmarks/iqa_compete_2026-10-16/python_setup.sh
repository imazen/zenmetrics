#!/usr/bin/env bash
# Create an owned adapter environment; existing oracle environments stay read-only.
# No automatic dependency resolution: only these named public packages install.
set -euo pipefail
out=$1
uv venv --python "$HOME/tmp/nnrt-research/venv/bin/python" "$out"
site=$(printf '%s\n' "$out"/lib/python*/site-packages)
printf '%s\n' "$HOME"/tmp/nnrt-research/venv/lib/python*/site-packages "$HOME"/tmp/nr-classic-scratch/venv/lib/python*/site-packages > "$site/oracle-environments.pth"
uv pip install --python "$out/bin/python" --no-deps torchmetrics==1.9.0 pytorch-msssim==1.0.0 lightning-utilities packaging cvvdp==0.5.7 ffmpeg-python matplotlib contourpy cycler fonttools kiwisolver pyparsing python-dateutil six PyEXR OpenEXR
"$out/bin/python" -c 'import importlib.metadata as m,json; print(json.dumps({n:m.version(n) for n in ["torch","torchvision","numpy","scipy","scikit-image","piq","torchmetrics","pytorch-msssim","lpips","DISTS-pytorch"]},indent=2))'
