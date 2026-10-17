#!/usr/bin/env bash
# Create an owned adapter environment; existing oracle environments stay read-only.
# No automatic dependency resolution: only these named public packages install.
set -euo pipefail
export TMPDIR="$HOME/tmp/iqa-compete/build-scratch"
mkdir -p "$TMPDIR"
out=$1
uv venv --python "$HOME/tmp/nnrt-research/venv/bin/python" "$out"
site=$(printf '%s\n' "$out"/lib/python*/site-packages)
printf '%s\n' "$HOME"/tmp/nnrt-research/venv/lib/python*/site-packages "$HOME"/tmp/nr-classic-scratch/venv/lib/python*/site-packages > "$site/oracle-environments.pth"
uv pip install --python "$out/bin/python" --no-deps torchmetrics==1.9.0 pytorch-msssim==1.0.0 lightning-utilities==0.15.3 packaging==26.3 cvvdp==0.5.7 ffmpeg-python==0.2.0 matplotlib==3.11.2 contourpy==1.4.0 cycler==0.12.1 fonttools==4.67.0 kiwisolver==1.5.1 pyparsing==3.3.3 python-dateutil==2.9.0.post0 six==1.17.0 PyEXR==0.5.0 OpenEXR==3.5.2
"$out/bin/python" -c 'import importlib.metadata as m,json; print(json.dumps({n:m.version(n) for n in ["torch","torchvision","numpy","scipy","scikit-image","piq","torchmetrics","pytorch-msssim","lpips","DISTS-pytorch"]},indent=2))'
