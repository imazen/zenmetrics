#!/usr/bin/env bash
# Build + push ghcr.io/imazen/zen-train:hybrid-cpu — the per-codec dual-model
# picker trainer image. A TAG on the canonical `zen-train` package (NOT a new
# package; see ghcr-packages.json + `just ghcr-check`).
#
# Bake-everything: the Rust bins are PRECOMPILED and COPY'd in (we never cargo
# build in the Dockerfile); torch/sklearn/pandas are pip-installed at BUILD time
# only. Snap-docker on this WSL2 box can't read /tmp, so the build context is
# staged under $HOME.
#
#   bash scripts/train/build_hybrid_image.sh [--push] [--tag <tag>]
set -euo pipefail

REPO_ZM="${REPO_ZM:-/home/lilith/work/zen/zenmetrics}"
ZA="${ZA:-/home/lilith/work/zen/zenanalyze}"
PICKER_TRAIN="${PICKER_TRAIN:-/home/lilith/work/zen/zenanalyze--dualmodel/zenpicker-train}"
IMAGE="${IMAGE:-ghcr.io/imazen/zen-train}"
TAG="hybrid-cpu"
PUSH=0
while [ $# -gt 0 ]; do case "$1" in
  --push) PUSH=1; shift;;
  --tag) TAG="$2"; shift 2;;
  *) echo "unknown arg $1"; exit 2;;
esac; done

PICKER_TREE_AB="${PICKER_TREE_AB:-$PICKER_TRAIN/target/release/picker_tree_ab}"
ZENPREDICT_BAKE="${ZENPREDICT_BAKE:-$ZA/target/debug/zenpredict-bake}"
[ -x "$PICKER_TREE_AB" ]  || { echo "FATAL: picker_tree_ab not found at $PICKER_TREE_AB (build it: cargo build --release -p zenpicker-train --bin picker_tree_ab)"; exit 1; }
[ -x "$ZENPREDICT_BAKE" ] || { echo "FATAL: zenpredict-bake not found at $ZENPREDICT_BAKE"; exit 1; }

# glibc sanity — bins must require <= the base image's glibc (ubuntu:24.04 = 2.39).
maxglibc=$(objdump -T "$PICKER_TREE_AB" 2>/dev/null | grep -oE 'GLIBC_[0-9.]+' | sort -V | tail -1 || true)
echo "picker_tree_ab max glibc req: ${maxglibc:-none} (base ubuntu:24.04 has 2.39)"

CTX="$HOME/work/zen/_hybrid-train-ctx"
echo "### staging build context at $CTX"
rm -rf "$CTX"; mkdir -p "$CTX/bins" "$CTX/code"
cp "$PICKER_TREE_AB"  "$CTX/bins/picker_tree_ab"
cp "$ZENPREDICT_BAKE" "$CTX/bins/zenpredict-bake"
chmod +x "$CTX/bins/"*

# training code (exclude caches)
rsync -a --exclude='__pycache__' --exclude='.pytest_cache' "$ZA/zentrain/tools/"    "$CTX/code/zentrain-tools/"
rsync -a --exclude='__pycache__' --exclude='.pytest_cache' "$ZA/zentrain/examples/" "$CTX/code/zentrain-examples/"
rsync -a --exclude='__pycache__' --exclude='.pytest_cache' "$ZA/tools/"             "$CTX/code/za-tools/"
rsync -a --exclude='__pycache__' --exclude='.pytest_cache' "$REPO_ZM/scripts/picker/" "$CTX/code/picker/"
# prep_combined + cart_analysis live in scripts/train but the runner expects them under /opt/picker
cp "$REPO_ZM/scripts/train/prep_combined.py" "$CTX/code/picker/prep_combined.py"
cp "$REPO_ZM/scripts/train/cart_analysis.py" "$CTX/code/picker/cart_analysis.py"

cp "$REPO_ZM/scripts/train/dualmodel_runner.sh" "$CTX/dualmodel_runner.sh"
cp "$REPO_ZM/scripts/train/Dockerfile.hybrid-cpu" "$CTX/Dockerfile"

echo "### context contents:"; find "$CTX" -maxdepth 2 -type d | sed "s#$CTX#  ctx#"
du -sh "$CTX"

echo "### docker build $IMAGE:$TAG"
docker build -t "$IMAGE:$TAG" "$CTX" 2>&1 | tail -30

if [ "$PUSH" = 1 ]; then
  echo "### ghcr login (gh token) + push"
  gh auth token | docker login ghcr.io -u lilith --password-stdin
  docker push "$IMAGE:$TAG"
  echo "### pushed $IMAGE:$TAG"
  echo "(if first push: ensure the package is PUBLIC so fleet boxes pull anonymously —"
  echo " gh api -X PATCH /orgs/imazen/packages/container/zen-train/visibility -f visibility=public  ||  set via the ghcr UI)"
else
  echo "### built locally (not pushed). re-run with --push to publish."
fi
echo "DONE $IMAGE:$TAG"
