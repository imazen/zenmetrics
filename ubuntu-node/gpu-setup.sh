#!/usr/bin/env bash
# gpu-setup.sh — make an installed zen node CUDA-capable (driver + container toolkit + docker runtime).
#
# Run ON a node that has an NVIDIA GPU (as the zen user; it uses sudo). Idempotent — safe to re-run.
# It does NOT switch the worker to the GPU image; it just makes the GPU available. To actually use
# the GPU afterward, edit /etc/systemd/system/zen-worker.service:
#     image  ghcr.io/imazen/zenfleet-worker:exec   ->  :exec-gpu
#     add    --gpus all   to the `docker run` line
#   then: sudo systemctl daemon-reload && sudo systemctl restart zen-worker
#
# A REBOOT is required after this script the first time — the NVIDIA module can't load while nouveau
# holds the GPU; the driver install blacklists nouveau, and a reboot brings up nvidia cleanly.
# From the dev box, the node reboots into Ubuntu on its own (PXE -> GRUB -> worker flag), e.g.
#     ./ubuntu-node/pxe/fleet-pxe worker <mac>   # ensure worker flag is set
#     ssh zen@<node> sudo systemctl reboot
#
#   Usage (on the node):  sudo bash gpu-setup.sh
#   Override driver:      sudo NVIDIA_DRIVER=nvidia-driver-595-open bash gpu-setup.sh
set -euo pipefail
[ "$(id -u)" = 0 ] || exec sudo -E bash "$0" "$@"

echo "== 0/5 preconditions =="
if ! lspci -nn 2>/dev/null | grep -qiE 'NVIDIA.*(VGA|3D|Display)'; then
  echo "no NVIDIA GPU found on this box — nothing to do."; exit 0
fi
lspci -nn | grep -iE 'NVIDIA.*(VGA|3D)' | head -1
export DEBIAN_FRONTEND=noninteractive

echo "== 1/5 NVIDIA container-toolkit apt repo (if not already present) =="
if ! apt-cache policy nvidia-container-toolkit 2>/dev/null | grep -q 'Candidate:'; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  echo "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/libnvidia-container/stable/deb/amd64 /" \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
fi
apt-get update -qq

echo "== 2/5 driver + container toolkit =="
if [ -n "${NVIDIA_DRIVER:-}" ]; then
  apt-get install -y "$NVIDIA_DRIVER" nvidia-container-toolkit
else
  apt-get install -y ubuntu-drivers-common nvidia-container-toolkit
  ubuntu-drivers install || apt-get install -y nvidia-driver-595-open   # fallback if autodetect misses
fi

echo "== 3/5 wire the NVIDIA runtime into docker =="
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

echo "== 4/5 restore PXE-first boot order (the driver install bumps the local OS ahead of PXE, =="
echo "==     which breaks the fleet flag mechanism) =="
if command -v efibootmgr >/dev/null; then
  pxe="$(efibootmgr 2>/dev/null | grep -iE 'PXE IPv4' | grep -oE 'Boot[0-9A-Fa-f]{4}' | head -1 | sed 's/Boot//')"
  if [ -n "$pxe" ]; then
    rest="$(efibootmgr 2>/dev/null | awk -F': ' '/^BootOrder/{print $2}' | tr ',' '\n' | grep -vix "$pxe" | paste -sd, -)"
    efibootmgr -o "${pxe}${rest:+,$rest}" >/dev/null && echo "BootOrder set to ${pxe}${rest:+,$rest} (PXE first)"
  else
    echo "  (no 'PXE IPv4' UEFI entry found — check the box's BIOS network-boot setting)"
  fi
fi

echo "== 5/5 done =="
echo "REBOOT the node now to load the driver:  sudo systemctl reboot"
echo "After reboot, verify:  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader"
echo "GPU stays UNUSED until you switch zen-worker.service to :exec-gpu + --gpus all (see header)."
