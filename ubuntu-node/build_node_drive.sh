#!/usr/bin/env bash
# build_node_drive.sh — turn a whole disk INTO a persistent, boot-anywhere zen compute node.
#
# Unlike the old installer flow, the target drive IS the distro: we partition it, debootstrap
# Ubuntu 26.04 (Resolute Raccoon) straight onto it, bake in Docker + the worker + SSH key + a
# 7-day R2 cred, and install a bootloader that comes up on ANY UEFI *or* legacy-BIOS machine on
# the LAN. Move the drive to a spare box, boot it, and it starts scoring — no install step, no
# Tailscale, no cloud-init. mDNS makes it reachable as <hostname>.local on the LAN.
#
# Runs on the DEV box as root. WSL2 note: WSL's loop driver has max_part=0, so we can't build in
# a loop-mounted image file — we provision the physical block device directly (real disks get
# partition nodes). The drive must therefore be attached to WSL (`wsl --mount --bare \\.\PHYSICALDRIVEn`).
#
# Usage:
#   sudo bash build_node_drive.sh --device /dev/sdX [--host zen-node-1] [--yes]
#
# --yes skips the "type the model to confirm" wipe guard (for re-runs). Everything on --device is destroyed.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUITE="${ZEN_SUITE:-resolute}"                 # Ubuntu 26.04 LTS
MIRROR="${ZEN_MIRROR:-http://archive.ubuntu.com/ubuntu}"
SECMIRROR="${ZEN_SECMIRROR:-http://security.ubuntu.com/ubuntu}"
HOSTNAME_N="zen-node-1"
DEVICE=""
ASSUME_YES=0
# dev-box SSH key authorized on the node (from the retired installer seed).
SSH_PUBKEY='ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIGI7zVd/Fd0aIgq9ykown4UKZIPc0cp/NTTG1hOtTnW7 lilith@lilith'
WORKER_IMAGE="${ZEN_WORKER_IMAGE:-ghcr.io/imazen/zenfleet-worker:exec}"
LOG="$HOME/tmp/build_node_$(printf %s "$HOSTNAME_N").log"

while [ $# -gt 0 ]; do
  case "$1" in
    --device) DEVICE="$2"; shift 2;;
    --host)   HOSTNAME_N="$2"; shift 2;;
    --yes)    ASSUME_YES=1; shift;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[ -n "$DEVICE" ] || { echo "need --device /dev/sdX" >&2; exit 2; }
[ -b "$DEVICE" ] || { echo "$DEVICE is not a block device" >&2; exit 2; }
[ "$(id -u)" = 0 ] || { echo "run as root" >&2; exit 2; }
mkdir -p "$HOME/tmp"; exec > >(tee -a "$LOG") 2>&1
echo "=== build_node_drive $(date -u +%FT%TZ) device=$DEVICE host=$HOSTNAME_N suite=$SUITE ==="

# ---- guardrails: never eat a mounted / system disk -------------------------------------------
BASE="$(basename "$DEVICE")"
if lsblk -no MOUNTPOINT "$DEVICE" | grep -qE '.'; then
  echo "REFUSING: $DEVICE has mounted partitions:"; lsblk "$DEVICE"; exit 1
fi
# refuse if it backs the running root / a WSL system mount
if grep -qE "^/dev/$BASE" /proc/mounts; then echo "REFUSING: $DEVICE is in /proc/mounts"; exit 1; fi
SIZE_H="$(lsblk -dno SIZE "$DEVICE")"; MODEL="$(lsblk -dno MODEL "$DEVICE" | tr -s ' ' || true)"
echo "target: $DEVICE  size=$SIZE_H  model='${MODEL:-unknown}'"
if [ "$ASSUME_YES" != 1 ]; then
  echo "This ERASES everything on $DEVICE ($SIZE_H). Type the size shown above to proceed:"
  read -r ans; [ "$ans" = "$SIZE_H" ] || { echo "mismatch ('$ans' != '$SIZE_H') — aborting"; exit 1; }
fi

# ---- dev-box build deps -----------------------------------------------------------------------
need_pkgs=""
command -v mmdebstrap >/dev/null || need_pkgs="$need_pkgs mmdebstrap"
command -v sgdisk    >/dev/null || need_pkgs="$need_pkgs gdisk"
command -v mkfs.vfat >/dev/null || need_pkgs="$need_pkgs dosfstools"
command -v mkfs.ext4 >/dev/null || need_pkgs="$need_pkgs e2fsprogs"
if [ -n "$need_pkgs" ]; then echo "installing build deps:$need_pkgs"; apt-get update -qq; apt-get install -y $need_pkgs; fi

# ---- partition: 1M BIOS-boot + 1G ESP + rest ext4 ---------------------------------------------
echo "--- partitioning $DEVICE ---"
for p in "$DEVICE"*[0-9]; do umount "$p" 2>/dev/null || true; done
wipefs -a "$DEVICE" >/dev/null 2>&1 || true
sgdisk --zap-all "$DEVICE"
sgdisk -n1:0:+1M   -t1:ef02 -c1:"BIOS"   "$DEVICE"     # legacy-BIOS grub core
sgdisk -n2:0:+1G   -t2:ef00 -c2:"ESP"    "$DEVICE"     # UEFI system partition
sgdisk -n3:0:0     -t3:8300 -c3:"zenroot" "$DEVICE"    # root, rest of disk
partprobe "$DEVICE" 2>/dev/null || true; sleep 2
# resolve partition node names (sdX2 vs nvmeXn1p2)
p_of(){ case "$DEVICE" in *[0-9]) echo "${DEVICE}p$1";; *) echo "${DEVICE}$1";; esac; }
ESP="$(p_of 2)"; ROOT="$(p_of 3)"
[ -b "$ESP" ] && [ -b "$ROOT" ] || { echo "partition nodes missing ($ESP,$ROOT)"; lsblk "$DEVICE"; exit 1; }

echo "--- formatting ---"
mkfs.vfat -F32 -n ESP "$ESP"
mkfs.ext4 -F -L zenroot "$ROOT"
ROOT_UUID="$(blkid -s UUID -o value "$ROOT")"
ESP_UUID="$(blkid -s UUID -o value "$ESP")"

# ---- mount + debootstrap ----------------------------------------------------------------------
T="$HOME/tmp/zennode-root"; mkdir -p "$T"
mount "$ROOT" "$T"     # ESP is mounted AFTER debootstrap (mmdebstrap wants an ~empty target)
cleanup(){ set +e; for m in "$T/boot/efi" "$T/dev/pts" "$T/dev" "$T/proc" "$T/sys" "$T/run" "$T"; do umount -l "$m" 2>/dev/null; done; }
trap cleanup EXIT

SRC="$HOME/tmp/zennode-sources.list"
cat > "$SRC" <<EOF
deb $MIRROR $SUITE main restricted universe multiverse
deb $MIRROR $SUITE-updates main restricted universe multiverse
deb $SECMIRROR $SUITE-security main restricted universe multiverse
EOF

# Package set: bare-metal kernel + both bootloaders + docker + ssh + mDNS + Skylake microcode.
# NB: no systemd-timesyncd — the 'important' variant already pulls chrony as the time-daemon
# (they conflict via the virtual time-daemon package); chrony auto-enables itself on install.
PKGS="linux-generic,grub-efi-amd64,grub-efi-amd64-signed,shim-signed,grub-pc-bin,\
initramfs-tools,intel-microcode,amd64-microcode,systemd-sysv,udev,netplan.io,\
systemd-resolved,openssh-server,docker.io,avahi-daemon,libnss-mdns,\
sudo,ca-certificates,curl,jq,zstd,less,vim-tiny,htop,ethtool,pciutils,usbutils,cloud-guest-utils"

echo "--- mmdebstrap $SUITE -> $T (this pulls ~700 MB) ---"
nice -n 19 ionice -c 3 mmdebstrap \
  --variant=important \
  --components=main,restricted,universe,multiverse \
  --architectures=amd64 \
  --include="$PKGS" \
  --aptopt='Acquire::Retries "5"' \
  --skip=check/empty \
  "$SUITE" "$T" "$SRC"

# ---- configure inside chroot ------------------------------------------------------------------
mkdir -p "$T/boot/efi"; mount "$ESP" "$T/boot/efi"
mount -t proc  proc  "$T/proc"
mount -t sysfs sys   "$T/sys"
mount --bind /dev    "$T/dev"
mount --bind /dev/pts "$T/dev/pts"
mount -t tmpfs tmpfs "$T/run"

install -Dm644 "$SRC" "$T/etc/apt/sources.list"

cat > "$T/etc/fstab" <<EOF
UUID=$ROOT_UUID  /          ext4  errors=remount-ro  0 1
UUID=$ESP_UUID   /boot/efi  vfat  umask=0077         0 1
EOF

echo "$HOSTNAME_N" > "$T/etc/hostname"
cat > "$T/etc/hosts" <<EOF
127.0.0.1 localhost
127.0.1.1 $HOSTNAME_N
::1 localhost ip6-localhost ip6-loopback
EOF

# any wired NIC -> DHCP (portable across machines); systemd-networkd + mDNS resolution
mkdir -p "$T/etc/systemd/network"
cat > "$T/etc/systemd/network/10-lan.network" <<'EOF'
[Match]
Name=en* eth*
[Network]
DHCP=yes
MulticastDNS=yes
EOF

# boot-anywhere initramfs (USB/NVMe/xhci + most storage drivers)
echo 'MODULES=most' > "$T/etc/initramfs-tools/conf.d/zennode"

# grub: headless-friendly, no OS prober, serial + vga console for debugging a dud boot
cat > "$T/etc/default/grub" <<EOF
GRUB_DEFAULT=0
GRUB_TIMEOUT=3
GRUB_DISTRIBUTOR=zen-node
GRUB_CMDLINE_LINUX_DEFAULT=""
GRUB_CMDLINE_LINUX="console=tty0 console=ttyS0,115200 net.ifnames=1"
GRUB_DISABLE_OS_PROBER=true
GRUB_TERMINAL="console serial"
GRUB_SERIAL_COMMAND="serial --unit=0 --speed=115200"
EOF

# ---- worker env + baked 7-day R2 cred ---------------------------------------------------------
mkdir -p "$T/etc/zen-node"
CREDLINES="$(bash "$HERE/mint_cred.sh")" || { echo "cred mint failed — is the CF token present?"; exit 1; }
for cf in "${R2_CREDENTIALS_FILE:-}" "${SUDO_USER:+/home/$SUDO_USER/.config/cloudflare/r2-credentials}" \
          /home/lilith/.config/cloudflare/r2-credentials "$HOME/.config/cloudflare/r2-credentials"; do
  [ -n "$cf" ] && [ -r "$cf" ] && { R2_ACCT="$(grep -m1 R2_ACCOUNT_ID "$cf" | cut -d= -f2 | tr -d '\r"')"; break; }
done
R2_ENDPOINT="https://${R2_ACCT}.r2.cloudflarestorage.com"
cat > "$T/etc/zen-node/worker.env" <<EOF
# 7-day R2 temp cred (session-token form). Re-baked by the dev-box refresh cron (see README).
$CREDLINES
AWS_REGION=auto
ZEN_R2_ENDPOINT=$R2_ENDPOINT
ZEN_BUCKET=zentrain
ZEN_POOL_RUNLIST=s3://zentrain/jobs/_pool/runlist.tsv
ZEN_CORPUS_PREFIX=refs/clean-picker-corpus-2026-06-26
ZEN_MAX_MIN=700
ZEN_CORE_OVERSUBSCRIBE=1
ZEN_PERSISTENT_EXEC=1
RAYON_NUM_THREADS=1
OMP_NUM_THREADS=1
ZEN_CHUNK_WALL_SEC=20
ZEN_PASS_TIMEOUT=5400
ZEN_PROVIDER=basement
ZEN_WORKER=$HOSTNAME_N
EOF
chmod 600 "$T/etc/zen-node/worker.env"

# worker unit — dedicated box, all cores; docker pulls the image on first start
cat > "$T/etc/systemd/system/zen-worker.service" <<EOF
[Unit]
Description=zen backfill worker (zensim-720)
Wants=network-online.target docker.service
After=network-online.target docker.service
[Service]
ExecStartPre=-/usr/bin/docker rm -f zen720
ExecStart=/usr/bin/docker run --rm --name zen720 --env-file /etc/zen-node/worker.env \\
  --entrypoint /usr/local/bin/fleet-entrypoint.sh $WORKER_IMAGE
ExecStop=/usr/bin/docker rm -f zen720
Restart=always
RestartSec=10
[Install]
WantedBy=multi-user.target
EOF

# zen user (console pw "zencompute"; SSH key-only), sudo + docker, dev-box key authorized
cat > "$T/tmp/inchroot.sh" <<CHROOT
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
useradd -m -s /bin/bash -G sudo zen || true
echo 'zen:zencompute' | chpasswd
usermod -aG docker zen || true
passwd -l root
install -d -m700 -o zen -g zen /home/zen/.ssh
printf '%s\n' '$SSH_PUBKEY' > /home/zen/.ssh/authorized_keys
chown zen:zen /home/zen/.ssh/authorized_keys; chmod 600 /home/zen/.ssh/authorized_keys
mkdir -p /etc/ssh/sshd_config.d
printf 'PasswordAuthentication no\nPermitRootLogin no\nKbdInteractiveAuthentication no\n' > /etc/ssh/sshd_config.d/10-zen.conf
# mDNS: resolve *.local
sed -i 's/^hosts:.*/hosts: files mdns4_minimal [NOTFOUND=return] dns/' /etc/nsswitch.conf || true
systemctl enable ssh docker avahi-daemon systemd-networkd systemd-resolved zen-worker
systemctl enable chrony 2>/dev/null || true
ln -sf /run/systemd/resolve/stub-resolv.conf /etc/resolv.conf || true
# bootloader: Ubuntu's SIGNED grub reads its config from /EFI/ubuntu (its baked-in prefix), so we
# MUST populate that dir or it drops to a grub> prompt on first boot. Install BOTH:
#   1. EFI/ubuntu  — the path the signed grub actually reads (--no-nvram: safe inside chroot/WSL,
#                    no efibootmgr). This is the fix for the "getting grub>" first-boot failure.
#   2. EFI/BOOT    — the machine-independent removable fallback (boots on ANY UEFI firmware).
# plus legacy i386-pc BIOS for CSM-only boards. All three resolve to /boot/grub/grub.cfg by fs-uuid.
grub-install --target=x86_64-efi --efi-directory=/boot/efi --bootloader-id=ubuntu --no-nvram --recheck || true
grub-install --target=x86_64-efi --efi-directory=/boot/efi --removable --recheck "$DEVICE" || \
  grub-install --target=x86_64-efi --efi-directory=/boot/efi --removable --recheck
grub-install --target=i386-pc --recheck "$DEVICE"
update-initramfs -c -k all
update-grub
CHROOT
chroot "$T" bash /tmp/inchroot.sh
rm -f "$T/tmp/inchroot.sh"

sync
echo "=== DONE. $DEVICE is now a bootable zen-node ($HOSTNAME_N). ==="
echo "Move it to the target box, boot from it (UEFI: pick the removable/USB entry)."
echo "It DHCPs on the LAN and appears as $HOSTNAME_N.local; zen-worker starts automatically."
echo "Verify from dev box:  ssh zen@$HOSTNAME_N.local 'systemctl status zen-worker --no-pager'"
