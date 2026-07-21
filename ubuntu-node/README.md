# zen compute node — the drive *is* the distro

Turn a whole drive into a persistent, boot-anywhere backfill worker. The drive itself holds a full
Ubuntu 26.04 install with Docker + the worker + your SSH key + a baked R2 credential. Move it to any
spare LAN box, boot it, and it starts claiming and scoring the R2 job pool — no installer step, no
cloud-init, no Tailscale. It DHCPs on the LAN and announces itself over mDNS as `<hostname>.local`.

This replaces the old autoinstall-USB flow (`user-data`/`meta-data`, removed): that *installed onto the
target machine's internal disk*; this makes the removable drive the OS.

## What's here

- `build_node_drive.sh` — **dev-box, root.** Partitions a drive, debootstraps Ubuntu 26.04 onto it,
  bakes in Docker + worker + SSH key + a 7-day R2 cred, installs a UEFI-*and*-BIOS bootloader that
  comes up on any machine. The whole node is built here.
- `mint_cred.sh` — mints a scoped **7-day** R2 temp cred (the R2 temp-cred maximum) and prints it as
  env lines. Used by the build and by `onboard_node.sh`.
- `onboard_node.sh` — dev-box helper to push a *fresh* 7-day cred to a running node and restart the
  worker. Run weekly (cron line inside the script) to keep the cred live.

## Build it (on the dev box)

The build writes a physical block device, so the drive must be visible to WSL. From an elevated
Windows PowerShell, attach it bare:

```powershell
wsl --mount --bare \\.\PHYSICALDRIVEn        # n = the drive's disk number (see: wmic diskdrive list brief)
```

It then shows up in WSL as `/dev/sdX` (check `lsblk`). Build:

```bash
sudo bash ubuntu-node/build_node_drive.sh --device /dev/sdX --host zen-node-1
```

It refuses a mounted/system disk and, unless `--yes`, makes you retype the drive's size to confirm the
wipe. ~10–20 min (≈700 MB debootstrap + write). Bump `--host zen-node-2` for a second drive so mDNS
names don't collide. Log lands in `~/tmp/build_node_<host>.log`.

## Boot it

Move the drive to the target box and boot from it:

- **UEFI** (default on Skylake-era Xeon and anything since): pick the removable/USB entry in the boot
  menu. We install GRUB to the firmware's removable path (`/EFI/BOOT/BOOTX64.EFI`), so it boots on *any*
  UEFI box with no per-machine NVRAM entry.
- **Legacy/CSM-only** old boards: also covered — a BIOS GRUB core is installed to the disk.

First boot brings up networking (DHCP on any wired NIC), Docker pulls `zenfleet-worker:exec` (~1–2 min),
and `zen-worker` starts. From the dev box:

```bash
ssh zen@zen-node-1.local 'systemctl status zen-worker --no-pager'
ssh zen@zen-node-1.local 'docker top zen720 | grep -c zenmetrics'   # ~= core count once scoring
```

SSH is key-only (the dev-box key is baked in). Console password is `zencompute` for a physical login.

**If you land at a `grub>` prompt** (only happens on drives built before the `EFI/ubuntu` fix): the drive
is fine, GRUB just can't find its menu. Boot it by hand —
```
search -f -s /boot/grub/grub.cfg
configfile /boot/grub/grub.cfg
```
— then make it permanent from the booted node:
```
sudo grub-install --target=x86_64-efi --efi-directory=/boot/efi --bootloader-id=ubuntu --recheck
sudo grub-install --target=x86_64-efi --efi-directory=/boot/efi --removable --recheck
sudo update-grub
```

## Credentials

- The drive is built with a **7-day** scoped R2 cred (bucket `zentrain`, object-read-write, only the
  `jobs/ jxl-lossy/runs/ canonical/2026-06-27/ refs/` prefixes) baked into `/etc/zen-node/worker.env`.
  It carries a session token — that's why the worker env has `AWS_SESSION_TOKEN`.
- **7 days is the hard cap** for R2 temp-access-credentials, so keep it live with a weekly push from the
  dev box: `bash ubuntu-node/onboard_node.sh zen-node-1.local`, or the crontab line inside that script.
- **Longer than 7 days?** Only a dashboard-created scoped R2 *API token* (R2 → Manage R2 API Tokens →
  scope to `zentrain`, Object Read & Write, custom expiry) gives a static key+secret with no session
  token and a 30-day+ life. Our management token can't create one programmatically (403), so that path
  needs one manual token creation; drop its key/secret into `worker.env` (no `AWS_SESSION_TOKEN` line)
  if you'd rather not run the weekly refresh.

## Notes

- Same R2 job pool as the tower and the cloud fleet; dedicated box, so it uses all cores (the worker's
  resource-aware admission bounds concurrency).
- GRUB and the kernel have a serial console (`ttyS0,115200`) enabled, so a dud boot on a headless box is
  debuggable over serial.
- Skylake-era Xeon is fully supported: bare-metal `linux-generic` kernel, `intel-microcode`, and the
  worker's runtime SIMD dispatch (AVX2/AVX-512 detected at runtime — the binary is portable baseline).
