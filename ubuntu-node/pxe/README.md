# zen-pxe — remote network install onto each box's local disk

Install/reinstall Ubuntu 26.04 on basement fleet boxes **over the network**, onto each
box's **own local disk**, controlled from the dev box. No USB, no walking to the machine
(after a one-time BIOS toggle). Runs on the Unraid tower (`192.168.50.170`).

This is the fleet counterpart to `../build_node_drive.sh` (which builds one drive locally).
The runtime OS still lives on each box's local disk — only the *install* is delivered over
the network.

## How it boots

**UEFI (the normal case) PXE-boots GRUB directly** — not iPXE. We tried iPXE first and hit two
walls on real firmware: (1) many UEFI firmwares don't honor iPXE's EFI initrd handoff
(`LoadFile2`), so the kernel boots with no initrd and panics; (2) GRUB chained *from* iPXE loses
`${net_default_mac}`. PXE-booting GRUB directly fixes both — GRUB loads the initrd itself (works
everywhere) and, being PXE-booted, knows its own MAC.

```
box powers on ──UEFI PXE──▶ router DHCP gives IP
                            tower dnsmasq (proxy-DHCP) says "load grubnet.efi via TFTP"
        ──GRUB──▶ net_bootp ──HTTP──▶ tower /api/grub/<its-MAC>
                            the control service decides, keyed on that MAC:
                              • inventory flag  → boot read-only, report disks, power off
                              • install flag AND registered serial → install to THAT serial
                              • otherwise (default) → `exit` to the local disk
```

Legacy BIOS boxes still go through iPXE (`undionly.kpxe` → `boot.ipxe` → `/api/boot/<mac>`).
Proxy-DHCP **coexists** with the router's DHCP — it only adds PXE options, never hands out IPs.
Verified end-to-end 2026-07-21 on a real UEFI box: PXE → GRUB → kernel+initrd → casper → inventory.

## The four safety layers — how we never wipe the wrong disk

1. **Opt-in per MAC.** Default for every box is *local boot*. A box installs only if you
   created its `install` flag. A stray/new/rebooting box never installs by accident.
2. **Disk matched by serial, never "the biggest disk."** You first boot the box **read-only**
   (inventory) so it reports its disk **serials**; you register the exact OS-disk serial; the
   autoinstall matches only that serial and **aborts** if it's absent. A second/data disk with
   a different serial is never a candidate.
3. **Install-once.** The installer clears its own MAC flag mid-install (it has network), so a
   re-PXE can't re-wipe.
4. **Blast radius.** Proxy-DHCP only answers PXE clients; your workstation/phones/tower are
   untouched. Three things must all be true to wipe: you flagged the MAC, you registered the
   serial, and that serial is present.

Layer 1 has a second, equally opt-in door — the console menu below. It replaces "you set a
flag from the dev box" with "a human at the box picked a specific disk and confirmed it";
layers 2-4 apply to it unchanged.

## Installing from the box's own console (no dev box needed)

A box the tower has **never seen registered or installed** gets an interactive GRUB menu
instead of the silent local-boot chainload, so you can provision it standing in front of it:

```
Boot from local disk  [DEFAULT]            <- selected automatically after 30s
Install Ubuntu to /dev/nvme0n1  931.5G  WD_BLACK SN850X (nvme)
  |- NO - go back (nothing is changed)     <- the submenu's default
  |- YES - ERASE serial 25286M803378 - ALL DATA ON IT IS LOST
Scan this hardware (read-only; reboots back to this menu)
```

- **Nothing installs unattended.** The default entry is local boot and the timeout picks it,
  so a box that PXE-boots with nobody watching always falls through to its disk.
- **Two deliberate picks**, and the confirm entry names the exact serial being erased.
- **The tower re-checks the serial** against that MAC's own inventory before arming, so a
  typo or hand-made `/api/choose/...` URL can't target a disk we've never seen. The
  serial-matched autoinstall (layer 2) then makes wiping the wrong disk impossible.
- **No disks listed?** Run the scan — it boots the read-only probe, reports the disks and
  reboots straight back to the menu. The probe sets a one-shot UEFI `BootNext` to the
  firmware's network entry before rebooting, so the return trip goes through PXE even on a
  box whose default boot order prefers a local disk (the ryzen5800xt 2026-08-02 trap:
  scan → reboot → firmware booted the disk → menu never returned). Permanent boot order is
  still yours to set — PXE-first remains the requirement for remotely-managed boxes.

**Boxes we already manage keep the old instant chainload.** jason/ian/i265 are PXE-first, so
they ask the tower on *every* boot; a menu there would delay each boot and leave a wipe option
on screen for whoever is sitting at the kids' PCs. Opt one in deliberately with
`./fleet-pxe menu <mac>` (and `nomenu` to undo) — useful for re-installing an existing box
from its console.

**Console installs get no R2 credential.** Only the dev box holds the Cloudflare token, so
this path renders the install with no `AWS_*` lines and leaves `zen-worker` installed but
**disabled** (enabling it uncredentialed would restart-loop); the box drops an
`/etc/zen-node/NEEDS-CRED` note. It is otherwise complete — ssh key, sudo, mosh/tmux,
herdr/Claude Code/rustup. Finish it from the dev box:

```bash
bash ubuntu-node/enroll_running_node.sh --start <ip>    # mints the 7-day cred, starts the worker
```

The hostname is auto-assigned from the MAC (`60:cf:84:76:20:d2` → `zen-7620d2`); rename later
with `hostnamectl set-hostname`.

## One-time per box (consumer boxes, no BMC)

In BIOS: enable **network/PXE boot** and **Wake-on-LAN**. Set boot order to **PXE first,
local disk second** — then it's remotely controllable (PXE asks the tower every boot; the
default answer is "boot local", so it just falls through to the disk unless you've flagged it).

## Workflow (from the dev box)

```bash
cd ubuntu-node/pxe
./fleet-pxe seen                              # 1. which MACs PXE-booted
./fleet-pxe inventory aa:bb:cc:dd:ee:ff       # 2. read-only boot; it reports disks, powers off
./fleet-pxe show aa:bb:cc:dd:ee:ff            # 3. look at disks; copy the OS disk's SERIAL
./fleet-pxe register aa:bb:cc:dd:ee:ff --serial <SERIAL> --host zen-node-2   # 4. mint 7d cred + render
./fleet-pxe install aa:bb:cc:dd:ee:ff         # 5. opt-in + WoL; installs ONLY to <SERIAL>, self-clears
./fleet-pxe cancel aa:bb:cc:dd:ee:ff          # abort (remove flags) any time
```

After install the box boots local, joins the LAN (DHCP + mDNS `zen-node-2.local`), and runs
`zen-worker`. Refresh its 7-day R2 cred weekly with `../onboard_node.sh zen-node-2.local`.

## Deploy / operate the tower stack

```bash
./deploy.sh            # sync configs + (re)start the 3 containers on the tower
./deploy.sh --assets   # also re-extract kernel/initrd from the ISO
```

Three host-networked containers on the tower: `zen-pxe-dnsmasq` (proxy-DHCP + TFTP),
`zen-pxe-nginx` (:3080 static assets + `/api/` reverse-proxy, native Range for the 2.9 GB ISO),
`zen-pxe-api` (:3081, `server.py` — the boot decisions). State lives under
`/mnt/user/coefficient/pxe/state/{flags,registry,inventory,seen}`.

## Files

| File | Role |
|---|---|
| `server.py` | control service — safe boot decisions, the console menu, inventory intake, install-done callback |
| `render_install.py` | renders a per-box autoinstall (serial-matched storage + docker/worker/ssh/cred; `CRED=""` ⇒ no-cred console install) |
| `install-defaults.env` | ssh key / password hash / worker image / R2 endpoint — one source of truth for BOTH render paths (dev box + tower) |
| `fleet-pxe` | dev-box CLI: seen / inventory / show / register / install / menu / nomenu / cancel / wol (over SSH to tower) |
| `dnsmasq.conf` | proxy-DHCP + TFTP + iPXE chainload |
| `nginx.conf` | static assets + `/api/` proxy |
| `http/boot.ipxe` | stage-2 iPXE: re-asks `/api/boot/<mac>` |
| `deploy.sh` | push + (re)start on the tower (Unraid has no compose plugin) |
| `docker-compose.yml` | reference topology (usable if you install the compose plugin) |

## Status (verified 2026-07-21 on a real UEFI box)

PXE → GRUB → kernel+initrd → casper → cloud-init → **inventory report + power-off** all confirmed
end-to-end on real hardware (box `04:7c:16:b3:18:51`). The install path (serial-matched autoinstall
+ worker payload) shares the identical GRUB/casper boot and is wired the same way. `dnsmasq.conf`
keeps `log-dhcp` on, so any box's boot is fully traceable in `docker logs zen-pxe-dnsmasq` and
`docker logs zen-pxe-nginx`.

Build the binaries (not committed, >30 KB) with `./deploy.sh --ipxe --grub` (needs
`grub-efi-amd64-bin grub-common` + `build-essential liblzma-dev` on the dev box). WoL uses the
tower's `etherwake -i br0 <mac>`; enable Wake-on-LAN in each box's BIOS.
