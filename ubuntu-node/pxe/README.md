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
| `server.py` | control service — safe boot decisions, inventory intake, install-done callback |
| `render_install.py` | renders a per-box autoinstall (serial-matched storage + docker/worker/ssh/cred) |
| `fleet-pxe` | dev-box CLI: seen / inventory / show / register / install / cancel / wol (over SSH to tower) |
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
