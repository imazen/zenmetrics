# Adding a compute node to the fleet

Every node — Linux, Windows, or Mac — becomes another `zenfleet-worker` claiming jobs off the **same**
R2 pool (`s3://zentrain/jobs/_pool/runlist.tsv`) and writing blobs + ledger shards back. They coordinate
purely through the R2 claim ledger, so any mix of node types converges. All are **CPU** feature/score
workers (GPU is a separate concern). Every node gets a **scoped, 7-day R2 temp cred** minted on the dev
box (`ubuntu-node/mint_cred.sh`) — never the account root key.

Pick the guide by node type:

| Node type | Guide | How it runs / stays up |
|---|---|---|
| **Linux (many boxes, remote install)** | [`ubuntu-node/pxe/`](ubuntu-node/pxe/README.md) | PXE network-install Ubuntu onto each box's own disk from the tower (dnsmasq proxy-DHCP + serial-matched autoinstall, 4-layer wipe safety); runs `zen-worker` as a systemd service. **Verified on real hardware 2026-07-21.** |
| **Linux (one box, build a drive)** | [`ubuntu-node/`](ubuntu-node/README.md) | `build_node_drive.sh` bakes a whole bootable Ubuntu drive (Docker + worker + SSH key + cred); move it to any LAN box and boot. |
| **Windows (idle desktops / kids' PCs)** | [`windows-worker/`](windows-worker/README.md) | native `.exe`s under a SYSTEM scheduled task; **idle-only**, survives logout/reboot. |
| **macOS (Mac mini/Studio/desktop)** | [`mac-worker/`](mac-worker/README.md) | native `darwin` binaries under a launchd LaunchDaemon (survives logout), or colima + the `:exec` image. |

Common to all: after the node is up, refresh its 7-day cred weekly (Linux: `ubuntu-node/onboard_node.sh`;
Windows/Mac: re-mint into `worker.env` and restart the worker). Watch the fleet with
`scripts/jobsys/fleet watch <run>` — it flags idle/failed/underutilized boxes (`zenfleet-core::idle`
thresholds) and tears down paid idle ones.

## Dual-boot kids' machines — shared, KIDS HAVE PRIORITY (our 6 household nodes)

Several of our 6 nodes are the household **kids' dual-boot PCs**: Windows on one disk (theirs), our Ubuntu
`zen-worker` installed on a *separate* disk/partition (ours, serial-locked so an install never touches the
Windows disk). **The kids own these machines.** They power them on into Windows, use them, and shut them
down whenever they want — **frequently interrupting our worker mid-run.** That is normal, not a failure:
the fleet is resumable (an interrupted cell just requeues via the R2 ledger and is re-scored next window),
so a kid yanking a box costs us nothing but the in-flight cell.

**Default OS = Windows, always.** The tower's per-box `worker` flag is kept **CLEARED** by default, so any
boot a kid does lands in **Windows** — they are never blocked by our work. We only flip a box to Ubuntu
*opportunistically*, when it's idle/logged-out/off, and we flip it back before they need it.

**Operator (whoever runs the fleet) handles reboot / admin / switch-back — the kids never touch our side:**

| When | Do | What happens |
|---|---|---|
| Box idle/off (kids done, logged out) | `fleet-pxe to-ubuntu <mac>` | sets `worker` flag + reboots the box (from Windows via the `zenswitch` switch, from Ubuntu via `zen@`, or WoL if off) → PXE → **Ubuntu worker** |
| Worker running | `ssh zen@<box>` | administer / check the worker (Ubuntu side) |
| Kids returning / activity resumes | `fleet-pxe to-windows <mac>` | clears the flag + reboots → PXE → **Windows** (hand the box back) |
| Box off | `fleet-pxe wol <mac>` | wake it (into whatever the flag currently selects) |

**Three one-time prerequisites per box** (without all three the switch can't work):

1. **Firmware: network/PXE boot FIRST**, ahead of the Windows disk. This is the linchpin — it makes *every*
   reboot go through the tower so the `worker` flag decides the OS. If the box boots its local Windows disk
   first, it never consults the flag and stays stuck in Windows (the failure we hit repeatedly).
2. **Windows: the reboot-switch** — run `windows-worker/setup-reboot-switch.ps1` **once, elevated**. It
   installs a **reboot-only, non-admin** `zenswitch` SSH (forced command = `shutdown /r`, no shell), so
   `to-ubuntu`/`to-windows` can reboot the box *from Windows*. A leaked key can only reboot; the tower (ours)
   decides the OS. Idempotent — skips the OpenSSH install if it's already there. Prefers the Windows-Update-
   managed OpenSSH (auto-patched) with a hash-pinned standalone fallback if WU is unavailable.
3. **Ubuntu side + tower**: installed & registered via `ubuntu-node/pxe/` (serial-matched to our disk), and
   the node runs **automatic security updates** (`unattended-upgrades`, auto-reboot 04:00) since it's long-
   lived and unattended.

**Roster** (keep current — MAC is the stable key; IPs float on DHCP):

| Node | Wired MAC | Windows / Ubuntu disks | State |
|---|---|---|---|
| zen-node-2 | `04:7c:16:b3:18:51` | dual-boot | installed + registered; reboot-switch pending (`setup-reboot-switch.ps1` re-run) |
| zen-node-3 | `04:7c:16:8a:b5:b7` | 2 TB Win + 1 TB Ubuntu (XFS) | installed + registered; reboot-switch pending |
| _(fill remaining kids' PCs)_ | | | |

Gotcha we learned the hard way: **Secure Boot must be OFF** on these boxes — the PXE chain (iPXE + grub +
kernel) isn't Secure-Boot-signed, so with it on the box loops at grub without ever booting the kernel.
