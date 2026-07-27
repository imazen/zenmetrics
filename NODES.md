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
| Worker running (Ubuntu) | `ssh zen@<box>` | administer / check the worker (Ubuntu side) |
| Box in Windows, need admin | `ssh zenadmin@<box>` | full admin shell — install/update/fix the box without the console |
| Kids returning / activity resumes | `fleet-pxe to-windows <mac>` | clears the flag + reboots → PXE → **Windows** (hand the box back) |
| Box off | `fleet-pxe wol <mac>` | wake it (into whatever the flag currently selects) |

**Three one-time prerequisites per box** (without all three the switch can't work):

1. **Firmware: network/PXE boot FIRST**, ahead of the Windows disk. This is the linchpin — it makes *every*
   reboot go through the tower so the `worker` flag decides the OS. If the box boots its local Windows disk
   first, it never consults the flag and stays stuck in Windows (the failure we hit repeatedly).
2. **Windows: the SSH accounts** — run `windows-worker/setup-reboot-switch.ps1` **once, elevated**. It
   creates two key-only SSH accounts (both authorized to the dev-box key):
   - **`zenswitch`** — reboot-ONLY (non-admin, forced command `shutdown /r`, no shell). `to-ubuntu`/
     `to-windows` use this to flip the box's OS *from Windows*; a leaked key can only reboot it.
   - **`zenadmin`** — FULL ADMIN (Administrators group, real shell), so the operator can install/update/fix
     the box remotely (`ssh zenadmin@<box>`) without the console. **Security:** this grants full control of
     the box to whoever holds the key — intended, but a real exposure; `Remove-LocalUser zenadmin` revokes
     it while `zenswitch` keeps working.
   Idempotent — skips the OpenSSH install if already present; prefers the Windows-Update-managed OpenSSH
   (auto-patched), hash-pinned standalone fallback if WU is unavailable. Validated on real Windows PS 5.1.
3. **Ubuntu side + tower**: installed & registered via `ubuntu-node/pxe/` (serial-matched to our disk), and
   the node runs **automatic security updates** (`unattended-upgrades`, auto-reboot 04:00) since it's long-
   lived and unattended.

Gotcha we learned the hard way: **Secure Boot must be OFF** on these boxes — the PXE chain (iPXE + grub +
kernel) isn't Secure-Boot-signed, so with it on the box loops at grub without ever booting the kernel.

## The household fleet — roster + how to reach each box

**The MAC is the one stable identifier** (survives OS flips — same NIC MAC in Windows and Ubuntu — and
sleep/DHCP churn). `fleet-pxe` already keys on it. IPs below are the current DHCP leases; **make them
permanent via router DHCP reservations** (see "Addressing" below) — do NOT rely on `.local` mDNS.

| Node | Host / mDNS | Type | Wired MAC | IP | Reach it as |
|---|---|---|---|---|---|
| **jason** | `jason-desktop` | dual-boot kids' PC (i5-13400F 10C/16T; Win + Ubuntu, PXE-first, no-sleep, WoL on) | `04:7c:16:b3:18:51` | 192.168.50.148 | Win: `ssh zenadmin@` (full) / `zenswitch@` (reboot); Ubuntu: `ssh zen@` |
| **ian** | `ian-desktop` | dual-boot kids' PC (Ryzen 5 5600G 6C/12T; 2 TB Win + 1 TB Ubuntu XFS; PXE-first, no-sleep, WoL on) | `04:7c:16:8a:b5:b7` | 192.168.50.193 | same as jason |
| **mac** | `lilith-mac` | macOS worker — Mac mini M4 Pro (idle-only, launchd) | `1c:f6:4c:8b:29:a9` | 192.168.50.224 | `ssh lilith@` — key authorized + NOPASSWD sudo; `SleepDisabled=1` |
| **lianli** | `lilith-lianli` | **always-Ubuntu** dedicated worker (Ubuntu 26.04; dev box + worker enrolled/stopped) | `74:56:3c:b8:45:8d` | 192.168.50.27 | `ssh lilith@` — key + NOPASSWD sudo; full dev toolchain; worker unit stopped (pool drained) |
| **wsl** | (this dev box) | WSL2 — the fleet **operator** (holds CF token, mints creds, runs `fleet-pxe`, builds) | — | WSL NAT (reaches LAN) | local shell |
| **i265** | `i265` (installed 2026-07-27; briefly `ultra7-265k`, was `zen-node-4` during enrollment, Win name `DESKTOP-VMLP4OC`) | always-Ubuntu worker — Intel Core Ultra 7 265K 20C/20T, 32 GB, WD_BLACK SN850X 1TB serial `25286M803378` (single disk; Windows wiped by the serial-matched install), Ubuntu 26.04 | `60:cf:84:76:20:d2` (wired eno1; wifi `c0:bf:be:8e:1e:28`) | 192.168.50.140 | `ssh zen@` (worker enrolled, stopped) |
| **tower** | (basement, Unraid) | PXE/dnsmasq/nginx server + R2 + the `zen924-basement` worker (was `zen720-basement` in the 720 wave) | (Unraid NIC) | 192.168.50.170 | `ssh root@tower` |

### Managing each type

- **Dual-boot kids' PCs (jason, ian)** — see the section above. Kids own them, default Windows; operator
  flips to Ubuntu with `fleet-pxe to-ubuntu <mac>` (idle only) and back with `to-windows`. Both are set:
  **UEFI PXE-boot first** (so the tower's worker flag decides the OS — check with
  `ssh zenadmin@<box> "bcdedit /enum firmware"`; fix with `bcdedit /set {fwbootmgr} displayorder {<pxe-guid>} /addfirst`),
  **sleep+hibernate+FastStartup off** (`powercfg /change standby-timeout-ac 0 & powercfg /h off`), and
  **WoL on** (`Set-NetAdapterPowerManagement -WakeOnMagicPacket Enabled`; wake with `etherwake -i br0 <mac>` on the tower).
  Full remote admin via `ssh zenadmin@<box>`.
  **KNOWN REGRESSION (found 2026-07-27, ian): the no-sleep + WoL config does NOT hold.** ian went to
  Windows sleep and was UNWAKEABLE remotely — no response to raw-ethernet `etherwake` (xN), UDP-broadcast
  `wakeonlan`, or ARP: the NIC has no standby power in that sleep state (likely Modern Standby powering
  the NIC down, or Realtek "Shutdown Wake-On-Lan"/wake-arm driver settings lost — e.g. via a Windows
  update or driver reinstall). **Recovery: physical power button only.** Hardening to apply on next
  wake (both kids' boxes — jason has the same exposure):
  1. Re-assert no-sleep: `powercfg /change standby-timeout-ac 0`, `powercfg /h off`; verify `powercfg /a`
     and check whether the box uses Modern Standby (S0ix) — if so, prefer disabling MS NIC power-down.
  2. Arm the NIC: `powercfg /devicequery wake_armed` must list the ethernet adapter; if not,
     `powercfg /deviceenablewake "<adapter>"` + `Set-NetAdapterPowerManagement -WakeOnMagicPacket Enabled`
     + Realtek advanced properties: "Shutdown Wake-On-Lan"=Enabled, "Energy Efficient Ethernet"=Off.
  3. VERIFY by round-trip: sleep the box, `etherwake` from tower, confirm it wakes. Config without the
     round-trip test is what regressed silently this time.
  4. Ubuntu side has the parallel gap (never armed): one-shot systemd unit `ethtool -s <nic> wol g`
     at boot + before suspend, and BIOS ErP/deep-sleep OFF so the NIC keeps S5 standby power.
- **mac (lilith-mac)** — always macOS (Mac mini M4 Pro `Mac16,11`, wired Ethernet en0, AC power). Reachable
  as `ssh lilith@` (dev-box key authorized) with **passwordless sudo** (`/etc/sudoers.d/90-lilith-nopasswd`
  → `sudo -n` works). **"Keeps going unreachable" was macOS idle-sleep — FIXED 2026-07-24**:
  `sudo pmset -a disablesleep 1 womp 1` → `SleepDisabled=1` (never sleeps) + wake-on-magic-packet. The
  worker runs idle-only under launchd (backs off if used in the last 10 min); have it hold `caffeinate`
  while running as a further guard. **Full Rust + C dev toolchain provisioned 2026-07-24** via
  [`mac-worker/setup-mac-dev.sh`](mac-worker/setup-mac-dev.sh) (150 brew formulae — LLVM/gcc/cmake/ninja +
  the codec libs — Rust stable+nightly with 7 cross targets, ~24 cargo tools; verified end-to-end).
- **lianli (lilith-lianli)** — always Ubuntu 26.04, dedicated worker; no OS flip. Reachable as `ssh lilith@`
  (dev-box key authorized; passwordless sudo). **Provisioned 2026-07-25** two ways: (1) full Rust + C
  **dev toolchain** via [`ubuntu-node/setup-node-dev.sh`](ubuntu-node/setup-node-dev.sh) — apt toolchain +
  codec `-dev` libs + valgrind/gdb/heaptrack + Rust stable/nightly + ~24 cargo tools + our
  cargo-read/copter/superwork + Claude Code + herdr; and (2) **worker-enrolled but STOPPED** via
  [`ubuntu-node/enroll_running_node.sh`](ubuntu-node/enroll_running_node.sh) — Docker + `zen-worker` unit +
  `:exec` image cached, left disabled so it can't restart-loop on the drained pool. **Activate the worker
  when a sweep queues:** `bash ubuntu-node/enroll_running_node.sh --start 192.168.50.27` (mints a 7-day
  cred + `systemctl enable --now`). A server, so it doesn't idle-sleep.
- **i265 (60:cf:84:76:20:d2)** — enrolled + installed 2026-07-27 (renamed from zen-node-4 to match its CPU per user; PXE registry updated) (seen → inventory → register serial
  `25286M803378` → serial-matched install). **KNOWN ISSUE: eno1 link-flaps.** Symptoms observed during
  enrollment: grub netboot phone-home succeeded on only ~1/3 of PXE launches (grubnet.efi always
  downloaded; the grub-stage DHCP/HTTP intermittently died), the autoinstall's first network call
  (install-flag clear) failed, DHCP re-solicits mid-install, and the installer console spammed
  network-change events ("send update change eno1"). WoL from S5 DID wake it (~80 s to firmware).
  Hardening once Ubuntu is up: `ethtool --set-eee eno1 eee off` (persist via systemd oneshot),
  check/replace cable + switch port, verify with a reboot loop (3/3 PXE phone-homes). Until fixed,
  expect netboot retries to need a power-cycle or two.
- **wsl (dev box)** — the operator, not a worker: holds the Cloudflare token, mints scoped 7-day R2 creds,
  runs `fleet-pxe` + the reboot/OS-flip loop, builds the binaries. WSL2 NAT breaks inbound UDP (TFTP) but
  reaches the LAN fine over TCP/SSH.
- **tower** — PXE server (dnsmasq proxy-DHCP + nginx + the serial-matched autoinstall), R2 storage, and a
  CPU worker. It also sends the WoL magic packets (`etherwake -i br0 <mac>`). **Work inside Docker ONLY on
  the tower** — Unraid's host is stateless (RAM-booted), so run all compute/tooling in containers (the
  `zen*` workers, the `zen-pxe-*` stack, or a throwaway `docker run --rm`), never install on the host;
  read-only host commands are fine. (Global `~/.claude/CLAUDE.md` "Tower … compute node" rule 7.)

### Remote terminals — mosh + tmux + herdr (rolled out 2026-07-26)

Every interactive box takes roaming `mosh` sessions (UDP 60000-61000; no firewall in the way on any of
them today — lianli ufw inactive, mac ALF disabled, tower LAN bridge open) with `tmux` for persistence,
plus [herdr](https://herdr.dev) (`curl -fsSL https://herdr.dev/install.sh | sh` → `~/.local/bin`, no root)
for agent workspaces:

| Box | mosh-server | tmux | herdr | Connect |
|---|---|---|---|---|
| **lianli** | 1.4.0 (apt) | apt | `~/.local/bin` | `mosh lilith@192.168.50.27` |
| **mac** | 1.4.0 (brew; non-interactive PATH via `~/.zshenv`) | brew | `~/.local/bin` | `mosh lilith@192.168.50.224` |
| **tower** | `zen-mosh` gateway container — host untouched | in-container; auto-attach `main`, panes are HOST root shells | in-container | `mosh --ssh='ssh -p 2223' root@192.168.50.170` |
| **jason** (Ubuntu side) | 1.4.0 (apt) | node base image | `~/.local/bin` | `mosh zen@192.168.50.148` (Ubuntu side only) |
| **wsl** (dev box) | installed, but WSL2 NAT blocks inbound UDP — mosh INTO wsl needs mirrored networking (`networkingMode=mirrored` in `.wslconfig`); outbound client works fine | ✓ | ✓ | n/a inbound |

Tower gateway details: image `zen-mosh:v1` built from [`tower/zen-mosh/`](tower/zen-mosh/README.md)
(array mirror `/mnt/user/coefficient/tools/zen-mosh/`); container runs `--network host --pid host
--privileged -v /root/.ssh:/rootkeys:ro --restart unless-stopped`. One-off `ssh -p 2223 root@tower '<cmd>'`
also executes on the HOST (nsenter); host sshd on 22 is untouched. Dual-boot gotcha: jason/ian present
DIFFERENT host keys per OS on the same IP — a "REMOTE HOST IDENTIFICATION HAS CHANGED" on .148/.193 is
almost always the OS flip, not an attack; `ssh-keygen -R <ip>` and reconnect. Do the `-R` BEFORE any
post-flip ssh poll — `accept-new` silently REJECTS a *changed* key, so a poll loop reads as "box never
came up" while sshd is answering fine (cost ~10 min on jason 2026-07-26). Related: if `fleet-pxe
to-ubuntu` prints "couldn't reach … as zen or zenswitch" while the box is in Windows (seen 2026-07-26:
zenswitch ssh refused while zenadmin worked — unresolved), the worker flag IS already set; complete the
flip with `ssh zenadmin@<box> 'shutdown /r /t 5'`.

### Addressing — fixed IPs + DNS, NOT mDNS (recommendation)

`.local` mDNS is unreliable across this mixed fleet: Windows boxes announce inconsistently, the mac drops
its announcement on sleep, and the dual-boot boxes change identity (Windows vs Ubuntu). We hit `.local`
resolution failures repeatedly. So:

1. **Set DHCP reservations on the router** (MAC → the current IP above) for all six boxes. Reservations are
   keyed on the MAC, so a dual-boot box keeps the same IP in *both* OSes, and it survives sleep/reboots.
   Keep the current leases as the reservations so nothing has to change.
2. **Then add DNS** (optional but nice): the tower already runs dnsmasq — give it `host-record` / `address=`
   entries (`<name> → reserved IP`) and point the dev box's resolver (or `/etc/hosts`) at them, so
   `jason`/`ian`/`lilith-mac`/`lilith-lianli` resolve without mDNS.
3. **Until reservations exist**, key tooling on the **MAC** and resolve to IP via the tower's ARP/mDNS as a
   fallback (what `fleet-pxe to-ubuntu <mac>` already does).
