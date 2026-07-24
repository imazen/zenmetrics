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
