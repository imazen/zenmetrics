# zen compute node — bootable Ubuntu drive

Turn a spare machine into a hands-off backfill worker: boot the USB once, and the machine installs
Ubuntu, joins your Tailscale tailnet (so I can reach it from anywhere), authorizes my SSH key, and
pre-pulls the worker image. Then one command from the dev box starts it working the R2 pool.

## What's here

- `user-data` — the Ubuntu autoinstall seed (SSH key, Docker, Tailscale, worker image). **Edit two things first.**
- `meta-data` — cloud-init instance id (bump per node).
- `onboard_node.sh` — dev-box command that starts the worker on a booted node (fresh 12h R2 cred).

## Before you write the USB — fill in 2 placeholders in `user-data`

1. **`TS_AUTHKEY_PLACEHOLDER`** → a Tailscale **pre-auth key**. Generate one at
   https://login.tailscale.com/admin/settings/keys (Reusable, ~90-day expiry, tag `tag:zen` if you use
   ACLs). This is what makes the node reachable from anywhere without touching your router.
2. **`hostname: zen-node-1`** (and `meta-data`'s `zen-node-1`) → bump to `zen-node-2` for the second
   machine so they don't collide in the tailnet.

(The console password is `zencompute`; SSH is key-only, so you only need it for a physical console.)

## Make the USB (CIDATA method — no ISO repacking)

1. Download **Ubuntu Server 24.04.x LTS** ISO.
2. Write the ISO to **USB #1** (the installer) with Rufus / balenaEtcher / `dd`.
3. On a **second FAT32 volume labeled `CIDATA`** (a small USB #2, or a spare partition), copy this
   folder's `user-data` and `meta-data` to its root.
4. Boot the machine from USB #1. At the GRUB menu press `e`, append `autoinstall` to the `linux` line,
   Ctrl-X to boot. The installer finds the `CIDATA` seed and installs **fully unattended** (it wipes the
   target disk — dedicated node).

> Prefer a single USB? `sudo bash make_seed_iso.sh` (add later) can bake the seed into a copy of the ISO
> so there's one drive to `dd`. The two-volume method above needs no extra tooling.

## After it boots

The node installs, reboots, joins your tailnet, and pulls the image (~1–2 min after first boot). Then:

```bash
tailscale status | grep zen-node          # find its tailnet IP
bash ubuntu-node/onboard_node.sh zen-node-1   # start the worker (or use the tailnet IP)
```

It immediately starts claiming + scoring the backfill remainder from the same R2 pool as the tower and
the cloud fleet. Dedicated box, so it uses all cores. Re-run `onboard_node.sh` within 12 h to refresh the
cred (temp creds cap at 12 h), or add a cron line like the tower's:
`13 4,15 * * * bash .../onboard_node.sh zen-node-1`.

## Notes

- Creds are **never baked onto the drive** — `onboard_node.sh` mints a fresh scoped 12 h cred from the
  dev box each time, so a lost/stolen USB exposes nothing but your SSH key + tailnet auth.
- Once a node is in the tailnet I can reach it from anywhere; no port-forwarding or LAN specifics needed.
- Verify work: `ssh zen@zen-node-1 'sudo docker top zen720 | grep -c zenmetrics'` (should be ~#cores).
