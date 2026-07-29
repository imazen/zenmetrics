#!/usr/bin/env python3
"""Render a per-box autoinstall user-data (the destructive install path).

Reads config from the environment (set by fleet-pxe register) and prints the
autoinstall YAML. The storage section matches the OS disk by EXACT SERIAL — if that
serial isn't present at install time, subiquity errors out instead of guessing a
different disk. That is the "never wipe the wrong disk" guarantee, in config.
"""
import os, textwrap

E = os.environ
SERIAL, HOST, MAC = E["SERIAL"], E["HOST"], E["MAC"]
R2EP, KEY, HASH, IMG, TOWER = E["R2EP"], E["KEY"], E["HASH"], E["IMG"], E["TOWER_IP"]

# CRED empty => "no-cred install" (the tower's interactive menu path: only the dev box holds the
# Cloudflare token, so a console-initiated install can't mint a 7-day R2 cred). The box still comes
# up as a complete, ssh-able node with the full toolkit — it just can't claim jobs yet, so we leave
# zen-worker INSTALLED BUT DISABLED (enabling it without creds would restart-loop forever) and drop
# a NEEDS-CRED note. Finish it from the dev box with:
#     bash ubuntu-node/enroll_running_node.sh --start <ip>
CRED = E.get("CRED", "").strip()
HAS_CRED = bool(CRED)

# worker.env: the 3 AWS_* cred lines from mint_cred.sh + the fixed worker config.
worker_env = ((CRED.rstrip("\n") + "\n") if HAS_CRED else "") + textwrap.dedent(f"""\
    AWS_REGION=auto
    ZEN_R2_ENDPOINT={R2EP}
    ZEN_BUCKET=zentrain
    ZEN_POOL_RUNLIST=s3://zentrain/jobs/_pool/runlist.tsv
    ZEN_CORPUS_PREFIX=refs/clean-picker-corpus-2026-06-26
    ZEN_MAX_MIN=700
    ZEN_CORE_OVERSUBSCRIBE=3
    ZEN_PERSISTENT_EXEC=1
    RAYON_NUM_THREADS=1
    OMP_NUM_THREADS=1
    ZEN_CHUNK_WALL_SEC=20
    ZEN_PASS_TIMEOUT=5400
    ZEN_PROVIDER=basement
    ZEN_WORKER={HOST}
    """)

worker_unit = textwrap.dedent(f"""\
    [Unit]
    Description=zen backfill worker (zensim-720)
    Wants=network-online.target docker.service
    After=network-online.target docker.service
    [Service]
    ExecStartPre=-/usr/bin/docker rm -f zen720
    ExecStart=/usr/bin/docker run --rm --name zen720 --env-file /etc/zen-node/worker.env \\
      --entrypoint /usr/local/bin/fleet-entrypoint.sh {IMG}
    ExecStop=/usr/bin/docker rm -f zen720
    Restart=always
    RestartSec=10
    [Install]
    WantedBy=multi-user.target
    """)

def indent(s, n):
    pad = " " * n
    return "\n".join(pad + l for l in s.rstrip("\n").split("\n"))

# Enabling a credential-less worker would restart-loop, so it stays disabled until enrolled.
worker_enable = "      - [ systemctl, enable, --now, zen-worker ]\n" if HAS_CRED else ""

# NOTE: build this with indent(), NOT a dedent()ed literal — dedent strips the common leading
# whitespace, which drops the block to column 0 and produces invalid YAML.
_needs_cred_body = textwrap.dedent(f"""\
    This node was installed from the tower's interactive PXE menu, which cannot mint an R2
    credential (only the dev box holds the Cloudflare token). worker.env has NO AWS_* lines
    and zen-worker is installed but DISABLED (enabling it uncredentialed would restart-loop).
    Finish enrollment from the dev box:
        bash ubuntu-node/enroll_running_node.sh --start <this box's ip>
    host={HOST} mac={MAC} disk-serial={SERIAL}
    """)
needs_cred_file = "" if HAS_CRED else (
    "      - path: /etc/zen-node/NEEDS-CRED\n"
    "        content: |\n" + indent(_needs_cred_body, 10) + "\n")

# Automatic SECURITY updates (unattended-upgrades). Ubuntu's default 50unattended-upgrades already allows
# the -security pocket; these turn on the periodic run and auto-reboot for kernel patches at 04:00 — a
# reboot is safe on a worker (in-flight cells requeue via the ledger). SECURITY only, not release upgrades.
apt_auto = (
    'APT::Periodic::Update-Package-Lists "1";\n'
    'APT::Periodic::Unattended-Upgrade "1";\n'
    'APT::Periodic::Download-Upgradeable-Packages "1";\n'
    'APT::Periodic::AutocleanInterval "7";\n'
    'Unattended-Upgrade::Automatic-Reboot "true";\n'
    'Unattended-Upgrade::Automatic-Reboot-Time "04:00";\n'
)

print(f"""#cloud-config
# zen-pxe install for {HOST} ({MAC}) -> disk serial {SERIAL}
autoinstall:
  version: 1
  refresh-installer: {{update: no}}
  locale: en_US.UTF-8
  keyboard: {{layout: us}}
  storage:
    config:
      - {{type: disk, id: disk0, match: {{serial: "*{SERIAL}*"}}, ptable: gpt, wipe: superblock-recursive, preserve: false, grub_device: false}}
      - {{type: partition, id: pbios, device: disk0, size: 1M, flag: bios_grub, preserve: false}}
      - {{type: partition, id: pefi,  device: disk0, size: 1G, flag: boot, grub_device: true, preserve: false}}
      - {{type: format,    id: fefi,  volume: pefi, fstype: fat32, preserve: false}}
      - {{type: partition, id: proot, device: disk0, size: -1, preserve: false}}
      - {{type: format,    id: froot, volume: proot, fstype: xfs, preserve: false}}
      - {{type: mount,     id: mroot, device: froot, path: /}}
      - {{type: mount,     id: mefi,  device: fefi, path: /boot/efi}}
  identity:
    hostname: {HOST}
    username: zen
    password: "{HASH}"
  ssh:
    install-server: true
    allow-pw: false
    authorized-keys:
      - "{KEY}"
  packages: [docker.io, avahi-daemon, libnss-mdns, jq, curl, chrony, unattended-upgrades,
             mosh, tmux, build-essential, pkg-config]
  late-commands:
    # install-once (guard 1): clear our own flag NOW (installer has network) so a re-PXE can't
    # re-wipe. Retry hard — a transient blip here must not leave the flag set. (Guard 2 is the
    # server-side 'installed' marker, which refuses re-install even if this never lands.)
    - "sh -c 'for i in 1 2 3 4 5 6 7 8; do curl -fsS -X POST http://{TOWER}:3080/api/done/{MAC} && break; sleep 3; done; true'"
  user-data:
    disable_root: true
    write_files:
      - path: /etc/zen-node/worker.env
        permissions: "0600"
        content: |
{indent(worker_env, 10)}
      - path: /etc/systemd/system/zen-worker.service
        content: |
{indent(worker_unit, 10)}
      - path: /etc/sudoers.d/90-zen-nopasswd
        permissions: "0440"
        content: "zen ALL=(ALL) NOPASSWD:ALL\\n"
      - path: /etc/modprobe.d/blacklist-nouveau.conf
        content: "blacklist nouveau\\noptions nouveau modeset=0\\n"
      - path: /etc/apt/apt.conf.d/52zen-auto-security
        content: |
{indent(apt_auto, 10)}
{needs_cred_file}
      # PATH for NON-interactive ssh too (pam_env reads this on every login, unlike
      # .profile/.bashrc which bash skips for `ssh node <cmd>`). Without it,
      # `ssh zen@node herdr ...` can't find the ~/.local/bin tools below.
      - path: /etc/environment
        content: "PATH=\\"/home/zen/.local/bin:/home/zen/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin\\"\\n"
    runcmd:
      - [ usermod, -aG, docker, zen ]
      - [ systemctl, daemon-reload ]
      - [ systemctl, enable, --now, docker ]
{worker_enable}      - [ systemctl, enable, --now, unattended-upgrades ]
      - [ update-initramfs, -u ]
      # Operator tooling, installed as `zen` (they are per-user: ~/.local/bin, ~/.cargo).
      # Retried 3x and ALWAYS exit 0 — a network blip must never fail the install; the box is a
      # working worker without them, and setup-node-dev.sh can add them (plus the full toolchain)
      # later. mosh/tmux come from `packages:` above (apt, at install time).
      - [ su, -l, zen, -c, "for i in 1 2 3; do curl -fsSL https://herdr.dev/install.sh | sh && break; sleep 5; done; true" ]
      - [ su, -l, zen, -c, "for i in 1 2 3; do curl -fsSL https://claude.ai/install.sh | bash && break; sleep 5; done; true" ]
      - [ su, -l, zen, -c, "for i in 1 2 3; do curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile default --default-toolchain stable && break; sleep 5; done; true" ]
""")
