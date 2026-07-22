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

# worker.env: the 3 AWS_* cred lines from mint_cred.sh + the fixed worker config.
worker_env = E["CRED"].rstrip("\n") + "\n" + textwrap.dedent(f"""\
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
      - {{type: format,    id: froot, volume: proot, fstype: ext4, preserve: false}}
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
  packages: [docker.io, avahi-daemon, libnss-mdns, jq, curl, chrony]
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
    runcmd:
      - [ usermod, -aG, docker, zen ]
      - [ systemctl, daemon-reload ]
      - [ systemctl, enable, --now, docker ]
      - [ systemctl, enable, --now, zen-worker ]
""")
