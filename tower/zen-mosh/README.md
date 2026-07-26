# zen-mosh — tower terminal gateway (mosh + tmux + herdr), Docker-only

Unraid's host OS is stateless (RAM-booted) and per workspace rules nothing may be
installed on it — but we still want roaming `mosh` sessions and persistent `tmux`
on the tower. This container provides both without touching the host: sshd on
port 2223 bootstraps mosh; mosh-server / tmux / herdr live in the image;
interactive logins auto-attach tmux session `main`, and every pane nsenters into
the host's namespaces — real host root shells, persisted across disconnects.

## Connect

    mosh --ssh='ssh -p 2223' root@tower       # roaming; auto-attaches tmux 'main'
    ssh -p 2223 root@tower                    # same session over plain ssh
    ssh -p 2223 root@tower 'df -h /mnt/user'  # one-off commands ALSO run on the HOST

Host sshd on port 22 is untouched and remains the path for scripted admin.

## Build + run (on the tower)

    cd /mnt/user/coefficient/tools/zen-mosh   # array mirror of this dir
    docker build -t zen-mosh:v1 .
    docker rm -f zen-mosh 2>/dev/null
    docker run -d --init --name zen-mosh --network host --pid host --privileged \
      -v /root/.ssh:/rootkeys:ro --restart unless-stopped zen-mosh:v1

## Design notes / gotchas (learned 2026-07-26)

- root's login shell is `tmux-host`: with no args it runs
  `tmux new-session -A -s main`; with `-c <cmd>` it nsenters and execs on the
  HOST — EXCEPT commands starting with `mosh-server`, which must exec in the
  CONTAINER (that's the mosh bootstrap; the session shell mosh-server then
  spawns is tmux-host again, so interactive mosh still lands in tmux).
- `/etc/tmux.conf` pins `default-shell /bin/bash`: tmux spawns panes via
  `<default-shell> -c <default-command>`, and root's shell (tmux-host) would
  nsenter to the host BEFORE resolving hostshell (a container-only path) —
  pane dies instantly with "No such file or directory".
- Key auth: the host's `/root/.ssh` is bind-mounted read-only; sshd reads
  `/rootkeys/authorized_keys` (`StrictModes no` because of the mount).
- `--init` so detached mosh-server processes get reaped; `--restart
  unless-stopped` so the gateway survives array/docker restarts and reboots.
- mosh's UDP 60000-61000 binds directly on the host network; there is no
  firewall on the LAN bridge.
- Verify after changes: `ssh -p 2223 root@tower hostname` must print `Tower`
  (host, not container); `docker exec zen-mosh tmux capture-pane -p -t main`
  shows a `root@Tower:~#` prompt; a full client probe is
  `mosh --ssh='ssh -p 2223' root@tower -- true`.
