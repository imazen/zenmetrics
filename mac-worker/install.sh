#!/usr/bin/env bash
# install.sh -- register the Zen macOS worker as a launchd LaunchDaemon (runs at boot, as root,
# survives logout). Run from inside the payload dir: `sudo bash install.sh`.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LABEL="com.imazen.zenworker"
DEST="/Library/LaunchDaemons/$LABEL.plist"

[ "$(id -u)" -eq 0 ] || { echo "run with sudo (LaunchDaemons need root)"; exit 1; }
for f in run.sh worker.env zenfleet-worker zenmetrics s5cmd; do
  [ -e "$HERE/$f" ] || { echo "FATAL: missing $HERE/$f -- assemble the payload first (see README)"; exit 2; }
done
chmod +x "$HERE/run.sh" "$HERE/zenfleet-worker" "$HERE/zenmetrics" "$HERE/s5cmd" 2>/dev/null || true

# template the payload path into the plist and install it
sed "s#__PAYLOAD_DIR__#$HERE#g" "$HERE/com.imazen.zenworker.plist" > "$DEST"
chown root:wheel "$DEST"; chmod 644 "$DEST"

# (re)load
launchctl bootout system "$DEST" 2>/dev/null || true
launchctl bootstrap system "$DEST"
launchctl enable "system/$LABEL"
echo "installed + started $LABEL"
echo "  status: sudo launchctl print system/$LABEL | grep -E 'state|pid'"
echo "  logs:   tail -f $HERE/worker.log"
