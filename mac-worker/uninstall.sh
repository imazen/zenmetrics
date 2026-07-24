#!/usr/bin/env bash
# uninstall.sh -- stop + remove the Zen macOS worker LaunchDaemon. `sudo bash uninstall.sh`.
set -euo pipefail
LABEL="com.imazen.zenworker"
DEST="/Library/LaunchDaemons/$LABEL.plist"
[ "$(id -u)" -eq 0 ] || { echo "run with sudo"; exit 1; }
launchctl bootout system "$DEST" 2>/dev/null || true
rm -f "$DEST"
echo "removed $LABEL (payload dir + logs left in place)"
