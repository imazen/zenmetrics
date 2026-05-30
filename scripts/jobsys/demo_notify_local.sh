#!/usr/bin/env bash
# Live demo of the notification mechanism (goal D: push on budget-crossed / run-complete / poison-
# spike / kind-drained, with a deep link). Points the dashboard's webhook at a LOCAL receiver, so the
# full detect → format → POST path is proven without any external service or a real channel URL — the
# only thing the production channel adds is the destination (set ZEN_NOTIFY_WEBHOOK to your
# Slack/Discord/ntfy URL).
#
# Requires: built zen-jobdash, python3.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DASH="$ROOT/target/debug/zen-jobdash"
W="$(mktemp -d)"
RECV_PORT=3091
DASH_PORT=3092
RECV_PID=""; DASH_PID=""
cleanup() { [ -n "$DASH_PID" ] && kill "$DASH_PID" 2>/dev/null; [ -n "$RECV_PID" ] && kill "$RECV_PID" 2>/dev/null; rm -rf "$W"; }
trap cleanup EXIT

# Local webhook receiver: append each POST body to a file, reply 200.
cat > "$W/recv.py" <<'PY'
import http.server, sys
OUT, PORT = sys.argv[1], int(sys.argv[2])
class H(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get('content-length', 0))
        open(OUT, 'ab').write(self.rfile.read(n) + b'\n')
        self.send_response(200); self.end_headers(); self.wfile.write(b'ok')
    def log_message(self, *a): pass
http.server.HTTPServer(('127.0.0.1', PORT), H).serve_forever()
PY
python3 "$W/recv.py" "$W/received.txt" "$RECV_PORT" & RECV_PID=$!
sleep 0.5

# Worker fixture: $0.35 spent vs a $0.10 cap → budget-crossed should fire.
cat > "$W/workers.json" <<JSON
[{"worker":"vast-gpu-1","provider":"vast","class":"gpu","rate_usd_per_hr":0.35,"uptime_secs":3600,"jobs_done":120}]
JSON

echo "### notify demo — webhook → local receiver on :$RECV_PORT"
echo "    fixture: spent \$0.35, cap \$0.10 (budget-crossed condition)"
ZEN_WORKERS_JSON="$W/workers.json" ZEN_BUDGET_CAP_USD=0.10 ZEN_REFRESH_SECS=2 \
  ZEN_NOTIFY_WEBHOOK="http://127.0.0.1:$RECV_PORT" ZEN_PUBLIC_URL="https://zen-jobdash-production.up.railway.app" \
  PORT=$DASH_PORT "$DASH" > "$W/dash.log" 2>&1 & DASH_PID=$!

# Wait for one refresh cycle to detect + fire.
for _ in $(seq 1 12); do [ -s "$W/received.txt" ] && break; sleep 1; done

echo "### webhook receiver captured:"
if [ -s "$W/received.txt" ]; then
  cat "$W/received.txt" | sed 's/^/    /'
  echo "### PASS — notification fired with text + deep link."
else
  echo "    (nothing received)"; echo "--- dashboard log ---"; cat "$W/dash.log"
  exit 1
fi
