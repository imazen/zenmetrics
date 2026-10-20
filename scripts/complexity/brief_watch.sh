#!/usr/bin/env bash
# Keep the deployed brief current: every BRIEF_INTERVAL seconds (default 300) fetch master and the
# private repo's refs, rebuild, and upload to Cloudflare Pages only when the page changed.
#
#   scripts/complexity/brief_watch.sh WORKSPACE_DIR
#
# WORKSPACE_DIR is a jj workspace of this repository used only by the watcher (created on first run
# with `jj workspace add`); the watcher moves it to master@origin each cycle. The private repo is
# refreshed with `jj --ignore-working-copy git fetch`, which never touches its working copy.
# Settings as in brief.sh, plus BRIEF_PROJECT (default metric-brief) and BRIEF_INTERVAL. Each cycle also
# reads the results sync's status.json over ssh (BRIEF_SYNC_HOST, default i265; BRIEF_SYNC_PATH) so a
# blocked or stopped sync shows on the page; set BRIEF_SYNC_HOST= to skip.
set -uo pipefail
SELF=$(cd "$(dirname "$0")/../.." && pwd)
WS=${1:?usage: brief_watch.sh WORKSPACE_DIR}
PROJECT=${BRIEF_PROJECT:-metric-brief}
INTERVAL=${BRIEF_INTERVAL:-300}
PRIVATE=${BRIEF_PRIVATE:-$HOME/work/zen/zenmetrics-private}
export BRIEF_OUT=${BRIEF_OUT:-$HOME/tmp/metric-brief-watch/out} BRIEF_PRIVATE=$PRIVATE
export CLOUDFLARE_API_TOKEN=${CLOUDFLARE_API_TOKEN:-$(cat "$HOME/.config/cloudflare/zenmetrics-pages-token")}
export PATH=$HOME/.local/opt/node/bin:$PATH
SYNC_HOST=${BRIEF_SYNC_HOST-i265}
SYNC_PATH=${BRIEF_SYNC_PATH:-tmp/iqa-results-five-minute-sync/status.json}
mkdir -p "$BRIEF_OUT"
[ -d "$WS" ] || (cd "$SELF" && jj workspace add "$WS" -r master@origin)
last=$(cat "$BRIEF_OUT/.deployed.sha256" 2>/dev/null || true)
while true; do
  ts=$(date -u +%FT%TZ)
  (cd "$WS" && jj --quiet git fetch && jj --quiet new master@origin) || echo "$ts fetch failed (public); building the previous revision"
  jj -R "$PRIVATE" --quiet --ignore-working-copy git fetch || echo "$ts fetch failed (private)"
  if [ -n "$SYNC_HOST" ]; then
    export BRIEF_SYNC_STATUS=$BRIEF_OUT.sync-status.json
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$SYNC_HOST" cat "$SYNC_PATH" > "$BRIEF_SYNC_STATUS.new" 2>/dev/null \
      && python3 -c "import json,sys;json.load(open(sys.argv[1]))" "$BRIEF_SYNC_STATUS.new" \
      || { echo "$ts sync status unreadable on $SYNC_HOST"; echo '{"state": "unreachable"}' > "$BRIEF_SYNC_STATUS.new"; }
    mv "$BRIEF_SYNC_STATUS.new" "$BRIEF_SYNC_STATUS"
    grep -q '"blocked": true' "$BRIEF_SYNC_STATUS" && echo "$ts RESULTS SYNC BLOCKED: $(python3 -c "import json,sys;d=json.load(open(sys.argv[1]));print(d.get('stalled_since'),d.get('reason'))" "$BRIEF_SYNC_STATUS")"
  fi
  rm -rf "$BRIEF_OUT/site"
  if (cd "$WS" && scripts/complexity/brief.sh > "$BRIEF_OUT.log" 2>&1); then
    now=$(grep -v 'data-build-time' "$BRIEF_OUT/site/index.html" | sha256sum | cut -d' ' -f1)  # the build-time line alone is not a change
    if [ "$now" != "$last" ]; then
      rev=$(python3 -c "import json;print(json.load(open('$BRIEF_OUT/brief_data.json'))['sources']['public'])")
      if npx --yes wrangler@4.149.0 pages deploy "$BRIEF_OUT/site" --project-name "$PROJECT" --branch main \
           --commit-hash "$rev" --commit-dirty=true > "$BRIEF_OUT.deploy.log" 2>&1; then
        last=$now; echo "$now" > "$BRIEF_OUT/.deployed.sha256"
        echo "$ts deployed $rev ($(grep -o 'https://[a-z0-9]*\.[a-z-]*\.pages\.dev' "$BRIEF_OUT.deploy.log" | tail -1))"
      else
        echo "$ts deploy FAILED; see $BRIEF_OUT.deploy.log"
      fi
    else
      echo "$ts unchanged"
    fi
  else
    echo "$ts build FAILED; see $BRIEF_OUT.log (the live page is unchanged)"
  fi
  sleep "$INTERVAL"
done
