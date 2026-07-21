#!/usr/bin/env bash
# mint_cred.sh — mint a scoped, 7-day R2 temp credential and print it as shell env lines.
#
# Run on the DEV box (it holds the CF management token in ~/.config/cloudflare/r2-credentials).
# 7 days (604800s) is the HARD MAX for R2 temp-access-credentials — a 30-day cred would need a
# dashboard-created scoped R2 API token instead (see README "Longer than 7 days"). The cred is
# scoped to the zentrain bucket + only the prefixes the worker touches, object-read-write.
#
# Output (to stdout) is three lines, ready to drop into an env file / eval:
#   AWS_ACCESS_KEY_ID=...
#   AWS_SECRET_ACCESS_KEY=...
#   AWS_SESSION_TOKEN=...
# The session token is MANDATORY — temp creds 403 on key+secret alone.
set -euo pipefail

# Resolve the CF creds file even under sudo (where $HOME becomes /root).
resolve_creds() {
  local c
  for c in "${R2_CREDENTIALS_FILE:-}" \
           "$HOME/.config/cloudflare/r2-credentials" \
           "${SUDO_USER:+/home/$SUDO_USER/.config/cloudflare/r2-credentials}" \
           "/home/lilith/.config/cloudflare/r2-credentials"; do
    [ -n "$c" ] && [ -r "$c" ] && { echo "$c"; return 0; }
  done
  return 1
}
CREDS="$(resolve_creds)" || { echo "mint_cred: no readable CF creds file found (set R2_CREDENTIALS_FILE)" >&2; exit 1; }
set -a; . "$CREDS"; set +a

BUCKET="${ZEN_BUCKET:-zentrain}"
TTL="${ZEN_CRED_TTL:-604800}"   # 7 days, the R2 temp-cred maximum

body=$(BK="$BUCKET" TL="$TTL" python3 - <<'PY'
import json, os
print(json.dumps({
    "bucket": os.environ["BK"],
    "parentAccessKeyId": os.environ["R2_ACCESS_KEY_ID"],
    "parentSecretAccessKey": os.environ["R2_SECRET_ACCESS_KEY"],
    "permission": "object-read-write",
    "ttlSeconds": int(os.environ["TL"]),
    "prefixes": ["jobs/", "jxl-lossy/runs/", "canonical/2026-06-27/", "refs/"],
}))
PY
)

J=$(curl -sS -X POST \
      -H "Authorization: Bearer $R2_API_TOKEN" \
      -H "Content-Type: application/json" -d "$body" \
      "https://api.cloudflare.com/client/v4/accounts/$R2_ACCOUNT_ID/r2/temp-access-credentials")

read -r AK SK ST < <(printf '%s' "$J" | python3 -c \
  'import json,sys;r=json.load(sys.stdin)["result"];print(r["accessKeyId"],r["secretAccessKey"],r["sessionToken"])' \
  2>/dev/null) || { echo "mint_cred: unexpected response: $J" >&2; exit 1; }
[ -n "${AK:-}" ] || { echo "mint_cred: mint failed: $J" >&2; exit 1; }

printf 'AWS_ACCESS_KEY_ID=%s\nAWS_SECRET_ACCESS_KEY=%s\nAWS_SESSION_TOKEN=%s\n' "$AK" "$SK" "$ST"
