#!/usr/bin/env bash
# s3env.sh — THE resolver for the object-store endpoint + credentials used by
# zenmetrics operator tooling. Source this instead of re-deriving endpoints:
#
#   . "$(dirname "${BASH_SOURCE[0]}")/../lib/s3env.sh"
#
# Contract (ONE env var switches the store):
#
#   ZEN_S3_ENDPOINT unset  -> Cloudflare R2. Endpoint is derived from
#                             R2_ACCOUNT_ID in ~/.config/cloudflare/r2-credentials.
#                             This is the legacy behavior, byte-for-byte unchanged.
#   ZEN_S3_ENDPOINT set    -> that endpoint verbatim (any S3-compatible store, e.g.
#                             a LAN deployment). Credentials come from
#                             ZEN_S3_ACCESS_KEY_ID / ZEN_S3_SECRET_ACCESS_KEY in the
#                             environment, else from the env file $ZEN_S3_ENV
#                             (default ~/.config/zen/lanstore.env).
#
# Exports on return:
#   EP                                    — the endpoint URL every s5cmd/aws call uses
#   R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY — creds under the legacy names existing
#                                           scripts already reference
#   R2_ACCOUNT_ID                         — only in the R2 arm
#   AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION — ready for s5cmd/aws
#
# Fail-loud: missing creds for the selected store is a hard error, never a silent
# fall-through to the other store. Worker-side code is NOT in scope here — workers
# receive ZEN_R2_ENDPOINT + AWS_* by injection at launch and are already
# endpoint-agnostic; launchers that source this lib inject $EP, so the whole fleet
# follows the operator's selection.

if [ -n "${ZEN_S3_ENDPOINT:-}" ]; then
  _zen_s3_envfile="${ZEN_S3_ENV:-$HOME/.config/zen/lanstore.env}"
  if [ -z "${ZEN_S3_ACCESS_KEY_ID:-}" ] && [ -f "$_zen_s3_envfile" ]; then
    # env file supplies creds; an explicitly-exported endpoint still wins over the file's
    _zen_s3_ep_keep="$ZEN_S3_ENDPOINT"
    set -a; . "$_zen_s3_envfile"; set +a
    ZEN_S3_ENDPOINT="$_zen_s3_ep_keep"; unset _zen_s3_ep_keep
  fi
  : "${ZEN_S3_ACCESS_KEY_ID:?ZEN_S3_ENDPOINT is set but ZEN_S3_ACCESS_KEY_ID is not (export it or provide $_zen_s3_envfile)}"
  : "${ZEN_S3_SECRET_ACCESS_KEY:?ZEN_S3_ENDPOINT is set but ZEN_S3_SECRET_ACCESS_KEY is not}"
  EP="$ZEN_S3_ENDPOINT"
  R2_ACCESS_KEY_ID="$ZEN_S3_ACCESS_KEY_ID"
  R2_SECRET_ACCESS_KEY="$ZEN_S3_SECRET_ACCESS_KEY"
else
  if [ -z "${R2_ACCOUNT_ID:-}" ] && [ -f "$HOME/.config/cloudflare/r2-credentials" ]; then
    set -a; . "$HOME/.config/cloudflare/r2-credentials"; set +a
  fi
  : "${R2_ACCOUNT_ID:?no ZEN_S3_ENDPOINT and no R2_ACCOUNT_ID (need ~/.config/cloudflare/r2-credentials)}"
  : "${R2_ACCESS_KEY_ID:?}" "${R2_SECRET_ACCESS_KEY:?}"
  EP="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
fi

export EP R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY
export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
export AWS_REGION="${AWS_REGION:-auto}"
