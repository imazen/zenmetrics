#!/usr/bin/env bash
# s3env.sh — THE resolver for the object-store endpoint + credentials used by
# zenmetrics operator tooling. Source this instead of re-deriving endpoints:
#
#   . "$(dirname "${BASH_SOURCE[0]}")/../lib/s3env.sh"
#
# Contract — `ZEN_STORE` selects the store; the LAN store is the DEFAULT.
#
#   ZEN_STORE unset / "tower" / "lan"  -> the LAN store (DEFAULT as of 2026-08-10,
#                             user directive). Endpoint + creds come from
#                             ZEN_S3_ENDPOINT / ZEN_S3_ACCESS_KEY_ID /
#                             ZEN_S3_SECRET_ACCESS_KEY in the environment, else from
#                             the env file $ZEN_S3_ENV (default
#                             ~/.config/zen/lanstore.env).
#   ZEN_STORE=r2           -> Cloudflare R2, endpoint derived from R2_ACCOUNT_ID in
#                             ~/.config/cloudflare/r2-credentials. This is the
#                             EXPLICIT opt-out and is REQUIRED for anything that
#                             launches cloud boxes (they cannot route to the LAN).
#   ZEN_S3_ENDPOINT set    -> that endpoint verbatim, whatever ZEN_STORE says
#                             (back-compat + arbitrary S3-compatible stores).
#
# BEFORE 2026-08-10 the default was R2 and the LAN store was opt-in. The flip means
# **a script that selects nothing now writes to the LAN store.** Anything that
# launches or feeds cloud boxes MUST set ZEN_STORE=r2 explicitly — see
# ZEN_S3_CLOUD_REACHABLE below.
#
# Exports on return:
#   EP                                    — the endpoint URL every s5cmd/aws call uses
#   ZEN_S3_STORE                          — resolved store kind: "lan" | "r2" | "explicit"
#   ZEN_S3_CLOUD_REACHABLE                — 1 if cloud boxes can route to EP, else 0.
#                                           Guards MUST branch on THIS, not on whether
#                                           ZEN_S3_ENDPOINT happens to be set (that test
#                                           silently inverted when the default flipped).
#                                           Force to 1 via ZEN_S3_CLOUD_REACHABLE=1 once
#                                           the LAN store is published through the home
#                                           gateway with a routable hostname.
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

# Precedence: an EXPLICIT ZEN_STORE=r2 wins over everything (so `ZEN_STORE=r2 <cmd>` is a
# reliable one-liner escape hatch even when ZEN_S3_ENDPOINT is exported ambiently by the
# shell profile); then an explicit ZEN_S3_ENDPOINT; then the default (the LAN store).
_zen_store_sel="${ZEN_STORE:-tower}"
if [ "$_zen_store_sel" != "r2" ] && [ "$_zen_store_sel" != "R2" ] && [ -n "${ZEN_S3_ENDPOINT:-}" ]; then
  _zen_store_sel="explicit"
fi

case "$_zen_store_sel" in
  r2|R2)
    if [ -z "${R2_ACCOUNT_ID:-}" ] && [ -f "$HOME/.config/cloudflare/r2-credentials" ]; then
      set -a; . "$HOME/.config/cloudflare/r2-credentials"; set +a
    fi
    : "${R2_ACCOUNT_ID:?ZEN_STORE=r2 but no R2_ACCOUNT_ID (need ~/.config/cloudflare/r2-credentials)}"
    : "${R2_ACCESS_KEY_ID:?}" "${R2_SECRET_ACCESS_KEY:?}"
    EP="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
    ZEN_S3_STORE="r2"
    : "${ZEN_S3_CLOUD_REACHABLE:=1}"
    ;;
  *)
    _zen_s3_envfile="${ZEN_S3_ENV:-$HOME/.config/zen/lanstore.env}"
    if { [ -z "${ZEN_S3_ENDPOINT:-}" ] || [ -z "${ZEN_S3_ACCESS_KEY_ID:-}" ]; } && [ -f "$_zen_s3_envfile" ]; then
      # env file supplies endpoint+creds; an explicitly-exported endpoint still wins
      _zen_s3_ep_keep="${ZEN_S3_ENDPOINT:-}"
      set -a; . "$_zen_s3_envfile"; set +a
      [ -n "$_zen_s3_ep_keep" ] && ZEN_S3_ENDPOINT="$_zen_s3_ep_keep"
      unset _zen_s3_ep_keep
    fi
    : "${ZEN_S3_ENDPOINT:?LAN store selected (ZEN_STORE=${ZEN_STORE:-tower}) but no ZEN_S3_ENDPOINT and no $_zen_s3_envfile — set ZEN_STORE=r2 to use Cloudflare R2}"
    : "${ZEN_S3_ACCESS_KEY_ID:?LAN store selected but ZEN_S3_ACCESS_KEY_ID is not set (export it or provide $_zen_s3_envfile)}"
    : "${ZEN_S3_SECRET_ACCESS_KEY:?LAN store selected but ZEN_S3_SECRET_ACCESS_KEY is not set}"
    EP="$ZEN_S3_ENDPOINT"
    R2_ACCESS_KEY_ID="$ZEN_S3_ACCESS_KEY_ID"
    R2_SECRET_ACCESS_KEY="$ZEN_S3_SECRET_ACCESS_KEY"
    ZEN_S3_STORE="$([ "$_zen_store_sel" = explicit ] && echo explicit || echo lan)"
    : "${ZEN_S3_CLOUD_REACHABLE:=0}"
    ;;
esac
unset _zen_store_sel

export EP R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY ZEN_S3_STORE ZEN_S3_CLOUD_REACHABLE
export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
export AWS_REGION="${AWS_REGION:-auto}"
