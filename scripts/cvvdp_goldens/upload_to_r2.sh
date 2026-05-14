#!/usr/bin/env bash
# Upload cvvdp-gpu golden artifacts to R2.
#
# Inputs:
#   $1 = local directory produced by build_goldens.py (contains manifest.json
#        plus per-pair tensor .bin files).
#   $2 = R2 prefix under s3://coefficient/cvvdp-goldens/, e.g. "v1".
#
# Requires:
#   ~/.config/cloudflare/r2-credentials with R2_ACCOUNT_ID, R2_ACCESS_KEY_ID,
#   R2_SECRET_ACCESS_KEY exported.
#
# The cvvdp-gpu Rust tests fetch over coefficient.r2.imazen.org (the
# public mirror of s3://coefficient/), so this script also confirms the
# manifest is reachable over that URL after upload.

set -euo pipefail

LOCAL_DIR=${1:?usage: upload_to_r2.sh <local-dir> <r2-prefix>}
R2_PREFIX=${2:?usage: upload_to_r2.sh <local-dir> <r2-prefix>}

# shellcheck disable=SC1091
source "$HOME/.config/cloudflare/r2-credentials"

R2_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"

S3() { aws --endpoint-url "$R2_ENDPOINT" "$@"; }

DEST="s3://coefficient/cvvdp-goldens/${R2_PREFIX}/"
echo "uploading $LOCAL_DIR/ -> $DEST"
S3 s3 cp --recursive "$LOCAL_DIR/" "$DEST"

echo "verifying listing..."
S3 s3 ls "$DEST" | head -20
