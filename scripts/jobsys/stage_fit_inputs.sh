#!/usr/bin/env bash
# Reuse immutable job inputs only after checking their SHA metadata and byte count.
set -euo pipefail
run=$1
shift
for artifact in "$@"; do
  pin=$(sha256sum "$artifact" | cut -d' ' -f1)
  bytes=$(stat -c %s "$artifact")
  key="jobs/$run/inputs/$pin"
  head=(aws --endpoint-url "$EP" s3api head-object --bucket zentrain --key "$key"
        --query '[Metadata.sha256,ContentLength]' --output text)
  if metadata=$("${head[@]}" 2>&1); then
    :
  elif [[ "$metadata" == *'(404)'* || "$metadata" == *'(NoSuchKey)'* ]]; then
    aws --endpoint-url "$EP" s3 cp "$artifact" "s3://zentrain/$key" \
      --metadata "sha256=$pin" --only-show-errors
    metadata=$("${head[@]}")
  else
    printf '%s\n' "$metadata" >&2
    exit 1
  fi
  read -r actual_pin actual_bytes <<< "$metadata"
  if [ "$actual_pin" != "$pin" ] || [ "$actual_bytes" != "$bytes" ]; then
    echo "immutable input metadata/size mismatch: $key" >&2
    exit 1
  fi
  echo "verified input $pin ($bytes bytes)"
done
