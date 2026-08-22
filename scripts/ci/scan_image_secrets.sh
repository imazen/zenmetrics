#!/usr/bin/env bash
# scan_image_secrets.sh — scan a container image for LEAKED CREDENTIALS.
#
# Why: every ghcr.io/imazen/* image is PUBLIC (world-pullable). A credential baked
# into a public image is a public leak the moment it is pushed. Our fleet images are
# "bake-everything" (CLAUDE.md) and READ creds from the runtime env (the launcher
# injects SCOPED temp R2 creds) — nothing should ever be baked. This gate proves it.
#
# Two complementary layers (a real secret trips at least one):
#   1) trufflehog docker mode, --results=verified --fail
#        Pulls every layer + the image config/ENV/history and fails on any VERIFIED
#        live secret (a known credential format it can validate against the provider).
#   2) curated high-signal grep over the flattened rootfs (crane export), scoped to
#        OUR layers (third-party dep trees excluded — that is where the benign
#        placeholder example-URIs live: pyarrow `username:password@host`, urllib3
#        `host.com:80/path`, the mc binary's MinIO help text, etc.). Catches the
#        custom-format secrets trufflehog's named detectors miss:
#          - AWS access-key ids (AKIA…)
#          - PEM private keys
#          - a credential-NAMED var/file assigned a LITERAL value (R2/AWS/CF/HCLOUD/
#            VAST/SALAD …_SECRET/_TOKEN/_KEY = <literal>), e.g. a baked ~/.aws/credentials
#            or .env. Env-var REFERENCES (${VAR}, $VAR, getenv, env::var) are NOT flagged.
#
# Usage:  scan_image_secrets.sh <image-ref> [<image-ref> ...]
#         TRUFFLEHOG=/path/to/trufflehog CRANE=/path/to/crane  (optional overrides)
#         SCAN_NO_GREP=1   skip layer 2 (trufflehog-only, faster)
# Exit:   0 = clean for every ref
#         1 = a finding in at least one ref (investigate + rotate)
#         2 = tooling/setup error (could not scan — treat as broken, do not "pass")
set -uo pipefail

THOG="${TRUFFLEHOG:-trufflehog}"
CRANE="${CRANE:-crane}"
command -v "$THOG"  >/dev/null || { echo "FATAL: trufflehog not on PATH (set TRUFFLEHOG=...)" >&2; exit 2; }
if [ "${SCAN_NO_GREP:-0}" != "1" ]; then
  command -v "$CRANE" >/dev/null || { echo "FATAL: crane not on PATH (set CRANE=... or SCAN_NO_GREP=1)" >&2; exit 2; }
  command -v tar      >/dev/null || { echo "FATAL: tar not on PATH" >&2; exit 2; }
fi
[ "$#" -ge 1 ] || { echo "usage: $0 <image-ref> [<image-ref> ...]" >&2; exit 2; }

# High-signal LITERAL-credential patterns (leak-shaped; verified zero FPs on current imazen images).
HISIG_RE='AKIA[0-9A-Z]{16}'
HISIG_RE+='|-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----'
HISIG_RE+='|(R2|AWS|CLOUDFLARE|CF|HCLOUD|VAST|SALAD|S3|MINIO)_[A-Z0-9_]*(SECRET|TOKEN|PASSWORD|ACCESS_KEY|API_?KEY|KEY)[A-Z0-9_]*[[:space:]]*[=:][[:space:]]*.?[A-Za-z0-9+/=_.-]{16,}'
HISIG_RE+='|aws_secret_access_key[[:space:]]*=[[:space:]]*[A-Za-z0-9+/=]{20,}'
# Exclude env-var REFERENCES, obvious placeholders, and the verified-benign example hosts.
# Applied to the matched LINE CONTENT only (not the file:lineno prefix) — see below.
# (`${VAR:-default}` param-expansion is already covered by the \$\{ alternative; do NOT
#  add a bare `:[-=?]` — it spuriously matches grep's `lineno:-----` on dash-led content.)
ALLOW_RE='\$\{|\$[A-Za-z_]|getenv|environ|env::var|std::env|os\.getenv|<[A-Za-z0-9_]+>|EXAMPLE|example\.(com|net|org)|localhost|YOUR_|REPLACE|placeholder|xxxxx|bogus\.net|host\.com|siteb\.example'

overall=0
for ref in "$@"; do
  echo "============================================================"
  echo ">>> scan $ref"
  found=0

  # ── Layer 1: trufflehog verified secrets ────────────────────────────────────
  tj="$(mktemp)"; te="$(mktemp)"
  if "$THOG" docker --image "$ref" --results=verified --fail --no-update --json >"$tj" 2>"$te"; then
    echo "    [trufflehog] no verified secrets"
  else
    rc=$?
    if [ "$rc" = "183" ] || grep -q '"DetectorName"' "$tj"; then   # 183 = trufflehog "found + --fail"
      echo "::error::[trufflehog] VERIFIED SECRET in $ref"
      jq -rc 'select(.DetectorName!=null)|{detector:.DetectorName,verified:.Verified,file:(.SourceMetadata.Data.Docker.file//"?"),redacted:.Redacted}' "$tj" 2>/dev/null || cat "$tj"
      found=1
    else
      echo "::error::[trufflehog] scan error (rc=$rc) for $ref — treat as BROKEN, not clean"; tail -5 "$te" >&2
      overall=2
    fi
  fi
  rm -f "$tj" "$te"

  # ── Layer 2: curated high-signal grep, SCOPED to OUR config/script/cred files ─
  # trufflehog (layer 1) scans every layer for known formats; it correctly ignores
  # the benign PEM/AKIA example strings that third-party tooling ships (aws-cli
  # examples/*.rst, libssh/libgnutls .so format markers + test vectors). Layer 2
  # fills trufflehog's gaps — custom-format creds (R2 non-AKIA keys), baked
  # ~/.aws/credentials/.env, and unverifiable private-key FILES — but ONLY over the
  # files where WE would put a secret, so the third-party noise never reaches it.
  if [ "${SCAN_NO_GREP:-0}" != "1" ]; then
    d="$(mktemp -d)"
    if "$CRANE" export "$ref" - 2>/dev/null | tar -x -C "$d" \
         --exclude='*/site-packages/*' --exclude='*/dist-packages/*' --exclude='*/opt/conda/*' \
         --exclude='*venv/*' --exclude='*/torch/*' --exclude='*/nvidia/*' --exclude='*/cuda*/*' \
         --exclude='*.pyc' --exclude='*/__pycache__/*' 2>/dev/null; then
      flist="$(mktemp)"
      { find "$d" -type f \( \
              -name '*.sh' -o -name '*.bash' -o -name '*.py' -o -name '*.env' -o -name '.env' \
              -o -name '.env.*' -o -name '*.json' -o -name '*.yaml' -o -name '*.yml' \
              -o -name '*.toml' -o -name '*.conf' -o -name '*.cfg' -o -name '*.ini' \
              -o -name '*.properties' -o -name 'credentials' -o -name '*.pem' -o -name '*.key' \
              -o -name '.netrc' -o -name '.git-credentials' -o -name 'rclone.conf' \
              -o -path '*/.aws/*' -o -path '*/.config/*' -o -path '*/.ssh/*' \) 2>/dev/null
        find "$d/root" "$d/home" -maxdepth 4 -type f -name '.*' 2>/dev/null   # dotfiles (.netrc, ...)
      } | grep -avE '/(aws-?cli|awscli|examples?|tests?|test[-_]?data|fixtures?|site-packages|dist-packages|conda|node_modules)/|\.rst$|\.so($|\.)|/usr/(lib|share)/|/usr/local/(aws-cli|lib|share)/|/usr/lib/python' \
        | sort -u > "$flist"
      hits=""
      if [ -s "$flist" ]; then
        raw="$(tr '\n' '\0' <"$flist" | xargs -0 -r grep -IanHE -e "$HISIG_RE" 2>/dev/null || true)"
        # Apply the allowlist to the matched CONTENT only (strip the `file:lineno:` prefix
        # first) so a benign path token can't mask a real secret, and so grep's lineno
        # colon can't collide with allow patterns.
        hits="$(printf '%s\n' "$raw" | while IFS= read -r ln; do
                  [ -n "$ln" ] || continue
                  c="${ln#*:}"; c="${c#*:}"
                  printf '%s' "$c" | grep -qaE -e "$ALLOW_RE" || printf '%s\n' "$ln"
                done)"
      fi
      if [ -n "$hits" ]; then
        echo "::error::[grep] high-signal LITERAL secret pattern in $ref (in OUR config/script/cred files):"
        echo "$hits" | sed "s#$d##" | head -40
        found=1
      else
        echo "    [grep] no literal-secret pattern in our config/script/cred files ($(wc -l <"$flist") files checked)"
      fi
      rm -f "$flist"
    else
      echo "::warning::[grep] could not export $ref rootfs — trufflehog layer still applies"
    fi
    rm -rf "$d"
  fi

  if [ "$found" = "1" ]; then echo "<<< FINDING in $ref"; overall=1; else echo "<<< clean: $ref"; fi
done

echo "============================================================"
case "$overall" in
  0) echo "RESULT: clean — no leaked credentials in any scanned image";;
  1) echo "RESULT: LEAK(S) FOUND — investigate, then ROTATE the exposed credential and delete/retag the image";;
  2) echo "RESULT: scan tooling error — could not complete; do NOT treat as clean";;
esac
exit "$overall"
