#!/usr/bin/env bash
# Build (and optionally deploy) the metric-cost brief from the current data.
#
#   scripts/complexity/brief.sh            # regenerate the paper, collect, render, print PDF
#   scripts/complexity/brief.sh --deploy   # ... then upload to Cloudflare Pages
#
# Runs locally because two inputs are private aggregates that never enter the repo:
#   BRIEF_AGREEMENT  space-separated agreement JSON files (agreement_from_scores.py output and the
#                    published-column agreement); default: the two files under ~/tmp/iqa-cx-lead
#   BRIEF_PRIVATE    zenmetrics-private checkout, read with git show origin/main (default ../zenmetrics-private)
# Other settings: BRIEF_OUT (default ~/tmp/metric-brief), BRIEF_CHROME (headless Chromium for the PDF),
# BRIEF_PROJECT (Cloudflare Pages project), CLOUDFLARE_API_TOKEN or ~/.config/cloudflare/zenmetrics-pages-token.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=${BRIEF_OUT:-$HOME/tmp/metric-brief}
PRIVATE=${BRIEF_PRIVATE:-$(cd .. && pwd)/zenmetrics-private}
[ -d "$PRIVATE" ] || PRIVATE=$HOME/work/zen/zenmetrics-private
AGREE=${BRIEF_AGREEMENT:-"$HOME/tmp/iqa-cx-lead/aic2026/agreement.json $HOME/tmp/iqa-cx-lead/dvifm-agree/dvifm_agreement.json"}
rm -rf "$OUT/site"; mkdir -p "$OUT/site"  # only the page and its PDF are deployed

python3 scripts/complexity/paper.py >/dev/null
args=()
for f in $AGREE; do [ -f "$f" ] && args+=(--agreement "$f") || echo "brief: agreement file $f missing; skipped" >&2; done
python3 scripts/complexity/brief_data.py --out "$OUT/brief_data.json" --private "$PRIVATE" "${args[@]}"
python3 scripts/complexity/brief_page.py --data "$OUT/brief_data.json" --out "$OUT/site"

CHROME=${BRIEF_CHROME:-$(ls -d "$HOME"/.cache/ms-playwright/chromium_headless_shell-*/*/chrome-headless-shell 2>/dev/null | tail -1)}
if [ -x "${CHROME:-}" ]; then
  "$CHROME" --no-sandbox --no-pdf-header-footer --print-to-pdf="$OUT/site/brief.pdf" "file://$OUT/site/index.html" 2>/dev/null
  echo "brief: $OUT/site/brief.pdf"
else
  echo "brief: no headless Chromium found (set BRIEF_CHROME); PDF skipped" >&2
fi
echo "brief: $OUT/site/index.html"

if [ "${1:-}" = "--deploy" ]; then
  PROJECT=${BRIEF_PROJECT:?set BRIEF_PROJECT to the Cloudflare Pages project}
  export CLOUDFLARE_API_TOKEN=${CLOUDFLARE_API_TOKEN:-$(cat "$HOME/.config/cloudflare/zenmetrics-pages-token")}
  npx --yes wrangler@4.149.0 pages deploy "$OUT/site" --project-name "$PROJECT" --branch main \
    --commit-hash "$(python3 -c "import json;print(json.load(open('$OUT/brief_data.json'))['sources']['public'])")"
fi
