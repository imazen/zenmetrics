#!/usr/bin/env bash
# One heavy job for the clean-room MDSI integration: full validation (tests with the
# 116-pair author-score gate required, clippy, fmt, oracle score+map parity), then the
# colour-accuracy panel and the speed record with the final code.
# Invoke through ~/tmp/devin/heavy from the workspace root. Nothing runs off-box.
set -euo pipefail
root=/var/tmp/gmsd-chroma
python3 scripts/gmsd-chroma/preflight.py
# Metric sources must be committed: the implementation commit is the last one touching them.
test -z "$(jj diff --name-only crates/gmsd)" || { echo "crates/gmsd has uncommitted changes"; exit 2; }
commit=$(jj log --no-graph -r 'latest(::@ & ~empty() & files("crates/gmsd"))' -T commit_id)
echo "implementation_commit $commit"
bash scripts/gmsd-chroma/validate_oracles.sh
python3 - "$commit" <<'PY'
import hashlib, json, sys
from pathlib import Path
files = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
         for p in [Path('crates/gmsd/Cargo.toml'), *sorted(Path('crates/gmsd/src').glob('*.rs')),
                   *sorted(Path('crates/gmsd/examples').glob('*.rs'))]}
Path('/var/tmp/gmsd-chroma/mdsi_cr_source_manifest.json').write_text(
    json.dumps({'implementation_commit': sys.argv[1], 'files': files}, indent=2) + '\n')
PY
python3 scripts/gmsd-chroma/score_colour.py --source-manifest "$root/mdsi_cr_source_manifest.json" \
    --out-dir "$root/colour_v3"
python3 scripts/gmsd-chroma/freeze_colour_report.py \
    --predictions "$root/colour_v3/predictions.tsv" \
    --predictions-manifest "$root/colour_v3/predictions.manifest.json" \
    --peers "$root/colour_v2/fast_ssim2.tsv" --peer-manifest "$root/colour_v2/fast_ssim2.manifest.json" \
    --panel "$root/panel" --output "$root/colour_v3/report_inputs.tsv"
python3 scripts/gmsd-chroma/report_colour.py \
    --table "$root/colour_v3/report_inputs.tsv" --manifest "$root/colour_v3/report_inputs.manifest.json" \
    --parity "$root/parity.json" --ms-parity "$root/ms_parity.json" \
    --panel "$root/panel" --owner "$root/stats-owner" --out "$root/colour_v3/report.json"
bash scripts/gmsd-chroma/run_speed.sh
"$root/target/release/examples/mdsi_vs_reference" "$root/mdsi_vs_reference.tsv"
