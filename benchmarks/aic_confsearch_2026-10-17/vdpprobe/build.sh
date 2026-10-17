#!/usr/bin/env bash
# Assemble a scratch mini-workspace (crates/hdrvdp + crates/zenmetrics-abstractions + this probe)
# and build the probe there, so the full zenmetrics workspace (and its sibling path deps) is not
# needed. hdrvdp's dev-dependencies, examples, benches and tests are stripped in the copy only.
# usage: build.sh <zenmetrics checkout> <scratch dir>
set -euo pipefail
REPO=$(cd "$1" && pwd); B=$2; HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$B"
rsync -a --delete --exclude target "$REPO/crates/hdrvdp/" "$B/hdrvdp/"
rsync -a --delete --exclude target "$REPO/crates/zenmetrics-abstractions/" "$B/zenmetrics-abstractions/"
rsync -a --delete "$HERE/" "$B/vdpprobe/" --exclude build.sh
rm -rf "$B/hdrvdp/examples" "$B/hdrvdp/benches" "$B/hdrvdp/tests"
python3 - "$REPO/Cargo.toml" "$B" <<'PY'
import re, sys
root = open(sys.argv[1]).read(); B = sys.argv[2]
lints = root[root.index('[workspace.lints.rust]'):]
nxt = lints.find('\n[', lints.index('[workspace.lints.clippy]') + 5)
lints = lints if nxt < 0 else lints[:nxt]
dep = lambda n: [l for l in root.splitlines() if l.startswith(n + ' =')][0]
open(f'{B}/Cargo.toml', 'w').write(f'''[workspace]
members = ["hdrvdp", "zenmetrics-abstractions", "vdpprobe"]
resolver = "2"
[workspace.package]
authors = ["Imazen <support@imazen.io>"]
homepage = "https://github.com/imazen/zenmetrics"
repository = "https://github.com/imazen/zenmetrics"
edition = "2024"
rust-version = "1.93"
[workspace.dependencies]
{dep('archmage')}
{dep('magetypes')}
{dep('image')}
{dep('rgb')}
{lints}
''')
m = open(f'{B}/hdrvdp/Cargo.toml').read()
m = re.sub(r'\[dev-dependencies\].*?(?=\n\[features\]|\n\[lints\]|\Z)', '', m, flags=re.S)
m = re.sub(r'\[\[example\]\]\nname = "[^"]+"\n', '', m)
m = re.sub(r'\[\[bench\]\]\nname = "[^"]+"\nharness = false\n', '', m)
open(f'{B}/hdrvdp/Cargo.toml', 'w').write(m)
PY
cd "$B" && cargo build --release -p vdpprobe
