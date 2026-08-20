#!/usr/bin/env bash
# inventory-software.sh — enumerate every software source on a zen dev box.
#
# Deliberately reproducible rather than hand-maintained: run it on any box and diff the
# result against another to see exactly what is missing. That is the whole point — the
# x86_64 dev box is being replaced, and "what does the new box still need?" has to be a
# command, not a memory.
#
#   bash inventory-software.sh                 # markdown index to stdout
#   bash inventory-software.sh --raw <dir>     # also drop raw per-source lists in <dir>
#
# Sources covered: OS/kernel/CPU, language runtimes, rustup toolchains, cargo-installed
# crates, npm/pnpm globals, uv/pipx tools, the three bin dirs, apt manual selections,
# snap and flatpak. Anything absent is reported as absent rather than skipped silently.
set -uo pipefail
RAW=""; [ "${1:-}" = "--raw" ] && { RAW="${2:?--raw needs a dir}"; mkdir -p "$RAW"; }
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$HOME/.deno/bin:$HOME/.bun/bin:/usr/local/go/bin:$PATH"
emit(){ [ -n "$RAW" ] && cat > "$RAW/$1" || cat > /dev/null; }
have(){ command -v "$1" >/dev/null 2>&1; }

echo "# Software inventory — $(hostname)"
echo
echo "Generated \`$(date -u +%Y-%m-%dT%H:%M:%SZ)\` by \`inventory-software.sh\`."
echo

echo "## Host"
echo '```'
echo "hostname   $(hostname)"
echo "os         $(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME")"
echo "kernel     $(uname -sr)"
echo "arch       $(uname -m)"
echo "cpu        $(grep -m1 '^model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | xargs)"
echo "cores      $(nproc 2>/dev/null)"
echo "memory     $(free -h 2>/dev/null | awk '/^Mem:/{print $2}')"
echo '```'
echo

echo "## Language runtimes"
echo
echo "| tool | version |"
echo "|---|---|"
for c in rustc cargo rustup node npm pnpm yarn python3 pip3 go deno bun uv pipx mise java ruby perl; do
  if have "$c"; then
    # not every tool honours --version: go wants `go version`, perl's first line is blank
    case "$c" in
      go)   v=$(go version 2>&1 | head -1);;
      perl) v=$(perl -e 'print $^V' 2>&1 | head -1);;
      java) v=$(java -version 2>&1 | head -1);;
      *)    v=$("$c" --version 2>&1 | head -1);;
    esac
    v=$(printf '%s' "$v" | cut -c1-60)
    [ -n "$v" ] || v="(present, version unreported)"
  else v="**MISSING**"; fi
  printf '| `%s` | %s |\n' "$c" "$v"
done
echo

echo "## rustup toolchains"
echo '```'
have rustup && rustup toolchain list 2>/dev/null | tee >(emit rustup.txt) || echo "(rustup absent)"
echo '```'
echo

echo "## cargo-installed crates"
if have cargo; then
  cargo install --list 2>/dev/null | emit cargo.txt
  n=$(cargo install --list 2>/dev/null | grep -cE '^[a-zA-Z0-9_-]+ v')
  echo
  echo "$n crates."
  echo
  echo "| crate | version | source |"
  echo "|---|---|---|"
  cargo install --list 2>/dev/null | grep -E '^[a-zA-Z0-9_-]+ v' | while read -r line; do
    name=$(echo "$line" | awk '{print $1}')
    ver=$(echo "$line" | awk '{print $2}' | tr -d ':')
    src=$(echo "$line" | sed -n 's/.*(\(.*\)).*/\1/p')
    case "$src" in
      "") src="crates.io";;
      /*) src="**local path** \`$src\`";;
      http*) src="**git** \`$src\`";;
    esac
    printf '| `%s` | %s | %s |\n' "$name" "$ver" "$src"
  done
else echo; echo "(cargo absent)"; fi
echo

echo "## npm / pnpm globals"
echo '```'
have npm  && npm ls -g --depth=0 2>/dev/null | tee >(emit npm.txt) || echo "(npm absent)"
have pnpm && pnpm ls -g --depth=0 2>/dev/null | tee >(emit pnpm.txt) || echo "(pnpm absent)"
echo '```'
echo

echo "## uv / pipx tools"
echo '```'
have uv   && uv tool list 2>&1 | tee >(emit uvtools.txt) || echo "(uv absent)"
have pipx && pipx list --short 2>&1 | tee >(emit pipx.txt) || echo "(pipx absent)"
echo '```'
echo

echo "## bin directories"
for d in "$HOME/bin" "$HOME/.local/bin" /usr/local/bin "$HOME/.cargo/bin"; do
  echo
  echo "### \`$d\`"
  if [ -d "$d" ]; then
    ls "$d" 2>/dev/null | emit "bin$(echo "$d" | tr '/' '_').txt"
    echo "$(ls "$d" 2>/dev/null | wc -l) entries."
    echo '```'
    ls "$d" 2>/dev/null | tr '\n' ' ' | fold -s -w 96
    echo
    echo '```'
  else echo "(absent)"; fi
done
echo

echo "## apt — manually selected"
if have apt-mark; then
  apt-mark showmanual 2>/dev/null | emit apt-manual.txt
  echo
  echo "$(apt-mark showmanual 2>/dev/null | wc -l) packages."
  echo '```'
  apt-mark showmanual 2>/dev/null | tr '\n' ' ' | fold -s -w 96
  echo
  echo '```'
else echo; echo "(apt absent)"; fi
echo

echo "## snap / flatpak"
echo '```'
have snap    && snap list 2>/dev/null | tee >(emit snap.txt) || echo "(snap absent)"
have flatpak && flatpak list 2>/dev/null | tee >(emit flatpak.txt) || echo "(flatpak absent)"
echo '```'
