#!/usr/bin/env bash
# Bounded native worker admission under ONE caller-owned run-heavy scope.
# GNU parallel reads the slot count file again when its native passes finish;
# operators may change that file after checking the enclosing scope's memory.
set -euo pipefail
native_root=$(realpath "${1:?runtime directory}")
native_repo=$(cd "$(dirname "$0")/../.." && pwd)
# Bash may reread a script after a long foreground worker returns. Pin the
# wrapper bytes before launching so later operator fixes cannot shift its offset.
native_script=$(python3 - "$native_repo" "$native_root" <<'PYSNAPSHOT'
import hashlib, shlex, sys
from pathlib import Path
repo, root = map(Path, sys.argv[1:])
source = (repo/'scripts/jobsys/native_fit_pass.sh').read_text()
anchor = 'native_repo=$(cd "$(dirname "$0")/../.." && pwd)'
if source.count(anchor) != 1:
    raise ValueError('native wrapper repository anchor differs')
content = source.replace(anchor, 'native_repo='+shlex.quote(str(repo))).encode()
target = root/'native-wrappers'/('pass-'+hashlib.sha256(content).hexdigest()+'.sh')
target.parent.mkdir(parents=True, exist_ok=True)
if target.exists():
    if target.read_bytes() != content:
        raise ValueError('immutable native wrapper snapshot differs')
else:
    with target.open('xb') as stream:
        stream.write(content)
print(target)
PYSNAPSHOT
)
native_mode=${2:-initial}
if [[ "$native_mode" == initial ]]; then
  cat /proc/self/cgroup > "$native_root/scope.cgroup"
  printf '%s\n' "$$" > "$native_root/batch.pid"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$native_root/STARTED_AT"
  native_unit=$(basename "$(cut -d: -f3 "$native_root/scope.cgroup")")
  [[ "$(systemctl --user show "$native_unit" -p Delegate --value)" == yes ]]
  [[ "$(systemctl --user show "$native_unit" -p OOMPolicy --value)" == continue ]]
  sudo -n python3 "$native_repo/scripts/jobsys/native_scope_limits.py" --runtime "$native_root"
  native_slots="$native_root/slots"
  native_plan="$native_root/worker-plan.tsv"
  native_log="$native_root/native-passes.tsv"
else
  [[ "$native_mode" =~ ^recovery[0-9]+$ ]]
  native_slots="$native_root/$native_mode-slots"
  native_plan="$native_root/$native_mode-plan.tsv"
  native_log="$native_root/$native_mode-passes.tsv"
fi
exec parallel --line-buffer --jobs "$native_slots" --halt soon,fail=1 \
  --joblog "$native_log" --colsep '\t' \
  bash "$native_script" "$native_root" '{1}' '{2}' :::: "$native_plan"
