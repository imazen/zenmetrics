#!/usr/bin/env bash
# Tear down a fleet launched by launch_fleet.sh: every box (any provider) is labeled group=<RUN>, so
# teardown is provider-agnostic. Idempotent — safe to re-run.
#   Usage: bash scripts/jobsys/teardown_fleet.sh <RUN>
set -uo pipefail
RUN="${1:?usage: teardown_fleet.sh <RUN>}"

# Hetzner (the dashboard's Kill button does the same via the label selector)
export HCLOUD_TOKEN=$(grep -E '^api_token=' ~/.config/hetzner/credentials | head -1 | cut -d= -f2- | tr -d ' \r')
for id in $(hcloud server list -l "group=$RUN" -o noheader -o columns=id 2>/dev/null); do
  hcloud server delete "$id" >/dev/null 2>&1 && echo "hetzner server $id deleted"
done

# vast (match the label)
vastai show instances-v1 --raw 2>/dev/null | python3 -c "
import sys,json
d=json.load(sys.stdin); insts=d if isinstance(d,list) else d.get('instances',[])
for x in insts:
    if str(x.get('label','')) == '$RUN': print(x['id'])
" | while read -r v; do yes | vastai destroy instance "$v" >/dev/null 2>&1 && echo "vast instance $v destroyed"; done

# Salad tier — DELETE the container group $RUN-salad (idempotent: 404 if it never existed)
SALAD_KEY=$(grep -E '^salad_' ~/.config/salad/credentials 2>/dev/null | head -1 | tr -d ' \r\n')
if [ -n "$SALAD_KEY" ]; then
  ORG="${SALAD_ORGANIZATION:-imazen}"; PROJ="${SALAD_PROJECT:-zenmetrics}"
  code=$(curl -s -o /dev/null -w '%{http_code}' -X DELETE -H "Salad-Api-Key: $SALAD_KEY" \
    "https://api.salad.com/api/public/organizations/$ORG/projects/$PROJ/containers/$RUN-salad")
  case "$code" in 2*) echo "salad group $RUN-salad deleted";; 404) :;; *) echo "salad delete -> HTTP $code";; esac
fi

# local docker tier
docker rm -f "$RUN-local" >/dev/null 2>&1 && echo "local worker container removed" || true

echo "teardown done for $RUN. verify: hcloud server list -l group=$RUN ; vastai show instances-v1 ; salad portal"
