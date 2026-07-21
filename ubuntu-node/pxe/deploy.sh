#!/usr/bin/env bash
# deploy.sh — push the zen-pxe stack to the Unraid tower and (re)start the 3 containers.
# Unraid has no docker-compose plugin, so we drive plain `docker run` over SSH. Idempotent.
#
#   bash deploy.sh              # sync configs + (re)start containers
#   bash deploy.sh --assets     # also (re)extract kernel/initrd from the ISO on the tower
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOWER_SSH="${PXE_TOWER_SSH:-root@tower}"
P="${PXE_TPATH:-/mnt/user/coefficient/pxe}"
SCP(){ scp -o BatchMode=yes -o StrictHostKeyChecking=no "$@"; }
T(){ ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$TOWER_SSH" "$@"; }

echo "== sync configs + code to $TOWER_SSH:$P =="
T "mkdir -p $P/http $P/tftp $P/state/flags $P/state/registry $P/state/inventory $P/state/seen"
SCP "$HERE/server.py" "$HERE/dnsmasq.conf" "$HERE/nginx.conf" "$HERE/docker-compose.yml" "$TOWER_SSH:$P/"
SCP "$HERE/http/boot.ipxe" "$TOWER_SSH:$P/http/boot.ipxe"

if [ "${1:-}" = "--assets" ]; then
  echo "== (re)extract kernel/initrd from ISO =="
  T "set -e; mkdir -p /tmp/isomnt $P/http/ubuntu-26.04 $P/http/iso
     ISO=\$(ls $P/http/iso/ubuntu-26.04-*-live-server-amd64.iso $P/iso/ubuntu-26.04-*-live-server-amd64.iso 2>/dev/null | head -1)
     [ -n \"\$ISO\" ] || { echo 'no ISO found under $P'; exit 1; }
     mount -o loop,ro \"\$ISO\" /tmp/isomnt
     cp -f /tmp/isomnt/casper/vmlinuz $P/http/ubuntu-26.04/vmlinuz
     cp -f /tmp/isomnt/casper/initrd  $P/http/ubuntu-26.04/initrd
     umount /tmp/isomnt
     [ -f $P/http/iso/\$(basename \$ISO) ] || mv -n \"\$ISO\" $P/http/iso/"
fi

echo "== (re)start containers =="
T "
docker rm -f zen-pxe-dnsmasq zen-pxe-nginx zen-pxe-api 2>/dev/null || true
docker run -d --name zen-pxe-dnsmasq --restart unless-stopped --network host --cap-add NET_ADMIN --cap-add NET_RAW \
  -v $P/dnsmasq.conf:/etc/dnsmasq.conf:ro -v $P/tftp:/tftp:ro \
  strm/dnsmasq -k --log-facility=- --conf-file=/etc/dnsmasq.conf
docker run -d --name zen-pxe-nginx --restart unless-stopped --network host \
  -v $P/nginx.conf:/etc/nginx/nginx.conf:ro -v $P/http:/www:ro nginx:stable
docker run -d --name zen-pxe-api --restart unless-stopped --network host \
  -e PXE_DATA=/data -e PXE_BASE=http://192.168.50.170:3080 -v $P:/data \
  python:3.12-slim python3 /data/server.py
"
sleep 3
T 'docker ps --filter name=zen-pxe --format "{{.Names}}: {{.Status}}"'
echo "== smoke test =="
T 'curl -s http://localhost:3080/api/boot/00-00-00-00-00-00 | tail -1; curl -sI http://localhost:3080/ubuntu-26.04/vmlinuz | head -1'
