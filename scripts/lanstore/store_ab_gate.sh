#!/usr/bin/env bash
# SeaweedFS 4.44 conditional-PUT gate + small-object A/B vs pinned MinIO.
# Methodology: LANSTORE.md 2026-08-10 re-evaluation (loopback docker, stock config,
# interleaved s5cmd reps). Runs on a LAN box (r7900x), never the operator.
set -uo pipefail
W=~/tmp/storegate-20260822
mkdir -p "$W"/{swdata,miniodata,payload}
cd "$W"
exec > >(tee -a "$W/run.log") 2>&1
echo "=== storegate $(date -u +%FT%TZ) on $(hostname) ==="

cat > "$W/zens3.json" <<'EOF'
{"identities":[{"name":"zenop","credentials":[{"accessKey":"zenop","secretKey":"zensecret1234"}],"actions":["Admin","Read","Write","List","Tagging"]}]}
EOF

docker rm -f sw-gate minio-gate >/dev/null 2>&1 || true
echo "--- pulling images"
docker pull chrislusf/seaweedfs:4.44 | tail -1
docker pull minio/minio:RELEASE.2025-09-07T16-13-09Z | tail -1

echo "--- starting seaweedfs 4.44"
docker run -d --name sw-gate -p 127.0.0.1:18333:8333 \
  -v "$W/swdata":/data -v "$W/zens3.json":/etc/zens3.json:ro \
  chrislusf/seaweedfs:4.44 server -dir=/data -s3 -s3.port=8333 -s3.config=/etc/zens3.json
echo "--- starting minio (control)"
docker run -d --name minio-gate -p 127.0.0.1:19000:9000 \
  -e MINIO_ROOT_USER=zenop -e MINIO_ROOT_PASSWORD=zensecret1234 \
  -v "$W/miniodata":/data \
  minio/minio:RELEASE.2025-09-07T16-13-09Z server /data --address :9000

for i in $(seq 1 60); do
  a=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:18333/ || true)
  b=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:19000/minio/health/live || true)
  [ "$a" != "000" ] && [ "$b" = "200" ] && break
  sleep 2
done
echo "--- health: seaweed http=$a minio=$b"
docker exec sw-gate weed version 2>/dev/null | head -1 || true

echo "--- boto3 venv"
python3 -m venv "$W/venv" >/dev/null 2>&1
"$W/venv/bin/pip" -q install boto3 >/dev/null

echo "--- GATE: seaweedfs"
"$W/venv/bin/python" "$W/condput_gate.py" http://127.0.0.1:18333 zenop zensecret1234 seaweedfs-4.44
SW_GATE=$?
echo "--- GATE: minio control"
"$W/venv/bin/python" "$W/condput_gate.py" http://127.0.0.1:19000 zenop zensecret1234 minio-2025-09-07
MINIO_GATE=$?

echo "--- payload: 1000 x 4KiB"
for i in $(seq -w 0 999); do head -c 4096 /dev/urandom > "$W/payload/f$i"; done

S5=$(command -v s5cmd || echo "")
if [ -z "$S5" ]; then echo "s5cmd MISSING on this box - skipping throughput A/B"; fi
export AWS_ACCESS_KEY_ID=zenop AWS_SECRET_ACCESS_KEY=zensecret1234 AWS_REGION=us-east-1
if [ -n "$S5" ]; then
  $S5 --endpoint-url http://127.0.0.1:18333 mb s3://zbench >/dev/null 2>&1 || true
  $S5 --endpoint-url http://127.0.0.1:19000 mb s3://zbench >/dev/null 2>&1 || true
  for rep in 1 2 3; do
    for st in sw minio; do
      [ $st = sw ] && EP=http://127.0.0.1:18333 || EP=http://127.0.0.1:19000
      t0=$(date +%s.%N)
      $S5 --endpoint-url $EP cp "$W/payload/" "s3://zbench/r$rep/" >/dev/null 2>&1
      t1=$(date +%s.%N)
      mkdir -p "$W/dl-$st-$rep"
      t2=$(date +%s.%N)
      $S5 --endpoint-url $EP cp "s3://zbench/r$rep/*" "$W/dl-$st-$rep/" >/dev/null 2>&1
      t3=$(date +%s.%N)
      n=$(ls "$W/dl-$st-$rep" | wc -l)
      echo "BENCH rep=$rep store=$st put=$(echo "$t1 $t0" | awk '{printf "%.2f",$1-$2}')s get=$(echo "$t3 $t2" | awk '{printf "%.2f",$1-$2}')s got=$n"
      rm -rf "$W/dl-$st-$rep"
    done
  done
fi

echo "--- cleanup"
docker rm -f sw-gate minio-gate >/dev/null 2>&1
rm -rf "$W/swdata" "$W/miniodata" "$W/payload"
echo "=== DONE sw_gate=$SW_GATE minio_gate=$MINIO_GATE ==="
