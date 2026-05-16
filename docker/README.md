# docker/ — vast.ai images for PyTorch reference metrics

Two Dockerfiles, one per metric. Each image ships:

- `zen-metrics` binary, built with `--features sweep,external-metric` so
  the `--metric cvvdp` / `--metric iwssim` flags route through the
  long-lived subprocess plugin (see
  `crates/zen-metrics-cli/src/metrics/external.rs`).
- A Python venv (`/opt/venvs/<metric>`) with the reference impl
  installed once at build time. No pip work on the worker.
- The `metric_server.py` + per-metric `*_impl.py` server scripts at
  `/opt/python/`.
- `s5cmd` + `jq` static binaries for R2 / chunk I/O.
- `zen-metrics-worker` (= `onstart_v3.sh`) as the entrypoint so vast.ai
  workers boot straight into the chunk loop.
- `ZEN_METRICS_EXTERNAL_<METRIC>` baked into the image env so the plugin
  picks up the launch command without orchestration glue.

## Building

Build context is the parent of `zenmetrics--pytorch-refs/` (`~/work/zen`),
matching the layout the existing `Dockerfile.sweep.v13` uses. Sibling
crates (`zenjpeg/`, `zenanalyze/`) are pulled in by path the same way.

```bash
cd ~/work/zen

# CVVDP — CUDA 12.4 + cuDNN, ~5 GB. Requires a GPU at runtime.
docker buildx build \
    -f zenmetrics--pytorch-refs/docker/Dockerfile.cvvdp \
    -t ghcr.io/imazen/zen-metrics-cvvdp:0.6.0 \
    --cache-to type=local,dest=/tmp/buildx-cache-cvvdp \
    --cache-from type=local,src=/tmp/buildx-cache-cvvdp \
    .

# IW-SSIM — CPU PyTorch, ~1.5 GB. Runs on cheap CPU-only boxes.
docker buildx build \
    -f zenmetrics--pytorch-refs/docker/Dockerfile.iwssim \
    -t ghcr.io/imazen/zen-metrics-iwssim:0.6.0 \
    --cache-to type=local,dest=/tmp/buildx-cache-iwssim \
    --cache-from type=local,src=/tmp/buildx-cache-iwssim \
    .
```

## Local smoke test

```bash
# Spawn just the Python server interactively — feed it one frame pair
# from any RGB8 buffer to confirm wire protocol.
docker run --rm -it --gpus all \
    ghcr.io/imazen/zen-metrics-cvvdp:0.6.0 \
    /opt/venvs/cvvdp/bin/python /opt/python/metric_server.py cvvdp
# Expected first line of stdout:
#   {"ready": true, "name": "cvvdp"}
```

```bash
# End-to-end via zen-metrics score, using a checkerboard reference and a
# JPEG-compressed variant. Bind-mounts a sources dir.
docker run --rm --gpus all \
    -v $PWD/sources:/srv:ro \
    ghcr.io/imazen/zen-metrics-cvvdp:0.6.0 \
    zen-metrics score \
        --metric cvvdp \
        --reference /srv/ref.png \
        --distorted /srv/dist.png \
        --output json
```

## vast.ai launch

The `scripts/sweep/v17_pytorch_refs/launch_*.sh` launchers wrap
`vastai create instance` with the right machine filters
(cvvdp: NVIDIA + ≥12 GB VRAM; iwssim: CPU-only) and pass the env vars
the onstart script needs (`R2_*`, `SWEEP_RUN_ID`).

## Plugin handshake summary

1. zen-metrics starts a worker process.
2. First time `--metric cvvdp` is needed, the external-metric plugin
   reads `ZEN_METRICS_EXTERNAL_CVVDP`, splits it on whitespace, and
   spawns the command with stdin/stdout piped.
3. The Python server writes one ready line: `{"ready":true,"name":"cvvdp"}`.
4. zen-metrics issues per-frame requests; the server replies one JSON
   score line per request.
5. When zen-metrics exits, the server is `kill()`-ed in `Drop`.
