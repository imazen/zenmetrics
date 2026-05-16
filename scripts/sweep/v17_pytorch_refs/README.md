# v17_pytorch_refs — cvvdp + iwssim sweep launchers

Two launchers, one per metric, both following the v15 / onstart_v3
pattern. They differ only in their image, vast.ai machine filter, and
which `--metric` is expected in each chunk's spec.

| Launcher        | Image                                         | Filter         | Typical $/hr |
|-----------------|-----------------------------------------------|----------------|--------------|
| `launch_gpu.sh` | `ghcr.io/imazen/zen-metrics-cvvdp:0.6.0`      | NVIDIA + 12 GB | $0.15–0.30   |
| `launch_cpu.sh` | `ghcr.io/imazen/zen-metrics-iwssim:0.6.0`     | CPU-only       | $0.02–0.05   |

## Prereqs

1. Build and push both images (see `docker/README.md`).
2. Generate a `chunks.jsonl` with `"metrics": ["cvvdp"]` or
   `"metrics": ["iwssim"]` per chunk. The existing
   `generate_jobspecs_v06.py` already wires this from a `--metric` arg —
   reuse it with the new metric names.
3. Upload chunks to
   `s3://coefficient/jobs/<SWEEP_RUN_ID>/chunks.jsonl`.
4. Make sure your sources are mirrored at
   `s3://zentrain/<SWEEP_RUN_ID>/sources/` (onstart_v3 walks the prefix).

## Run

```bash
# cvvdp fleet (default 20 GPU boxes, $0.30/hr cap)
N_BOXES=20 ./launch_gpu.sh

# iwssim fleet (default 20 CPU boxes, $0.05/hr cap)
N_BOXES=20 ./launch_cpu.sh
```

Instance IDs land in `/tmp/${SWEEP_RUN_ID}_instances.txt`, same shape
as v15 for compatibility with `sweep_janitor.py` and
`destroy_all.sh`.

## Plugin handshake

Both images bake `ZEN_METRICS_EXTERNAL_<METRIC>` into their env, so the
plugin loads the right Python server without any chunk-side glue. A
zen-metrics worker spawns one Python subprocess per process — the
existing onstart fan-out (one zen-metrics per cgroup-allocated core)
naturally turns into N Python servers per box, each holding its own
model in memory.

## When to use which

- **cvvdp**: for picker training that needs perceptually-grounded JOD
  scores; the canonical reference for HDR / colour-aware quality work.
  Run alongside zensim when triangulating two models.
- **iwssim**: for cheap, content-aware single-scale SSIM; useful as a
  second-opinion signal against the GPU-accelerated SSIMULACRA2 / DSSIM
  fleet. Same chunk grid, different content weighting.

Mixed fleets are fine: build the picker training set from one cvvdp
sweep + one iwssim sweep over the same `chunks.jsonl` source list,
then join on `(image_path, codec, q, knob_tuple_json)` in zenanalyze.
