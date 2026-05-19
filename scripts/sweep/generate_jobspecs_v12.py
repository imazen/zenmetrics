"""v12 jobspec — tightened for 30-min completion at 100 workers."""
import json, sys
from pathlib import Path

SOURCES_DIR = Path("/tmp/v12-sweep-sources")
OUT = Path("/tmp/v12_chunks.jsonl")

# 200 images, but smaller knob grid
images = sorted([p.name for p in SOURCES_DIR.glob("*.png")])
print(f"sources: {len(images)}", file=sys.stderr)

# Tightened q grid: 5 levels covers q5-q95 range
Q_GRID = "10,30,60,80,90"

KNOBS = {
    "zenjxl": {
        "distance": [0.5, 1.0, 2.0, 5.0],
        "effort": [3, 7],
        "patches": [True, False],
        "gaborish": [True, False],
    },
    "zenavif": {
        "speed": [4, 8],
        "qm": [True, False],
    },
    "zenwebp": {},  # just q grid
}

CHUNK_SIZE = 1  # 1 image per chunk for max parallelism

n_chunks = 0
with OUT.open("w") as f:
    for codec, knob_grid in KNOBS.items():
        for i in range(0, len(images), CHUNK_SIZE):
            chunk_imgs = images[i:i+CHUNK_SIZE]
            chunk_id = f"{codec}-{i//CHUNK_SIZE:03d}"
            spec = {
                "codec": codec,
                "chunk_id": chunk_id,
                "q_grid": Q_GRID,
                "knob_grid": json.dumps(knob_grid),
                "metrics": ["zensim", "ssim2_gpu"],
                "images": chunk_imgs,
            }
            f.write(json.dumps(spec))
            f.write("\n")
            n_chunks += 1

print(f"wrote {n_chunks} chunks", file=sys.stderr)

# Estimate
n_per_img = {
    "zenjxl": 4 * 2 * 2 * 2 * 5,    # 80 cells
    "zenavif": 2 * 2 * 5,            # 20 cells
    "zenwebp": 5,                    # 5 cells
}
total = sum(n_per_img[c] * len(images) for c in KNOBS)
print(f"total cells: {total:,}", file=sys.stderr)
print(f"chunks: {n_chunks}", file=sys.stderr)
print(f"per chunk: ~{total/n_chunks:.0f} cells", file=sys.stderr)
print(f"~3s per cell, 100 workers: {total*3/60/100:.1f} min wall time", file=sys.stderr)
