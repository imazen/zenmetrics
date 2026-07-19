#!/usr/bin/env python3
"""fleet smoke-image <IMAGE> — prove a rebuilt GPU executor image JIT-compiles and
runs its CUDA metric kernels on a REAL card before you repoint the fleet at it.

Why this exists: local `docker run` can't exercise the GPU (snap-docker on WSL2
can't pass `--gpus all`), so a rebuilt `:exec-gpu` image passes local smoke
(`list-metrics`, `--serve`) yet can still fail at runtime — the NVRTC / missing
`cuda_runtime.h` JIT failure the KADIS notes warn about only surfaces when cubecl
actually compiles a kernel on a card. This launches ONE cheap vast.ai GPU box,
scores a tiny baked (ref,dist) pair with EVERY GPU metric under
`--use-legacy-scheduler` (so a GPU failure can't be silently masked by the
orchestrator's CPU-fallback ladder), reads the result via `vastai logs`, and
ALWAYS destroys the box (never injects a vast key into the box).

PASS = every `-gpu` metric returns rc=0 with a numeric score and no
NVRTC/compile/panic flag. A real `cvvdp-gpu` JOD (not the 10.0 panic fall-through)
and `zensim-gpu` matching CPU zensim are the strongest tells the kernels ran.

  usage: fleet smoke-image ghcr.io/imazen/zenfleet-worker:exec-gpu
     or: scripts/jobsys/fleet_smoke_image.py <IMAGE>

Deps: `vastai` (authed) + python stdlib. The 96x96 test pair is embedded (below)
so there is nothing to fetch and the score is deterministic.
"""
import base64, json, subprocess, sys, time

# 96x96 structured ref + a mild-blur/bright distortion — structure preserved so the
# PNGs stay tiny (fit an onstart arg) yet the metrics see real, non-identical content.
REF_PNG_B64 = (
'iVBORw0KGgoAAAANSUhEUgAAAGAAAABgCAIAAABt+uBvAAADQUlEQVR4nO1cW27jMAyccZRn073yYo+2R8kRsk3jvNrsB2NCdeVHUARiIALCYEAQgjDWMLQdi79rvIIy1uAafAFX4BKcg3NwCgZwAgIE+AlewAt4BI9gDdbgO7hrxj/w7evYge/gHqzBA3gET+AF/ACvIECCE3AKzsA5uABX4Ap8AdfN2l7BX1/XuW7SFuACnDXrrECAV/ATPINn8AgewBrcR0t9+7bUXbPUulnqCQy4ErgClWMSRaAK+MBtjzj/wgM+ZAc5plEEqhrZjg1RFUuPB5wZuc55m4tAhjxvjQecmN3nljHgGHsvHttUUNUtJV8EUu/F6HGpQTWz+9wyF4FMuN0m6g4S2fL3HdbiAXt2+NC51KA9s/vcMg94Z3afW0YRKNkXdI3NPcl6NZ51/lYNGoNl5esOsuJ5a9xr0H01KH/fYS3ufdAA9z5osAbddpBjGlv3YvHYpoKqbin5ejc/vl8oKx5wYHafW+b6RNExja1n0vn7Dmtxfathpe+wxgMuzO5zy1wEMuF2m9h6N/+I8feRk+vVftTkAZ+MvPcIfO75/f9BA7xqroBjGgMYezh/32EtLgIl/excatCE2X1umQcEZve5ZQyYxt6LxzYVVHVLyReB1Htj+ouy4gFzZve5ZS4CmXC7TQxYxJ7M33dYiwcs2eFD51KDlszuc8tcd5BjGgNWsffGjM09yXo1nnX+gBdG3huDZeWLQIY8b42LxUy43SbqDhLZ8vcd1uKtGuS8zb0GDdag2w5yTGNPH7RNBVXdUvL1V2x8v1BW3O/FBmuQ34v1ou4gkS1/32Etrk8UrfQd1rg/kx6sQTNm97llFIGSfcGzjD8PnV/frH7vC54FH7t+EciQ563xgIrZfW4ZRSBDfYe1eGXhKlnG0ChnxfPWuJ6C59i1gzr/J71NBVXdUvL1FLzx/UVZcf9WY7AG3U7Bc0yjfA5lqO+wFm+dYea8zf2b1cEadGB2n1vGnu/mu8bmnmS9Gs86f893811YVr6f3TFYg/zsjl7084MG4n5+0AD3M8wGa5CfYdaLPecoblNBVbeUfN1B4/uFsuLeB3kf9NMatI89mb/vsBZv3Ys5/94H+Zn2/jzoJ/gfsXN1sVdAh/AAAAAASUVORK5CYII='
)
DIST_PNG_B64 = (
'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'
)
METRICS = "cvvdp-gpu zensim-gpu ssim2-gpu butteraugli-gpu dssim-gpu iwssim-gpu"
IMG = sys.argv[1] if len(sys.argv) > 1 else "ghcr.io/imazen/zenfleet-worker:exec-gpu"


def sh(cmd, timeout=90):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    onstart = f"""set +e
echo SMOKE_BEGIN
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1 | head -1 | sed 's/^/SMOKE_GPU /'
mkdir -p /sm
echo '{REF_PNG_B64}' | base64 -d > /sm/r.png
echo '{DIST_PNG_B64}' | base64 -d > /sm/d.png
echo "SMOKE_FILES $(ls -la /sm/r.png /sm/d.png | wc -l)"
( m=0; for i in $(seq 1 400); do u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null|head -1|tr -d ' '); [ "${{u:-0}}" -gt "$m" ] 2>/dev/null && m=$u; echo $m>/sm/mu; sleep 0.2; done ) &
SAMP=$!
for M in {METRICS}; do
  out=$(/usr/local/bin/zenmetrics score --metric $M --use-legacy-scheduler --reference /sm/r.png --distorted /sm/d.png 2>&1); rc=$?
  nv=$(printf '%s' "$out" | grep -aiE 'nvrtc|cuda_runtime|failed to compile|panic|ServerUnhealthy|fall.?back|no cuda|not available|error' | head -2 | tr '\\n' '~')
  sc=$(printf '%s' "$out" | grep -aoE 'metric=[^ ]+ [^ ]+=[-0-9.]+' | head -1)
  echo "SMOKE_M $M rc=$rc score=[$sc] flag=[$nv]"
done
kill $SAMP 2>/dev/null
echo "SMOKE_MAXUTIL $(cat /sm/mu 2>/dev/null)"
echo SMOKE_DONE
"""

    q = "reliability>0.98 num_gpus=1 gpu_ram>=12 rentable=true inet_down>400 disk_space>30 cuda_max_good>=12.6"
    rc, out, err = sh(["vastai", "search", "offers", q, "--order", "dph_total", "--raw"])
    offers = json.loads(out)
    offers = offers if isinstance(offers, list) else offers.get("offers", [])
    if not offers:
        log("no offers matched — cannot smoke")
        return 2
    log(f"{len(offers)} offers; cheapest id={offers[0]['id']} {offers[0].get('gpu_name')} ${offers[0]['dph_total']:.3f}/hr")

    inst = None
    try:
        for off in offers[:8]:
            rc, out, err = sh(["vastai", "create", "instance", str(off["id"]), "--image", IMG,
                               "--disk", "32", "--label", "fleet-smoke-image",
                               "--onstart-cmd", onstart, "--raw"], timeout=90)
            try:
                j = json.loads(out)
            except Exception:
                j = {}
            if j.get("success") and j.get("new_contract"):
                inst = j["new_contract"]
                log(f"launched instance {inst} on offer {off['id']} ({off.get('gpu_name')} ${off['dph_total']:.3f}/hr)")
                break
            log(f"offer {off['id']} create failed: {out.strip()[:160]}")
        if not inst:
            log("could not launch any offer")
            return 3

        deadline = time.time() + 420   # image pull (1.4GB) + boot + first-run NVRTC JIT
        seen = ""
        while time.time() < deadline:
            time.sleep(20)
            rc, out, err = sh(["vastai", "logs", str(inst), "--tail", "400"], timeout=60)
            blob = out + err
            new = blob[len(seen):] if blob.startswith(seen) else blob
            for line in new.splitlines():
                if any(k in line for k in ("SMOKE_", "nvidia", "CUDA", "Error", "error", "pull")):
                    log(f"box| {line.strip()[:200]}")
            seen = blob
            if "SMOKE_DONE" in blob:
                log("SMOKE_DONE seen")
                break
        else:
            log("TIMEOUT waiting for SMOKE_DONE")

        rc, out, err = sh(["vastai", "logs", str(inst), "--tail", "400"], timeout=60)
        blob = out + err
        rows = [l for l in blob.splitlines() if l.startswith("SMOKE_")]
        print("\n===== SMOKE VERDICT =====")
        for l in rows:
            print("  " + l.strip())
        metric_rows = [l for l in rows if l.startswith("SMOKE_M ")]
        ok = bool(metric_rows)
        if not metric_rows:
            print("  !! no metric rows — smoke did not run")
        for l in metric_rows:
            good = "rc=0 " in l and "score=[metric=" in l and not any(
                b in l.lower() for b in ("nvrtc", "cuda_runtime", "failed to compile", "panic", "serverunhealthy"))
            if not good:
                ok = False
        print(f"\n  VERDICT: {'PASS -- GPU kernels JIT+ran on a real card' if ok else 'FAIL -- see rows above'}")
        return 0 if ok else 1
    finally:
        if inst:
            log(f"destroying instance {inst} …")
            for _ in range(3):
                rc, out, err = sh(["vastai", "destroy", "instance", str(inst), "-y"], timeout=60)
                if "success" in (out + err).lower() or rc == 0:
                    log(f"destroyed {inst}: {(out or err).strip()[:120]}")
                    break
                time.sleep(5)
            else:
                log(f"!!! DESTROY MAY HAVE FAILED for {inst} — check `vastai show instances`")


if __name__ == "__main__":
    sys.exit(main())
