#!/usr/bin/env python3
"""fleet smoke-image <IMAGE> — prove a rebuilt GPU executor image JIT-compiles and
runs its CUDA metric kernels on a REAL card before you repoint the fleet at it.

Why this exists: local `docker run` can't exercise the GPU (snap-docker on WSL2
can't pass `--gpus all`), so a rebuilt `:exec-gpu` image passes local smoke
(`list-metrics`, `--serve`) yet can still fail at runtime — the NVRTC / missing
`cuda_runtime.h` JIT failure the KADIS notes warn about only surfaces when cubecl
actually compiles a kernel on a card. This launches ONE cheap vast.ai GPU box,
scores a tiny (ref,dist) pair with EVERY GPU metric under `--use-legacy-scheduler`
(so a GPU failure can't be silently masked by the orchestrator's CPU-fallback
ladder), reads the result via `vastai logs`, and ALWAYS destroys the box (never
injects a vast key into the box).

PASS = every `-gpu` metric returns rc=0 with a numeric score and no
NVRTC/compile/panic flag. A real `cvvdp-gpu` JOD (not the 10.0 panic fall-through)
and `zensim-gpu` matching CPU zensim are the strongest tells the kernels ran.

  usage: fleet smoke-image ghcr.io/imazen/zenfleet-worker:exec-gpu
     or: scripts/jobsys/fleet_smoke_image.py <IMAGE>

Deps: `vastai` (authed) + Pillow (`PIL`) in the launcher env + python stdlib. The
96x96 test pair is generated fresh each run (deterministic formula, no randomness)
and base64-embedded into the onstart — do NOT hard-code the base64 as a constant:
a corrupt paste yields a PNG PIL tolerates but the strict zenpng decoder rejects
(`Decompress(BadData)`), which looks like an image bug but is a harness bug.
"""
import base64, io, json, math, subprocess, sys, time


def _make_pair():
    """Deterministic 96x96 structured ref + a mild blur/bright distortion. Both
    stay tiny (structure preserved) so they fit an onstart arg. Returns (ref_b64,
    dist_b64)."""
    from PIL import Image, ImageFilter
    w = h = 96
    ref = Image.new("RGB", (w, h))
    ref.putdata([(int(127 + 120 * math.sin(x / 9.0)), int(127 + 120 * math.cos(y / 11.0)), (x ^ y) & 255)
                 for y in range(h) for x in range(w)])
    rb = io.BytesIO(); ref.save(rb, "PNG")
    db = io.BytesIO()
    ref.filter(ImageFilter.GaussianBlur(1.4)).point(lambda v: min(255, int(v * 1.08))).save(db, "PNG")
    return base64.b64encode(rb.getvalue()).decode(), base64.b64encode(db.getvalue()).decode()


REF_PNG_B64, DIST_PNG_B64 = _make_pair()
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
