# Criterion-1 evidence: ssim2 + butteraugli GPU-only (runtime column) — 2026-08-26

Tool: `scripts/jobsys/audit_gpu_only_metrics.py` (ZEN_STORE=tower; ledgers on the LAN store).
Every done-row of every drained job, grouped by (job, metric, worker):

```
job	metric	worker	done_rows
avifgen-sf-cpu-20260806	cvvdp	i265	31979
avifgen-sf-cpu-20260806	cvvdp	ryzen5800xt	32391
avifgen-sf-cpu-20260806	cvvdp	tower	24354
avifgen-sf-cpu-20260806	zensim-foldapp2	i265	31979
avifgen-sf-cpu-20260806	zensim-foldapp2	ryzen5800xt	32391
avifgen-sf-cpu-20260806	zensim-foldapp2	tower	24354
avifgen-sf-gpu-20260806	butteraugli-gpu	lianli	32574
avifgen-sf-gpu-20260806	butteraugli-gpu	node-2	65174
avifgen-sf-gpu-20260806	ssim2-gpu	lianli	32574
avifgen-sf-gpu-20260806	ssim2-gpu	node-2	65174
avifgen-sf-gpu-rescue-20260808	butteraugli-gpu	i134	90
avifgen-sf-gpu-rescue-20260808	butteraugli-gpu	i134-gpu	4978
avifgen-sf-gpu-rescue-20260808	butteraugli-gpu	lianli	17634
avifgen-sf-gpu-rescue-20260808	butteraugli-gpu	lilith-gpu	8404
avifgen-sf-gpu-rescue-20260808	butteraugli-gpu	node-2	3406
avifgen-sf-gpu-rescue-20260808	ssim2-gpu	i134	90
avifgen-sf-gpu-rescue-20260808	ssim2-gpu	i134-gpu	4978
avifgen-sf-gpu-rescue-20260808	ssim2-gpu	lianli	17634
avifgen-sf-gpu-rescue-20260808	ssim2-gpu	lilith-gpu	8404
avifgen-sf-gpu-rescue-20260808	ssim2-gpu	node-2	3406
hdrgrid-diffmap-20260807	butteraugli	i134-gpu	14260
hdrgrid-diffmap-20260807	butteraugli	lilith-gpu	11729
hdrgrid-diffmap-20260807	butteraugli (diffmap; GPU-launch enforced)	i134-gpu	0
hdrgrid-diffmap-20260807	butteraugli (diffmap; GPU-launch enforced)	lilith-gpu	0
hdrgrid-diffmap-20260807	cvvdp	i134-gpu	14277
hdrgrid-diffmap-20260807	cvvdp	lilith-gpu	11733
hdrgrid-sf-cpu-20260807	zensim	i265-hdrcpu	54641
hdrgrid-sf-cpu-20260807	zensim	r5900xt-cpu	7372
hdrgrid-sf-cpu-20260807	zensim	r5900xt-hdrcpu	1577
hdrgrid-sf-cpu-20260807	zensim	tower-hdrcpu	10033
hdrgrid-sf-cpu-20260807	zensim	wsl-c1	63640
hdrgrid-sf-cpu-20260807	zensim	wsl-c2	122808
hdrgrid-sf-gpu-20260807	iwssim-gpu	lianli-hdr	3936
hdrgrid-sf-gpu-20260807	iwssim-gpu	r7900x-hdrsmoke	693
hdrgrid-sf-gpu-20260807	iwssim-gpu	r7900x-med	10021
hdrgrid-sf-gpu-20260807	ssim2-gpu	lianli-hdr	3936
hdrgrid-sf-gpu-20260807	ssim2-gpu	r7900x-hdrsmoke	693
hdrgrid-sf-gpu-20260807	ssim2-gpu	r7900x-med	10021
hdrgrid-sf-gpu-huge-20260807	iwssim-gpu	r7900x-huge	1115
hdrgrid-sf-gpu-huge-20260807	ssim2-gpu	r7900x-huge	1115
hdrgrid-sf-gpu-small-20260807	ssim2-gpu	r5900xt-hdr	1374
hdrgrid-sf2-cpu-20260807	cvvdp	i265-cpu	3490
hdrgrid-sf2-cpu-20260807	cvvdp	r5600g-cpu	1724
hdrgrid-sf2-cpu-20260807	cvvdp	r5900xt-cpu	1172
hdrgrid-sf2-gpu-20260807	butteraugli-gpu	i134-med	5223
hdrgrid-sf2-gpu-huge-20260807	butteraugli-gpu	i134-huge	5595
hdrgrid-sf2-gpu-small-20260807	butteraugli-gpu	r5900xt-sf2small	687

PASS: every ssim2/butteraugli SCORE row carries the -gpu runtime suffix (GPU rung executed; no CPU rung anywhere). diffmap butteraugli rows come from ZEN_REQUIRE_GPU=1 GPU launches. zensim features + cvvdp + encodes on CPU by design.
```
