"""CAMBI ingress probe: libvmaf (pinned 3.2.1 build) `--feature cambi` on the distorted image of each pair,
with the PNG -> YUV conversion varied.
usage: cambi_probe.py <vmaf_bin> <pairs.tsv> <out.tsv> <scratch_dir> <variant>[,...]
variants: formula601   round(16+(65.481R+128.553G+24.966B)/255), even-cropped, 8-bit (earlier rounds)
          ff_<opts>    ffmpeg rawvideo, opts joined by '+': 420|444, 10 (yuv*p10le), acc (accurate_rnd),
                       709 (out_color_matrix=bt709), full (out_range=full)
          optional ':<cambi options>' suffix passed to --feature cambi=..."""
import csv, json, subprocess, sys, os, pathlib
import numpy as np
from PIL import Image
vmaf, pairs, out, scratch = sys.argv[1:5]
variants = sys.argv[5].split(',')
S = pathlib.Path(scratch); S.mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader(open(pairs), delimiter='\t'))
new = not os.path.exists(out)
f = open(out, 'a')
if new: f.write('variant\tdistorted\tscore\n')
def run_vmaf(yuv, w, h, fmt, bits, opts):
    feat = 'cambi' + (('=' + opts) if opts else '')
    tgt = S / 'out.json'
    r = subprocess.run([vmaf, '-r', str(yuv), '-d', str(yuv), '-w', str(w), '-h', str(h), '-p', fmt, '-b', str(bits),
                        '--feature', feat, '--no_prediction', '--json', '-o', str(tgt)], capture_output=True, text=True)
    if r.returncode: raise RuntimeError(r.stderr[-500:])
    m = json.load(tgt.open())['frames'][0]['metrics']
    return next(v for k, v in m.items() if k.startswith('cambi'))
for v in variants:
    base, _, opts = v.partition(':')
    for r in rows:
        img = Image.open(r['dist_path']).convert('RGB'); w, h = img.size
        yuv = S / 'in.yuv'
        if base == 'formula601':
            a = np.asarray(img, np.float64); h2, w2 = h - h % 2, w - w % 2; a = a[:h2, :w2]
            y = np.floor(16 + (65.481 * a[..., 0] + 128.553 * a[..., 1] + 24.966 * a[..., 2]) / 255 + .5).astype(np.uint8)
            yuv.write_bytes(y.tobytes() + bytes([128]) * (w2 * h2 // 2)); fmt, bits, W, H = '420', 8, w2, h2
        else:
            o = base[3:].split('+')
            chroma = '444' if '444' in o else '420'; bits = 10 if '10' in o else 8
            pix = f'yuv{chroma}p' + ('10le' if bits == 10 else '')
            vf = []
            sc = []
            if '709' in o: sc.append('out_color_matrix=bt709')
            if 'full' in o: sc.append('out_range=full')
            if sc: vf = ['-vf', 'scale=' + ':'.join(sc)]
            flags = ['-sws_flags', 'accurate_rnd+bitexact+full_chroma_int'] if 'acc' in o else []
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', r['dist_path'], *flags, *vf, '-pix_fmt', pix, '-f', 'rawvideo', str(yuv)], check=True)
            W, H = w, h
            if chroma == '420' and (w % 2 or h % 2): raise SystemExit('odd dims')
            fmt = chroma
        s = run_vmaf(yuv, W, H, fmt, bits, opts)
        f.write(f"{v}\t{r['distorted']}\t{s!r}\n"); f.flush()
