"""Check that ffmpeg's rgb24->yuv420p luma (swscale) equals the integer formula
t = (ry*R + gy*G + by*B + (32<<14) + (1<<8)) >> 9 ; Y = (t + 32) >> 6,
ry,gy,by = rint({0.299,0.587,0.114} * 219/255 * 2^15). usage: sws_luma_check.py <pairs.tsv>"""
import subprocess, sys, csv, numpy as np
from PIL import Image
S = 15; ry, gy, by = (int(np.rint(c * 219 / 255 * (1 << S))) for c in (0.299, 0.587, 0.114))
print('coefficients', ry, gy, by)
tot = bad = 0
for r in csv.DictReader(open(sys.argv[1]), delimiter='\t'):
    p = r['dist_path']; a = np.asarray(Image.open(p).convert('RGB')).astype(np.int64); h, w = a.shape[:2]
    y = np.frombuffer(subprocess.run(['ffmpeg', '-v', 'error', '-i', p, '-pix_fmt', 'yuv420p', '-f', 'rawvideo', '-'], capture_output=True, check=True).stdout, np.uint8)[:w * h].reshape(h, w)
    t = (ry * a[..., 0] + gy * a[..., 1] + by * a[..., 2] + (32 << (S - 1)) + (1 << (S - 7))) >> (S - 6)
    bad += int(((t + 32) >> 6 != y).sum()); tot += y.size
print('images', 'pixels', tot, 'mismatches', bad)
