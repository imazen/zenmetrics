"""Run the jpeg-ai-qaf (0628a6b) HDR_VDP harness as a black-box oracle on AIC pairs.

usage: qaf_vdp.py <qaf_dir> <pairs.tsv> <out.tsv> [key=value ...]
keys: version=VDP2|VDP3 task=quality ppd=<float>|disp diag/w/h/dist eotf=pq|srgb|g22|none scale=<none multiplier> swap=1 stride=N
      refl=1|0 bits=10|8|0 peak=<nits for srgb> black=<nits> enc=rgb-bt.709 limit=N dtype=f32|f64
"""
import sys, csv, time, math
import numpy as np
qaf, pairs, out = sys.argv[1:4]
cfg = dict(version='VDP3', task='quality', w='3840', h='2160', diag='64.5', dist='1.32528',
           eotf='pq', refl='1', bits='10', peak='100', black='0', enc='rgb-bt.709', limit='0',
           dtype='f32', swap='0', stride='1', scale='1', ppd='', amb='100', rho='0.005', lmax='10000')
for kv in sys.argv[4:]:
    k, v = kv.split('=', 1); assert k in cfg, k; cfg[k] = v
sys.path.insert(0, qaf)
# oracle-side shims for 2023 code on current scipy/numpy (renames only, same math)
import scipy.integrate as _si
if not hasattr(_si, 'trapz'): _si.trapz = _si.trapezoid
if not hasattr(_si, 'cumtrapz'): _si.cumtrapz = _si.cumulative_trapezoid
if not hasattr(np, 'trapz'): np.trapz = np.trapezoid
from VDP2.hdrvdp2 import hdrvdp2
from VDP3.hdrvdp3 import hdrvdp3
from VDP3.pre_processing import hdrvdp_pix_per_deg, pq2lin
from PIL import Image

dt = np.float32 if cfg['dtype'] == 'f32' else np.float64
def load(p):
    a = np.array(Image.open(p).convert('RGB')).astype(dt) / dt(255)
    b = int(cfg['bits'])
    if b: a = (np.round(a * dt((1 << b) - 1)) / dt((1 << b) - 1)).astype(dt)
    return a
ppd = float(cfg['ppd']) if cfg['ppd'] else hdrvdp_pix_per_deg(float(cfg['diag']), [float(cfg['w']), float(cfg['h'])], float(cfg['dist']))
L_refl = float(cfg['rho']) * float(cfg['amb']) / np.pi if cfg['refl'] == '1' else 0.0
L_max = float(cfg['lmax'])
def lum(a):
    if cfg['eotf'] == 'pq': L = pq2lin(a)
    elif cfg['eotf'] == 'srgb':
        lin = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
        L = (float(cfg['peak']) - float(cfg['black'])) * lin + float(cfg['black'])
    elif cfg['eotf'] == 'g22':
        L = (float(cfg['peak']) - float(cfg['black'])) * a ** 2.2 + float(cfg['black'])
    elif cfg['eotf'] == 'none': L = a * float(cfg['scale'])
    else: raise SystemExit(cfg['eotf'])
    return np.minimum(L + L_refl, L_max + L_refl)
rows = list(csv.DictReader(open(pairs), delimiter='\t'))
rows = rows[::int(cfg['stride'])]
lim = int(cfg['limit']) or len(rows)
with open(out, 'w') as f:
    f.write('distorted\tscore\tsec\n')
    for r in rows[:lim]:
        t0 = time.time()
        R, T = lum(load(r['ref_path'])), lum(load(r['dist_path']))
        if cfg['swap'] == '1': R, T = T, R
        if cfg['version'] == 'VDP2':
            s = hdrvdp2(T, R, cfg['enc'], ppd)[0]
        else:
            s = hdrvdp3(cfg['task'], T, R, cfg['enc'], ppd)['Q_JOD']
        s = float(np.asarray(s).ravel()[0])
        f.write(f"{r['distorted']}\t{s!r}\t{time.time()-t0:.1f}\n"); f.flush()
print('ppd', ppd, 'cfg', cfg, file=sys.stderr)
