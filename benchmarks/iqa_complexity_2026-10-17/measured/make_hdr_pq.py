# Timing-only HDR inputs: the zenmetrics-corpus pair (source.png vs q20.jpg), mirror-tiled to WxH, sRGB -> linear
# -> 203 cd/m^2 diffuse white -> SMPTE ST 2084 (PQ), written as 16-bit RGB PNG with a cICP chunk (BT.709 primaries,
# PQ transfer, RGB, full range). Not a colorimetric HDR rendition; it only gives the HDR route realistic content.
import sys, zlib, struct, numpy as np
from PIL import Image
W, H = int(sys.argv[1]), int(sys.argv[2]); out = sys.argv[3]
def tile(path):
    a = np.asarray(Image.open(path).convert('RGB')).astype(np.float64) / 255
    ys = np.array([i % (2*a.shape[0]) for i in range(H)]); ys = np.where(ys < a.shape[0], ys, 2*a.shape[0]-1-ys)
    xs = np.array([i % (2*a.shape[1]) for i in range(W)]); xs = np.where(xs < a.shape[1], xs, 2*a.shape[1]-1-xs)
    return a[ys][:, xs]
def pq(nits):
    y = np.clip(nits / 10000.0, 0, 1); m1, m2, c1, c2, c3 = 0.1593017578125, 78.84375, 0.8359375, 18.8515625, 18.6875
    return ((c1 + c2*y**m1) / (1 + c3*y**m1))**m2
def lin(v): return np.where(v <= 0.04045, v/12.92, ((v+0.055)/1.055)**2.4)
def png16(rgb, path):
    raw = b''.join(b'\x00' + row.astype('>u2').tobytes() for row in rgb)
    def chunk(t, d): return struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
    data = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', W, H, 16, 2, 0, 0, 0)) + chunk(b'cICP', bytes([1, 16, 0, 1])) \
        + chunk(b'IDAT', zlib.compress(raw, 6)) + chunk(b'IEND', b'')
    open(path, 'wb').write(data)
base = '/home/lilith/work/zen/zenmetrics--dash/crates/zenmetrics-corpus/data/'
for name, src in (('ref', 'source.png'), ('dist', 'q20.jpg')):
    png16(np.round(pq(lin(tile(base + src)) * 203.0) * 65535).astype(np.uint16), f'{out}/{name}_{W}x{H}.png')
