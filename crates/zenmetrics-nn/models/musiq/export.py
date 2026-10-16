"""Export MUSIQ (upstream google-research/google-research/musiq, Apache-2.0;
J. Ke, Q. Wang, Y. Wang, P. Milanfar, F. Yang, "MUSIQ: Multi-scale Image
Quality Transformer", ICCV 2021) for zenmetrics-nn.

usage: export.py CKPT_NPZ OUT_DIR GOLDENS_TSV LADDER_DIR

CKPT_NPZ is the official KonIQ checkpoint `koniq_ckpt.npz` (the
`opt/target/*` flax parameter tree) from the project's GCS bucket —
gs://gresearch/musiq/, linked from upstream's README ("Model Zoo").
GOLDENS_TSV, required, is `oracle.py`'s output and the ONNX graph is
checked against it.

The npz is mapped into a PyTorch mirror of upstream's model: a per-patch
5-layer weight-standardized ResNet stem + Dense(384) embedding, hashed
spatial and scale embeddings, a learned CLS token, 14 pre-norm transformer
blocks (6 heads, mlp 1152, tanh GELU), final LayerNorm and a scalar head.
Attention masking uses upstream's additive -1e10 padding bias, CLS marked
valid. Verified against the JAX oracle (jax 0.3.25 + flax 0.3.3, CPU fp32):
worst |d| 6.5e-5 over the 1415_256 pair.

Files written (all prefixed `musiq`):
  musiq.onnx        x[1,l,3075] in -> score `musiq`[1]; dynamic patch axis
                    `l` (49+144+ceil(h/32)*ceil(w/32) for the ladder).
  musiq.meta        `channels 1` and `min_size 1` (every input resizes).
  musiq_export.meta versions and sha256 of every output.

Preprocessing is NOT in the graph: `musiq_goldens.tsv` (the JAX oracle's
scores) is produced by `oracle.py` under the musiq39 venv — the legacy
jax/flax/tensorflow stack cannot coexist with this script's torch env.
"""
import hashlib
import os
import platform
import runpy
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if len(sys.argv) != 5:
    raise SystemExit('usage: export.py CKPT_NPZ OUT_DIR GOLDENS_TSV LADDER_DIR')
NPZ, OUT, GOLDENS, LADDER = sys.argv[1:]
checks = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'export_checks.py'))
GOLDEN_ROWS = checks['golden_rows'](GOLDENS, LADDER, 'musiq')
os.makedirs(OUT, exist_ok=True)
torch.set_grad_enabled(False)

f32 = np.float32
PATCH = 32
GRID = 10
LONGER = [224, 384]
HID = 384
NHEAD = 6
NLAYERS = 14
MLP = 1152


def gauss_spans(ni, no):
    """TF scale_and_translate spans for kernel_type='gaussian',
    antialias=False (tf.image.resize's GAUSSIAN default): kernel_scale=1,
    sigma=0.5, radius 1.5, weights exp(-2 d^2) for |d| < 1.5, normalised by
    reciprocal multiply. Mirrors src/resize.rs::resize_gaussian_tf_f32 (zenresize gaussian_tf)."""
    scale = f32(no) / f32(ni)
    inv = f32(1.0) / scale
    span_n = min(2 * int(np.ceil(f32(1.5))) + 1, ni)
    starts = np.zeros(no, np.int64)
    counts = np.zeros(no, np.int64)
    W = np.zeros((no, span_n), np.float32)
    for ox in range(no):
        sf = f32(ox + 0.5) * inv
        if sf < 0 or sf > ni:
            continue
        ss = max(0, min(ni - 1, int(np.ceil(sf - f32(1.5) - f32(0.5)))))
        se = max(0, min(ni - 1, int(np.floor(sf + f32(1.5) - f32(0.5))))) + 1
        tot = f32(0.0)
        tw = []
        for src in range(ss, se):
            kp = f32(src + 0.5) - sf
            w = f32(0.0) if abs(kp) >= f32(1.5) else f32(np.exp(f32(-2.0) * kp * kp))
            tot = f32(tot + w)
            tw.append(w)
        if abs(tot) >= 1000.0 * np.finfo(np.float32).tiny:
            oot = f32(1.0) / tot
            for i, w in enumerate(tw):
                W[ox, i] = f32(w * oot)
        starts[ox] = ss
        counts[ox] = se - ss
    return starts, counts, W


def resize_gaussian(img, oh, ow):
    """img [h,w,3] f32 -> [oh,ow,3]; vertical pass then horizontal."""
    h, w, _ = img.shape
    sy, cy, Wy = gauss_spans(h, oh)
    tmp = np.zeros((oh, w, 3), np.float32)
    for oy in range(oh):
        for i in range(int(cy[oy])):
            tmp[oy] = f32(tmp[oy] + Wy[oy, i] * img[int(sy[oy] + i)])
    sx, cx, Wx = gauss_spans(w, ow)
    out = np.zeros((oh, ow, 3), np.float32)
    for ox in range(ow):
        for i in range(int(cx[ox])):
            out[:, ox] = f32(out[:, ox] + Wx[ox, i] * tmp[:, int(sx[ox] + i)])
    return out


def hash_pos(count):
    """tf.compat.v1 resize NEAREST of arange(GRID): floor(i * (GRID/count))."""
    sc = f32(GRID) / f32(count)
    return np.array([min(int(np.floor(f32(i) * sc)), GRID - 1) for i in range(count)])


def extract(img, cap, scale_id):
    """extract_patches SAME + hash/scale/mask cols; zero-pad to cap rows."""
    h, w, _ = img.shape
    ch, cw = int(np.ceil(h / 32)), int(np.ceil(w / 32))
    pt, pl = (ch * 32 - h) // 2, (cw * 32 - w) // 2
    ph, pw = hash_pos(ch), hash_pos(cw)
    n = ch * cw
    rows = np.zeros((n, 32 * 32 * 3 + 3), np.float32)
    k = 0
    for dy in range(ch):
        for dx in range(cw):
            for py in range(32):
                sy = dy * 32 + py - pt
                if 0 <= sy < h:
                    for px in range(32):
                        sx = dx * 32 + px - pl
                        if 0 <= sx < w:
                            rows[k, (py * 32 + px) * 3:(py * 32 + px + 1) * 3] = img[sy, sx]
            rows[k, 3072] = f32(ph[dy] * GRID + pw[dx])
            rows[k, 3073] = f32(scale_id)
            rows[k, 3074] = f32(1.0)
            k += 1
    if cap >= 0:
        rows = np.concatenate([rows, np.zeros((cap, 3075), np.float32)])[:cap]
    return rows


def preprocess(img_u8):
    """Full get_multiscale_patches: u8 [h,w,3] -> [1,L,3075] f32."""
    h, w, _ = img_u8.shape
    t = img_u8.astype(np.float32) / f32(255.0)
    x = f32(-1.0) + t * f32(2.0)
    outs = []
    for scale_id, longer in enumerate(sorted(LONGER)):
        ratio = f32(longer) / f32(max(h, w))
        rh = int(np.rint(f32(h) * ratio))  # tf.round is ties-to-even
        rw = int(np.rint(f32(w) * ratio))
        assert rh > 0 and rw > 0, 'output dimensions must be positive'
        r = resize_gaussian(x, rh, rw)
        cap = int(np.ceil(longer / 32)) ** 2
        outs.append(extract(r, cap, scale_id))
    outs.append(extract(x, -1, len(LONGER)))
    return np.concatenate(outs, axis=0)[None]


def std_conv(w):
    """HWIO kernel -> OIHW standardized weight (flax StdConv: axes HWI,
    eps 1e-5)."""
    w = w.transpose(3, 2, 0, 1).astype(np.float32)
    m = w.mean(axis=(1, 2, 3), keepdims=True)
    s = w.std(axis=(1, 2, 3), keepdims=True)
    return torch.from_numpy((w - m) / (s + 1e-5))


def dense(k, b):
    """flax Dense kernel [in,out] -> torch Linear weight [out,in]."""
    lin = nn.Linear(k.shape[0], k.shape[1])
    with torch.no_grad():
        lin.weight.copy_(torch.from_numpy(np.asarray(k.T, np.float32)))
        lin.bias.copy_(torch.from_numpy(np.asarray(b, np.float32)))
    return lin


def conv(k, stride=1, padding=0):
    """flax StdConv HWIO -> Conv2d (bias-free), weight standardized."""
    c = nn.Conv2d(
        k.shape[2], k.shape[3], k.shape[:2], stride=stride, padding=padding,
        bias=False)
    with torch.no_grad():
        c.weight.copy_(std_conv(k))
    return c


def gn(scale, bias, eps):
    """flax GroupNorm (32 groups) scale/bias -> GroupNorm."""
    c = int(np.asarray(scale).size)
    g = nn.GroupNorm(32, c, eps=eps)
    with torch.no_grad():
        g.weight.copy_(torch.from_numpy(np.asarray(scale).reshape(c)))
        g.bias.copy_(torch.from_numpy(np.asarray(bias).reshape(c)))
    return g


def ln(scale, bias):
    l = nn.LayerNorm(HID, eps=1e-6)
    with torch.no_grad():
        l.weight.copy_(torch.from_numpy(np.asarray(scale, np.float32)))
        l.bias.copy_(torch.from_numpy(np.asarray(bias, np.float32)))
    return l


class Block(nn.Module):
    def __init__(self, t, i):
        super().__init__()
        p = f'opt/target/Transformer/encoderblock_{i}'
        self.ln0 = ln(t[f'{p}/LayerNorm_0/scale'], t[f'{p}/LayerNorm_0/bias'])
        a = f'{p}/MultiHeadDotProductAttention_1'
        self.qkv = nn.ModuleList()
        for nm in ('query', 'key', 'value'):
            k = t[f'{a}/{nm}/kernel'].reshape(HID, HID)
            self.qkv.append(dense(k, t[f'{a}/{nm}/bias'].reshape(HID)))
        self.out = dense(
            t[f'{a}/out/kernel'].reshape(HID, HID), t[f'{a}/out/bias'])
        self.ln2 = ln(t[f'{p}/LayerNorm_2/scale'], t[f'{p}/LayerNorm_2/bias'])
        m = f'{p}/MlpBlock_3'
        self.fc0 = dense(t[f'{m}/Dense_0/kernel'], t[f'{m}/Dense_0/bias'])
        self.fc1 = dense(t[f'{m}/Dense_1/kernel'], t[f'{m}/Dense_1/bias'])

    def forward(self, x, bias):
        y = self.ln0(x)
        B, T, _ = y.shape
        q, k, v = [
            proj(y).view(B, T, NHEAD, 64).transpose(1, 2) for proj in self.qkv]
        att = torch.softmax(
            (q / np.sqrt(64.0)).matmul(k.transpose(2, 3)) + bias, dim=-1)
        y = att.matmul(v).transpose(1, 2).reshape(B, T, HID)
        x = x + self.out(y)
        x = x + self.fc1(F.gelu(self.fc0(self.ln2(x)), approximate='tanh'))
        return x


class Musiq(nn.Module):
    def __init__(self, npz):
        super().__init__()
        t = {k: npz[k] for k in npz.files}
        self.conv_root = conv(t['opt/target/conv_root/kernel'], stride=2)
        self.gn_root = gn(
            t['opt/target/gn_root/scale'], t['opt/target/gn_root/bias'], 1e-6)
        b = 'opt/target/block1/unit1'
        self.conv1 = conv(t[f'{b}/conv1/kernel'])
        self.gn1 = gn(t[f'{b}/gn1/scale'], t[f'{b}/gn1/bias'], 1e-4)
        self.conv2 = conv(t[f'{b}/conv2/kernel'], padding=1)
        self.gn2 = gn(t[f'{b}/gn2/scale'], t[f'{b}/gn2/bias'], 1e-4)
        self.conv3 = conv(t[f'{b}/conv3/kernel'])
        self.gn3 = gn(t[f'{b}/gn3/scale'], t[f'{b}/gn3/bias'], 1e-4)
        self.conv_proj = conv(t[f'{b}/conv_proj/kernel'])
        self.gn_proj = gn(t[f'{b}/gn_proj/scale'], t[f'{b}/gn_proj/bias'], 1e-4)
        self.embedding = dense(
            t['opt/target/embedding/kernel'], t['opt/target/embedding/bias'])
        self.register_buffer('posemb', torch.from_numpy(
            np.asarray(t['opt/target/Transformer/posembed_input/pos_embedding'],
                       np.float32)))
        self.register_buffer('scaleemb', torch.from_numpy(
            np.asarray(t['opt/target/Transformer/scaleembed_input/scale_embedding'],
                       np.float32)))
        self.register_buffer('cls', torch.from_numpy(
            np.asarray(t['opt/target/Transformer/cls'], np.float32)))
        self.blocks = nn.ModuleList(
            [Block(t, i) for i in range(NLAYERS)])
        self.enc_norm = ln(
            t['opt/target/Transformer/encoder_norm/scale'],
            t['opt/target/Transformer/encoder_norm/bias'])
        self.head = dense(t['opt/target/head/kernel'], t['opt/target/head/bias'])

    def stem(self, p):
        """[L,3,32,32] NCHW -> [L,16384] in (h,w,c) flatten order."""
        p = self.conv_root(F.pad(p, (2, 3, 2, 3)))
        p = F.relu(self.gn_root(p))
        p = F.max_pool2d(F.pad(p, (0, 1, 0, 1), value=float('-inf')), 3, 2)
        res = self.gn_proj(self.conv_proj(p))
        x = F.relu(self.gn1(self.conv1(p)))
        x = F.relu(self.gn2(self.conv2(x)))
        x = self.gn3(self.conv3(x))
        x = F.relu(res + x)
        return x.permute(0, 2, 3, 1).reshape(x.shape[0], -1)

    def forward(self, x):
        """x [1,L,3075] -> score [1]."""
        sp = x[0, :, -3].long()
        sc = x[0, :, -2].long()
        mask = x[0, :, -1]
        L = x.shape[1]
        p = x[0, :, :-3].reshape(L, PATCH, PATCH, 3).permute(0, 3, 1, 2)
        t = self.embedding(self.stem(p.contiguous()))
        t = t + self.posemb[0, sp] + self.scaleemb[0, sc]
        t = torch.cat([self.cls, t.unsqueeze(0)], dim=1)
        mask = torch.cat(
            [torch.ones(1, 1, device=x.device), mask.unsqueeze(0)], 1)
        bias = (mask - 1.0) * 1e10
        bias = bias.view(1, 1, 1, -1)
        for blk in self.blocks:
            t = blk(t, bias)
        t = self.enc_norm(t)
        return self.head(t[:, 0]).reshape(1)


def load_rgb8(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert('RGB'))


npz = np.load(NPZ)
model = Musiq(npz).eval()

onnx_path = os.path.join(OUT, 'musiq.onnx')
dummy = torch.zeros(1, 257, 3075)
torch.onnx.export(
    model, (dummy,), onnx_path, opset_version=17, dynamo=False,
    do_constant_folding=True, input_names=['x'], output_names=['musiq'],
    dynamic_axes={'x': {1: 'l'}, 'musiq': {}})

# Self-check: ORT CPU vs the torch mirror on a real input.
import onnxruntime as ort  # noqa: E402

img = load_rgb8(os.path.join(os.path.dirname(OUT) or '.', 'check.png')) \
    if os.path.exists(os.path.join(os.path.dirname(OUT) or '.', 'check.png')) \
    else np.random.RandomState(0).randint(0, 256, (256, 256, 3), np.uint8)
x = torch.from_numpy(preprocess(img))
with torch.no_grad():
    want = float(model(x).item())
session_opts = ort.SessionOptions()
session_opts.intra_op_num_threads = int(os.environ.get('OMP_NUM_THREADS', '1'))
session_opts.inter_op_num_threads = 1
sess = ort.InferenceSession(onnx_path, sess_options=session_opts, providers=['CPUExecutionProvider'])
got = float(sess.run(
    ['musiq'], {'x': x.numpy()})[0][0])
print(f'self-check: torch {want:.9g} ort-cpu {got:.9g} |d| {abs(got - want):.3e}')
assert abs(got - want) <= 1e-4, (got, want)

# Required qualification against the upstream goldens; missing files and
# mismatches are errors rather than an optional printed diagnostic.
worst = 0.0
for row in GOLDEN_ROWS:
    for kind, key in (('ref', 'musiq_ref'), ('dist', 'musiq')):
        path = os.path.join(LADDER, f"{row['image']}_{row['size']}_{kind}.png")
        x = torch.from_numpy(preprocess(load_rgb8(path)))
        got_s = float(sess.run(['musiq'], {'x': x.numpy()})[0][0])
        worst = max(worst, checks['assert_parity']('musiq', got_s, float(row[key])))
print(f'goldens check: {len(GOLDEN_ROWS)} pairs, worst |d| {worst:.3e}')

with open(os.path.join(OUT, 'musiq.meta'), 'w') as f:
    f.write('channels 1\nmin_size 1\n')


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


with open(os.path.join(OUT, 'musiq_export.meta'), 'w') as f:
    f.write(f'torch {torch.__version__}\ngoogle-research/musiq @3ef5211\n'
            f'python {platform.python_version()}\nhost {platform.node()}\n'
            f'numpy {np.__version__}\n')
    f.write(f'ckpt_sha256 {sha(NPZ)}\n')
    for fn in ('musiq.onnx', 'musiq.meta'):
        f.write(f'sha256 {sha(os.path.join(OUT, fn))} {fn}\n')
print('done musiq')
