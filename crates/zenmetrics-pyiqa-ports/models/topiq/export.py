"""Export TOPIQ-FR (pyiqa CFANet, ResNet50, KADID and PIPAL checkpoints) for zenmetrics-pyiqa-ports (src/topiq.rs).

usage: export.py PAIRS_TSV OUT_DIR [VARIANT...]

VARIANT is `topiq_fr` (pyiqa `topiq_fr`, cfanet_fr_kadid_res50) and/or `topiq_fr_pipal`
(pyiqa `topiq_fr-pipal`, cfanet_fr_pipal_res50); default both. Run with a venv that has
torch 2.14.1+cpu and pyiqa 0.1.16, with TORCH_HOME pointing at its weight cache.

Writes, into OUT_DIR, per variant <col>:
  <col>_trunk.onnx    x[1,3,H,W] in [0,1] -> tap0..tap4: ImageNet mean/std normalisation and the
                      timm resnet50 features_only stages (act1, layer1..layer4), BN folded.
  <col>_head.onnx     d0..d4, r0..r4 (the distorted and reference taps), ph0..ph4 [t_i, H_i],
                      pw0..pw4 [W_i, u_i] (adaptive-average-pool matrices, identity where pyiqa
                      does not pool), pos0..pos4 [t_i*u_i, 1, 256] (positional embedding
                      bicubic-resized to the pooled size) -> `score` [1, 1]: the rest of
                      CFANet.forward_cross_attention, per level coarse to fine:
                      diff = sqrt((d-r)^2 + 1e-12); f = cat(d, r, diff) * weight_pool(diff);
                      f = ph @ f @ pw; dim_reduce; tokens + pos; self-attention; then cross-scale
                      attention, attention pooling, token mean and the MLP. dist_to_mos is x*1.
  <col>_posemb.f32    pos_emb [256, 32, 32] (h_emb repeated along w, then w_emb repeated along h),
                      little-endian f32: the host resizes it (torch bicubic, a = -0.75,
                      align_corners=False, no antialias; rten has no cubic Resize).
  topiq.meta          `channels 64,256,512,1024,2048` and `min_size n`.
  topiq_goldens.tsv   distorted, <col>... scored by pyiqa itself (metric(dist_path, ref_path)).
  topiq_export.meta   versions, host and sha256 of every output.
The script also replays every golden pair through onnxruntime (CPU) with numpy-built pooling
matrices and torch-resized embeddings, and prints the worst |onnx - pyiqa|.
"""
import csv, hashlib, os, platform, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import pyiqa, onnxruntime as ort

PAIRS, OUT = sys.argv[1], sys.argv[2]
VARIANTS = {'topiq_fr': 'topiq_fr', 'topiq_fr_pipal': 'topiq_fr-pipal'}
cols = sys.argv[3:] or list(VARIANTS)
os.makedirs(OUT, exist_ok=True)
torch.set_grad_enabled(False)
pairs = list(csv.DictReader(open(PAIRS), delimiter='\t'))


class Trunk(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return tuple(self.net.semantic_model(self.net.preprocess(x)))


def mha(m, q, kv):
    """nn.MultiheadAttention (batch 1, eval, no mask) as F.multi_head_attention_forward computes
    it, written with -1 reshapes: the traced module bakes the token count into its Reshapes."""
    e, nh = m.embed_dim, m.num_heads
    hd = e // nh
    w, b = m.in_proj_weight, m.in_proj_bias
    if q is kv:
        qp, kp, vp = F.linear(q, w, b).chunk(3, dim=-1)
    else:
        qp = F.linear(q, w[:e], b[:e])
        kp, vp = F.linear(kv, w[e:], b[e:]).chunk(2, dim=-1)
    qp = qp.reshape(-1, nh, hd).transpose(0, 1) * (1.0 / hd) ** 0.5
    kp = kp.reshape(-1, nh, hd).transpose(0, 1)
    vp = vp.reshape(-1, nh, hd).transpose(0, 1)
    o = torch.matmul(torch.softmax(torch.matmul(qp, kp.transpose(1, 2)), dim=-1), vp)
    o = F.linear(o.transpose(0, 1).reshape(-1, e), m.out_proj.weight, m.out_proj.bias)
    return o.reshape(-1, 1, e)


def enc_layer(ly, src):
    s2 = ly.norm1(src)
    src = src + mha(ly.self_attn, s2, s2)
    return src + ly.linear2(ly.activation(ly.linear1(ly.norm2(src))))


def dec_layer(ly, tgt, memory):
    memory = ly.norm2(memory)
    tgt = tgt + mha(ly.multihead_attn, ly.norm1(tgt), memory)
    return tgt + ly.linear2(ly.activation(ly.linear1(ly.norm3(tgt))))


class Head(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, *a):
        n = self.net
        d, r, ph, pw, pos = a[0:5], a[5:10], a[10:15], a[15:20], a[20:25]
        tokens = []
        for i in reversed(range(5)):
            diff = n.dist_func(d[i], r[i])
            f = torch.cat([d[i], r[i], diff], dim=1) * n.weight_pool[i](diff)
            f = torch.matmul(torch.matmul(ph[i], f), pw[i])
            f = n.dim_reduce[i](f).flatten(2).permute(2, 0, 1) + pos[i]
            for ly in n.sa_attn_blks[i].layers:
                f = enc_layer(ly, f)
            tokens.append(f)
        q = tokens[0]
        for i in range(4):
            for ly in n.attn_blks[i].layers:
                q = dec_layer(ly, q, tokens[i + 1])
        return n.score_linear(enc_layer(n.attn_pool, q).mean(dim=0))


def pool_matrix(n_in, n_out):
    """torch adaptive_avg_pool window [floor(i*n/o), ceil((i+1)*n/o)) as an [o, n] matrix."""
    m = np.zeros((n_out, n_in), np.float32)
    for i in range(n_out):
        s, e = (i * n_in) // n_out, -((-(i + 1) * n_in) // n_out)
        m[i, s:e] = np.float32(1.0) / np.float32(e - s)
    return m


def head_aux(net, taps):
    """The pooling matrices and resized embeddings for one tap geometry (as the Rust host builds)."""
    th, tw = taps[4].shape[2:]
    pos_emb = torch.cat((net.h_emb.repeat(1, 1, 1, net.w_emb.shape[3]),
                         net.w_emb.repeat(1, 1, net.h_emb.shape[2], 1)), dim=1)
    ph, pw, pos = [], [], []
    for t in taps:
        h, w = t.shape[2:]
        pooled = h > th and w > tw
        oh, ow = (th, tw) if pooled else (h, w)
        ph.append(pool_matrix(h, oh) if pooled else np.eye(h, dtype=np.float32))
        pw.append((pool_matrix(w, ow) if pooled else np.eye(w, dtype=np.float32)).T.copy())
        e = F.interpolate(pos_emb, size=(oh, ow), mode='bicubic', align_corners=False)
        pos.append(e.flatten(2).permute(2, 0, 1).contiguous().numpy())
    return ph, pw, pos


def load(p):
    return pyiqa.utils.img_util.imread2tensor(p, rgb=True).unsqueeze(0)


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


goldens = {}
for col in cols:
    metric = pyiqa.create_metric(VARIANTS[col], device='cpu', as_loss=False)
    net = metric.net.eval()
    assert net.crops == 1 and net.test_img_size is None and net.num_class == 1
    trunk, head = Trunk(net).eval(), Head(net).eval()
    x = torch.rand(1, 3, 96, 80)
    tp = os.path.join(OUT, f'{col}_trunk.onnx')
    torch.onnx.export(trunk, (x,), tp, opset_version=17, dynamo=False, do_constant_folding=True,
                      input_names=['x'], output_names=[f'tap{i}' for i in range(5)],
                      dynamic_axes={'x': {2: 'h', 3: 'w'},
                                    **{f'tap{i}': {2: f'h{i}', 3: f'w{i}'} for i in range(5)}})
    taps = trunk(x)
    ph, pw, pos = head_aux(net, taps)
    args = tuple(taps) + tuple(trunk(torch.rand(1, 3, 96, 80))) + tuple(
        torch.from_numpy(v) for v in ph + pw + pos)
    names = [f'd{i}' for i in range(5)] + [f'r{i}' for i in range(5)] + [f'ph{i}' for i in range(5)] + \
        [f'pw{i}' for i in range(5)] + [f'pos{i}' for i in range(5)]
    dyn = {}
    for i in range(5):
        dyn[f'd{i}'] = dyn[f'r{i}'] = {2: f'h{i}', 3: f'w{i}'}
        dyn[f'ph{i}'] = {0: f't{i}', 1: f'h{i}'}
        dyn[f'pw{i}'] = {0: f'w{i}', 1: f'u{i}'}
        dyn[f'pos{i}'] = {0: f'n{i}'}
    hp = os.path.join(OUT, f'{col}_head.onnx')
    torch.onnx.export(head, args, hp, opset_version=17, dynamo=False, do_constant_folding=True,
                      input_names=names, output_names=['score'], dynamic_axes=dyn)
    pos_emb = torch.cat((net.h_emb.repeat(1, 1, 1, net.w_emb.shape[3]),
                         net.w_emb.repeat(1, 1, net.h_emb.shape[2], 1)), dim=1)
    assert pos_emb.shape == (1, 256, 32, 32)
    pp = os.path.join(OUT, f'{col}_posemb.f32')
    pos_emb.reshape(-1).numpy().astype('<f4').tofile(pp)

    so = ort.SessionOptions()
    so.intra_op_num_threads = int(os.environ.get('OMP_NUM_THREADS', '8'))
    st = ort.InferenceSession(tp, so, providers=['CPUExecutionProvider'])
    sh = ort.InferenceSession(hp, so, providers=['CPUExecutionProvider'])
    worst = 0.0
    for p in pairs:
        dist, ref = load(p['dist_path']), load(p['ref_path'])
        s = float(metric(p['dist_path'], p['ref_path']).flatten()[0])
        goldens.setdefault(p['distorted'], {})[col] = s
        td = st.run(None, {'x': dist.numpy()})
        tr = st.run(None, {'x': ref.numpy()})
        ph, pw, pos = head_aux(net, [torch.from_numpy(t) for t in td])
        o = float(sh.run(None, dict(zip(names, td + tr + ph + pw + pos)))[0].flatten()[0])
        worst = max(worst, abs(o - s))
        print(col, p['distorted'], f'{s:.9g}', f'onnx {o:.9g}', f'{o - s:+.3g}', flush=True)
    print(col, 'worst |onnx - pyiqa|', worst, flush=True)


def ok(n):
    try:
        t = Trunk(pyiqa.create_metric('topiq_fr', device='cpu').net.eval())(torch.zeros(1, 3, n, n))
        return all(x.shape[2] > 0 and x.shape[3] > 0 for x in t)
    except RuntimeError:
        return False


min_size = next(n for n in range(1, 257) if ok(n))
with open(os.path.join(OUT, 'topiq.meta'), 'w') as f:
    f.write(f'channels 64,256,512,1024,2048\nmin_size {min_size}\n')
# Merge into an existing goldens file, so variants can be exported in separate runs.
gp = os.path.join(OUT, 'topiq_goldens.tsv')
if os.path.exists(gp):
    for r in csv.DictReader(open(gp), delimiter='\t'):
        for c in VARIANTS:
            if c in r and c not in cols:
                goldens.setdefault(r['distorted'], {})[c] = float(r[c])
all_cols = [c for c in VARIANTS if all(c in goldens.get(p['distorted'], {}) for p in pairs)]
with open(gp, 'w') as g:
    g.write('distorted\t' + '\t'.join(all_cols) + '\n')
    for p in pairs:
        g.write(p['distorted'] + '\t' + '\t'.join(repr(goldens[p['distorted']][c]) for c in all_cols) + '\n')
outputs = sorted(f for f in os.listdir(OUT) if f.startswith('topiq') and not f.endswith('_export.meta'))
with open(os.path.join(OUT, 'topiq_export.meta'), 'w') as f:
    f.write(f'torch {torch.__version__}\npyiqa {pyiqa.__version__}\nonnxruntime {ort.__version__}\n'
            f'python {platform.python_version()}\nhost {platform.node()}\npairs {PAIRS} {sha(PAIRS)}\n')
    for fn in outputs:
        f.write(f'sha256 {sha(os.path.join(OUT, fn))} {fn}\n')
print('done', cols, 'min_size', min_size)
