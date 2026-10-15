# DMM export (upstream Baoliang93/DMM, MIT).
#
# `python export.py <ladder_dir> <out_dir> <dmm_repo>`
#
# DMM (Debiased Mapping, IEEE TIP 2025) is a VGG16 feature metric: MaxPool2d
# is replaced by L2Pool2d (3x3 Hann-window RMS pooling), taps relu3_3 and
# relu4_3, and the head unfolds 16x16 patches at stride 4 and compares the
# per-patch SVDs (singular-value distance weighted by base-angle
# consistency, times an SSIM-ish channel-mean gate) — the head has no ONNX
# op, so it is Rust; the graph is the L2-pooled VGG trunk only.
#
# Written files (in <out_dir>):
#   dmm_trunk.onnx     x01 [1,3,H,W] -> relu3_3 [1,256,H/4,W/4], relu4_3
#                    [1,512,H/8,W/8] (dynamic H,W; L2 pooling folded in)
#   dmm.meta           `channels 256,512`, `min_size 128`
#   dmm_goldens.tsv    upstream DMM() distance on the ladder (CPU, fp32),
#                    under upstream's prepare_image_adt preprocess
#                    (min side >128 -> PIL bilinear short-side resize to
#                    max(floor(min/48)*32,128), then ToTensor -> [0,1]).
#   dmm_export.meta    versions and sha256 of every output.
import gc, hashlib, os, sys, platform
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image

LADDER, OUT, DMM_SRC = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, DMM_SRC)
torch.set_grad_enabled(False)

import torchvision
from DMM_PyTorch.DMM import DMM, L2Pool2d, prepare_image_adt

VGG = torchvision.models.vgg16(weights='IMAGENET1K_V1').eval()
for p in VGG.parameters():
    p.requires_grad_(False)


def l2_trunk():
    """vgg16.features[:23] (through relu4_3) with L2 pooling, returning the
    relu3_3 (index 15) and relu4_3 (index 22) taps — mirrors
    FeaturesExtractor(replace_pooling=True)."""
    feats = torchvision.models.vgg16(weights='IMAGENET1K_V1').features
    for p in feats.parameters():
        p.requires_grad_(False)
    layers = list(feats.children())[:23]
    layers = [L2Pool2d() if isinstance(m, torch.nn.MaxPool2d) else m
              for m in layers]

    class Trunk(torch.nn.Module):
        def forward(self, x):
            t3 = None
            for i, m in enumerate(layers):
                x = m(x)
                if i == 15:
                    t3 = x
            return t3, x
    return Trunk().eval()


trunk = l2_trunk()

# self-check: ONNX trunk taps must equal upstream FeaturesExtractor taps
x = torch.rand(1, 3, 300, 400)
ref_dmm = DMM().eval()
ref_taps = ref_dmm.features_extractor(x)
my_t3, my_t4 = trunk(x)
assert float((my_t3 - ref_taps['relu3_3']).abs().max()) < 1e-6
assert float((my_t4 - ref_taps['relu4_3']).abs().max()) < 1e-6

# self-check: upstream score from MY trunk taps == upstream DMM() score
# (re-run the unfold+SVD head on the trunk taps manually)
def head_score(t_r, t_d):
    """replicates DMM.forward's per-tap branch for a pair of tap dicts."""
    unfold = torch.nn.Unfold(kernel_size=16, stride=4)
    c1 = c2 = 1e-6
    dist = 0.
    for key in ('relu3_3', 'relu4_3'):
        td, tp = t_d[key], t_r[key]
        a, b, hh, ww = td.size()
        k = 16
        # identical math to upstream, chunked over channels AND unfolded
        # per channel slice (a whole-tap unfold is (a, b·k·k, pt) — 7 GB
        # on relu3_3 of a 4096-prepared pair) so the batch stays ~1 GB
        diff_s = td.new_zeros(a)
        for c0 in range(0, b, 32):
            c1x = min(c0 + 32, b)
            dpc = unfold(td[:, c0:c1x]).view(a, c1x - c0, k, k, -1) \
                .transpose(2, 4).contiguous().view(-1, k, k)
            ppc = unfold(tp[:, c0:c1x]).view(a, c1x - c0, k, k, -1) \
                .transpose(2, 4).contiguous().view(-1, k, k)
            pt = dpc.numel() // (a * (c1x - c0) * k * k)
            u_r, s_r, _ = torch.svd(ppc)
            u_d, s_d, _ = torch.svd(dpc)
            s_r = s_r.view(a, c1x - c0, pt, k)
            s_d = s_d.view(a, c1x - c0, pt, k)
            d_s = (s_r - s_d) ** 2
            u_rd = (torch.abs(u_r.view(a, c1x - c0, pt, k, k)
                              * u_d.view(a, c1x - c0, pt, k, k))).sum(-1)
            wt = u_rd.std(-1) / (u_rd.mean(-1) + 1e-9)
            diff_s = diff_s + (d_s.sum(-1) * wt).mean(-1).sum(-1) / b
        xm = td.mean([2, 3], keepdim=True)
        ym = tp.mean([2, 3], keepdim=True)
        S1 = (2 * xm * ym + c1) / (xm ** 2 + ym ** 2 + c2)
        glb = torch.exp(-2.0 * S1.squeeze(-1).squeeze(-1).mean(1))
        dist = dist + diff_s * glb
    return dist


r, d = torch.rand(1, 3, 300, 400), torch.rand(1, 3, 300, 400)
ref_score = ref_dmm(r, d)
my_score = head_score(ref_dmm.features_extractor(r), ref_dmm.features_extractor(d))
assert abs(float(ref_score - my_score)) < 1e-5, (ref_score, my_score)

torch.onnx.export(
    trunk, (x,), os.path.join(OUT, 'dmm_trunk.onnx'),
    opset_version=17, dynamo=False, do_constant_folding=True,
    input_names=['x'], output_names=['t3', 't4'],
    dynamic_axes={'x': {2: 'h', 3: 'w'},
                  't3': {2: 'h4', 3: 'w4'}, 't4': {2: 'h8', 3: 'w8'}})

# goldens: full upstream DMM score on the ladder under prepare_image_adt.
# Upstream's forward materialises the whole unfold batch + (a,b,pt,16,16)
# u_rd products per call — ~14 GB on the 2048 pair, ~30 GB on 4096 — so
# rows past DIRECT_MAX are scored by head_score (the chunked replica
# asserted <1e-5 vs upstream above), which schedules the SAME upstream
# ops per channel block. Rows are tagged in dmm_export.meta.
DIRECT_MAX = 1200  # prepared short-side cap for the literal upstream call
GOLD = os.path.join(OUT, 'dmm_goldens.tsv')
# Torch's CPU caching allocator ratchets RSS across the big pairs (the
# multi-GB unfold buffers fragment instead of returning to the OS), so
# rows are appended as they finish and already-written stems skipped —
# the loop is resumable after a kill.
done = set()
if os.path.exists(GOLD):
    with open(GOLD) as f:
        next(f, None)
        done = {ln.split('\t')[0] + '_' + ln.split('\t')[1] for ln in f if ln.strip()}
new_rows = not done
with open(GOLD, 'a') as out:
    if new_rows:
        out.write('image\tsize\tdmm\n')
    for f in sorted(os.listdir(LADDER)):
        if not f.endswith('_dist.png'):
            continue
        stem = f[:-9]
        if stem in done:
            continue
        ref_p = os.path.join(LADDER, stem + '_ref.png')
        dis_p = os.path.join(LADDER, f)
        src, size = stem.rsplit('_', 1)
        rt = prepare_image_adt(Image.open(ref_p).convert('RGB'))
        dt = prepare_image_adt(Image.open(dis_p).convert('RGB'))
        if min(rt.shape[2], rt.shape[3]) < 128:
            # relu4_3 tap < one 16x16 patch — upstream's own forward
            # raises the same F.unfold error; the port refuses with
            # TooSmall. No golden exists upstream either.
            print(stem, 'skipped (prepared <128)', flush=True)
            continue
        if min(rt.shape[2], rt.shape[3]) <= DIRECT_MAX:
            s = float(ref_dmm(rt, dt))
        else:
            tr = ref_dmm.features_extractor(rt)
            td = ref_dmm.features_extractor(dt)
            s = float(head_score(tr, td))
        del rt, dt
        gc.collect()
        out.write(f'{src}\t{size}\t{s:.9e}\n')
        out.flush()
        print(stem, f'{s:.9e}', flush=True)

with open(os.path.join(OUT, 'dmm.meta'), 'w') as f:
    f.write('channels 256,512\nmin_size 128\n')

meta = {'python': platform.python_version(), 'torch': torch.__version__,
        'torchvision': torchvision.__version__}
# replica_rows is recomputed over the FINAL tsv, not just this run's
# rows: a row was chunked iff its prepared short side exceeds
# DIRECT_MAX, which is derivable from the source image's PIL header
# (prepare_image_adt resizes short -> max(min//48*32, 128)).
def _prep_short(path):
    w, h = Image.open(path).size
    m = min(w, h)
    return max(m // 48 * 32, 128) if m > 128 else m
replica_rows = []
with open(GOLD) as gf:
    next(gf, None)
    for ln in gf:
        if not ln.strip():
            continue
        stem = f'{ln.split(chr(9))[0]}_{ln.split(chr(9))[1]}'
        if _prep_short(os.path.join(LADDER, stem + '_ref.png')) > DIRECT_MAX:
            replica_rows.append(stem)
with open(os.path.join(OUT, 'dmm_export.meta'), 'w') as f:
    f.write(f'# dmm export {meta}\n')
    f.write(f'# goldens: upstream DMM() direct; chunked-replica rows '
            f'(same math, <1e-5 verified): {sorted(replica_rows)}\n')
    for n in sorted(os.listdir(OUT)):
        if n.endswith(('.onnx', '.meta', '.tsv')):
            h = hashlib.sha256(open(os.path.join(OUT, n), 'rb').read()).hexdigest()
            f.write(f'{h}  {n}\n')
print('goldens complete')
