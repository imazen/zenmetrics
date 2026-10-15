"""Export A-FINE (upstream ChrisDud0257/AFINE, Apache-2.0; "A-FINE: Adaptive
Fidelity-Naturalness Evaluation for AI-Generated Image Quality Assessment",
TMM 2024) for zenmetrics-nn.

usage: export.py LADDER_DIR OUT_DIR UPSTREAM_DIR AFINE_PTH CLIP_PTH [GOLDEN_SIZES]

UPSTREAM_DIR is a clone of github.com/ChrisDud0257/AFINE; AFINE_PTH is its
`afine.pth` (Google Drive link in upstream README). CLIP_PTH is OpenAI's
ViT-B-32.pt (URL in upstream's
QuickInference/afine.py). The checkpoint carries no stated licence -> the
Rust loader requires WeightsConsent.

Upstream's afine.py:
  1. normalizes each image with the CLIP stats (0.48145466, 0.4578275,
     0.40821073 / 0.26862954, 0.26130258, 0.27577711) on 0..1 RGB;
  2. upsamples the NORMALIZED tensor with F.interpolate(mode='bicubic',
     align_corners=False) to the next multiple of 32 in each axis that is
     not already a multiple (torch bicubic = cubic a=-0.75, 4 taps, index
     clamp at borders, no weight renormalization — verified below against
     F.interpolate impulse responses);
  3. runs the finetuned CLIP ViT-B/32 visual trunk; encode_image returns
     (ln_post(cls), [h[:, 1:, :] for the 12 transformer layers]) — 12 token
     feature maps [1, N, 768], N = H/32 * W/32;
  4. AFINEQhead (naturalness, NR): per feature — denormalized pixels plus
     relu(each tap) — take token-mean and token-variance, project each
     1536-dim mean||var vector to 128, concat with the 6-dim pixel stats,
     MLP(1542 -> 768 -> 1);
  5. AFINEDhead (fidelity, FR): per feature, SSIM-style structure/contrast
     terms between ref and dist token statistics, channel-weighted by
     softplus(alpha)/softplus(beta) normalized; score = 1 - (d1 + d2);
  6. NLM scalers map each term into (-2, 2); AFINELearnLambda combines
     (n_d, n_r, f) -> u = exp(k*(n_r - n_d)) * n_d + f = afine_all.

The published AIC-4 `A-FINE` column is `afine_all` (raw adapter output,
range -1.95..-0.66 on the crops; the 0..100 `scale_finalscore` wrapper is
NOT what was published). afine_all is asymmetric and lower-better.

The export therefore emits:
  afine_trunk.onnx   x_norm[1,3,H,W], pe[1,1,N+1,768] -> 12 feature maps
                     [1,768,N,1] (patch tokens laid out as 768-channel
                     planes of N). `pe` is the visual positional embedding
                     bicubic-resampled 50 -> N+1 on its sequence axis,
                     seq-major — computed on the HOST (afine.rs
                     resize_bicubic). Two constraints drive the odd input
                     contract: (a) rten's ONNX Resize has no cubic mode and
                     silently degrades it to linear, which drifts every
                     tap, so the real resample cannot live in the graph;
                     (b) the legacy exporter freezes the token axis to its
                     trace-time length unless some tensor-size-derived op
                     produces it, so the graph still runs pe through
                     F.interpolate(size=(x.shape[1],768)) — a runtime
                     IDENTITY (N+1 -> N+1 is exact under every kernel) that
                     keeps the sequence dim symbolic on every engine.
  afine_full.onnx    r_norm, d_norm [1,3,H,W] -> afine_all scalar — the
                     whole upstream pipeline for whole-graph engines (ONNX
                     Runtime executes the pos-emb cubic Resize correctly).
  afine_heads.f32    the head weights for the Rust implementation of steps
                     4-6, in this order (all little-endian f32):
                       proj_feat.weight  [128, 1536] row-major
                       proj_feat.bias    [128]
                       proj_head.0.weight [768, 1542] row-major
                       proj_head.0.bias  [768]
                       proj_head.2.weight [1, 768]
                       proj_head.2.bias  [1]
                       alpha             [9219]   (3 + 12*768)
                       beta              [9219]
                       nlm_nr            [2] (yita3, yita4)
                       nlm_fr            [2] (yita3, yita4)
                       adapter           [1] (k)
                       pos_emb           [50, 768] seq-major — the raw
                                         visual.positional_embedding;
                                         resize_bicubic resamples it as a
                                         single-channel 50x768 plane.
  afine.meta         `taps 12`, `channels 768`, `min_size n`.
  afine_goldens.tsv  upstream afine_all on the ladder (CPU, fp32), under
                     the upstream prep each side applies first.
  afine_export.meta  versions and sha256 of every output.
"""
import hashlib, os, sys, platform
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image

LADDER, OUT, UPSTREAM, AFINE_PTH, CLIP_PTH = sys.argv[1:6]
GOLDEN_SIZES = [int(s) for s in (sys.argv[6] if len(sys.argv) > 6 else '64,128,256,512,1024').split(',')]
sys.path.insert(0, os.path.join(UPSTREAM, 'QuickInference'))
# clip/__init__.py pulls in the tokenizer (ftfy); model.py needs only torch,
# so load it standalone — identical to clip.load(CLIP_PTH, jit=False).
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location(
    'clip_model', os.path.join(UPSTREAM, 'QuickInference', 'CLIP_ReturnFea', 'clip', 'model.py'))
clip_model_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(clip_model_mod)
from utils.network_archs import (  # noqa: E402  (upstream heads)
    AFINEQhead, AFINEDhead, AFINENLM_NR_Fit, AFINENLM_FR_Fit_with_limit, AFINELearnLambda,
)

os.makedirs(OUT, exist_ok=True)
torch.set_grad_enabled(False)

clip_model = clip_model_mod.build_model(torch.jit.load(CLIP_PTH, map_location='cpu').state_dict()).float()
ck = torch.load(AFINE_PTH, map_location='cpu', weights_only=False)
clip_model.load_state_dict(ck['finetuned_clip'])
clip_model.eval().requires_grad_(False)

qhead = AFINEQhead(); qhead.load_state_dict(ck['natural']); qhead.eval()
dhead = AFINEDhead(); dhead.load_state_dict(ck['fidelity']); dhead.eval()
scale_fr = AFINENLM_FR_Fit_with_limit(yita1=2, yita2=-2, yita3=0.5, yita4=0.15,
                                      yita3_upper=0.95, yita3_lower=0.05,
                                      yita4_upper=0.70, yita4_lower=0.01)
scale_fr.load_state_dict(ck['fidelity_scale']); scale_fr.eval()
scale_nr = AFINENLM_NR_Fit(yita1=2, yita2=-2, yita3=4.9592, yita4=21.5968)
scale_nr.load_state_dict(ck['natural_scale']); scale_nr.eval()
adapter = AFINELearnLambda(k=5); adapter.load_state_dict(ck['adapter']); adapter.eval()
for mod in (qhead, dhead, scale_fr, scale_nr, adapter):
    mod.requires_grad_(False)

MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)


def prep(t):
    """Upstream preprocessing: CLIP-normalize the 0..1 RGB image, then
    F.interpolate bicubic-upsample to the next multiple of 32 on each axis
    that needs it."""
    t = torch.stack([(t[i] - m) / s for i, (m, s) in enumerate(zip(MEAN, STD))])
    _, h, w = t.shape
    nh = h + (32 - h % 32) % 32
    nw = w + (32 - w % 32) % 32
    if nh != h or nw != w:
        t = F.interpolate(t.unsqueeze(0), size=(nh, nw), mode='bicubic',
                          align_corners=False).squeeze(0)
    return t.unsqueeze(0)


def score(r_norm, d_norm):
    """The composed upstream pipeline on already-prepped inputs."""
    _, f_d = clip_model.encode_image(d_norm)
    _, f_r = clip_model.encode_image(r_norm)
    n_d = qhead(d_norm, f_d)
    n_r = qhead(r_norm, f_r)
    fid = dhead(d_norm, r_norm, f_d, f_r)
    return adapter(scale_nr(n_d), scale_nr(n_r), scale_fr(fid))


def pe_torch(h, w):
    """The visual positional embedding resampled to h/32*w/32+1 tokens,
    exactly as upstream's visual.forward does it. Returned as the `pe`
    trunk input layout [1, 1, n+1, 768]."""
    n_tok = (h // 32) * (w // 32) + 1
    v = clip_model.visual
    pe = v.positional_embedding.unsqueeze(0).unsqueeze(0)  # [1,1,50,768]
    return F.interpolate(pe, size=(n_tok, pe.shape[3]), mode='bicubic',
                         align_corners=False)  # [1,1,n_tok,768]


class Trunk(torch.nn.Module):
    """visual.forward with the pos-embedding resample input-externalized:
    `pe` (second input, [1,1,N+1,768]) is the already-resized embedding;
    the graph still interpolates it to x.shape[1] so the token axis stays
    symbolic — an identity resize at runtime."""

    def __init__(self):
        super().__init__()
        self.visual = clip_model.visual

    def forward(self, x, pe):
        v = self.visual
        x = v.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        x = torch.cat([
            v.class_embedding.to(x.dtype)
            + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x], dim=1)
        # Same shape contract as upstream's forward: pe arrives already
        # resampled, but is still passed through interpolate with a
        # tensor-derived size so the exported graph keeps a symbolic token
        # axis. At runtime the resize is N+1 -> N+1, an exact identity on
        # every engine (incl. rten, which has no cubic mode).
        positional_embedding = F.interpolate(
            pe, size=(x.shape[1], pe.shape[3]), mode='bicubic',
            align_corners=False)
        positional_embedding = positional_embedding.squeeze(0).squeeze(0)
        x = x + positional_embedding.to(x.dtype)
        x = v.ln_pre(x)
        x = x.permute(1, 0, 2)
        _, feats = v.transformer(x)
        return [f.permute(0, 2, 1).unsqueeze(3) for f in feats]


class Full(torch.nn.Module):
    """(r_norm, d_norm) -> afine_all, the whole upstream pipeline. The NLM
    scalers' `exp_pow >= 10` python-if is expressed branch-free (torch.where)
    so the traced ONNX graph is exact for every input, not just the branch
    the trace happened to take. Both expressions equal 4/(1+e^e)-2 as
    e -> inf, so they coincide at the switch and thereafter."""

    def forward(self, r, d):
        _, f_d = clip_model.encode_image(d)
        _, f_r = clip_model.encode_image(r)
        n_d = qhead(d, f_d)
        n_r = qhead(r, f_r)
        fid = dhead(d, r, f_d, f_r)

        def nlm(x, y3, y4, lo3, hi3, lo4, hi4):
            y3 = torch.clamp(y3, lo3, hi3)
            y4 = torch.clamp(y4, lo4, hi4)
            e = -1 * (x - y3) / (torch.abs(y4) + 1e-10)
            a = 4 * torch.exp(-e) / (1 + torch.exp(-e)) - 2
            b = 4 / (1 + torch.exp(e)) - 2
            return torch.where(e >= 10, a, b)

        sn_d = nlm(n_d, scale_nr.yita3, scale_nr.yita4, -1e30, 1e30, -1e30, 1e30)
        sn_r = nlm(n_r, scale_nr.yita3, scale_nr.yita4, -1e30, 1e30, -1e30, 1e30)
        sf = nlm(fid, scale_fr.yita3, scale_fr.yita4,
                 scale_fr.yita3_lower, scale_fr.yita3_upper,
                 scale_fr.yita4_lower, scale_fr.yita4_upper)
        k_ = F.softplus(adapter.k)
        return (torch.exp(k_ * (sn_r - sn_d)) * sn_d + sf).reshape(())


x = torch.rand(1, 3, 64, 96) * 2 - 1  # already-normalized dummy input
trunk, full = Trunk().eval(), Full().eval()
feats = trunk(x, pe_torch(64, 96))
assert len(feats) == 12 and feats[0].shape[1] == 768 and feats[0].shape[2] == 2 * 3

# The pe-fed trunk must reproduce encode_image's taps at more than the
# trace size (the pos-emb resample depends on the runtime token count).
for (hh, ww) in ((64, 96), (96, 128)):
    xt = torch.rand(1, 3, hh, ww) * 2 - 1
    _, ref_feats = clip_model.encode_image(xt)
    got = trunk(xt, pe_torch(hh, ww))
    for g, r in zip(got, ref_feats):
        assert float((g.permute(0, 2, 1, 3).squeeze(-1) - r).abs().max()) <= 1e-5

for name, mod, args, inputs, outs, dyn in (
    ('afine_trunk', trunk, (x, pe_torch(64, 96)), ['x', 'pe'],
     [f'f{i}' for i in range(12)],
     {'x': {2: 'h', 3: 'w'}, 'pe': {2: 'n_tok'}}),
    ('afine_full', full, (x, x.clone()), ['r', 'd'], ['afine'],
     {'r': {2: 'h', 3: 'w'}, 'd': {2: 'h', 3: 'w'}}),
):
    torch.onnx.export(mod, args, os.path.join(OUT, name + '.onnx'),
                      opset_version=17, dynamo=False, do_constant_folding=True,
                      input_names=inputs, output_names=outs, dynamic_axes=dyn)

# The composed path must equal the literal upstream forward.
with torch.no_grad():
    a, b = torch.rand(1, 3, 96, 96) * 2 - 1, torch.rand(1, 3, 96, 96) * 2 - 1
    _, fd = clip_model.encode_image(b)
    _, fr = clip_model.encode_image(a)
    ref = adapter(scale_nr(qhead(b, fd)), scale_nr(qhead(a, fr)),
                  scale_fr(dhead(b, a, fd, fr)))
    assert float((full(a, b) - ref).abs()) <= 1e-6


def ok(n):
    try:
        full(prep(torch.rand(3, n, n)), prep(torch.rand(3, n, n)))
        return True
    except RuntimeError:
        return False


min_size = next(n for n in range(1, 96) if ok(n) and all(ok(k) for k in range(n, n + 8)))
with open(os.path.join(OUT, 'afine.meta'), 'w') as f:
    f.write('taps 12\nchannels 768\nmin_size %d\n' % min_size)

# Head weights for the Rust implementation, in the documented order.
def w(f, t):
    f.write(np.asarray(t.detach().cpu().numpy(), dtype='<f4').tobytes())


with open(os.path.join(OUT, 'afine_heads.f32'), 'wb') as f:
    w(f, qhead.proj_feat.weight); w(f, qhead.proj_feat.bias)
    w(f, qhead.proj_head[0].weight); w(f, qhead.proj_head[0].bias)
    w(f, qhead.proj_head[2].weight); w(f, qhead.proj_head[2].bias)
    w(f, dhead.alpha.reshape(-1)); w(f, dhead.beta.reshape(-1))
    w(f, torch.stack([scale_nr.yita3.detach(), scale_nr.yita4.detach()]))
    w(f, torch.stack([scale_fr.yita3.detach(), scale_fr.yita4.detach()]))
    w(f, adapter.k.detach().reshape(1))
    # visual.positional_embedding [50, 768] seq-major: afine.rs resamples
    # it with resize_bicubic as a single-channel 50x768 plane -> [N+1,768]
    # for the trunk's pe input [1,1,N+1,768].
    w(f, clip_model.visual.positional_embedding.detach())


def load(p):
    a = np.asarray(Image.open(p).convert('RGB'), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)


with open(os.path.join(OUT, 'afine_goldens.tsv'), 'w') as g:
    g.write('image\tsize\tafine\n')
    for name in sorted(os.listdir(LADDER)):
        if not name.endswith('_ref.png'):
            continue
        stem, size = name.split('_')[0], int(name.split('_')[1])
        if size not in GOLDEN_SIZES:
            continue
        r, d = load(os.path.join(LADDER, name)), load(os.path.join(LADDER, name.replace('_ref', '_dist')))
        s = float(full(prep(r), prep(d)).item())
        g.write(f'{stem}\t{size}\t{s:.9g}\n')
        print('afine', stem, size, s, flush=True)


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


with open(os.path.join(OUT, 'afine_export.meta'), 'w') as f:
    f.write(f'torch {torch.__version__}\nChrisDud0257/AFINE (git, upstream QuickInference)\npython {platform.python_version()}\nhost {platform.node()}\n')
    for fn in ('afine_trunk.onnx', 'afine_full.onnx', 'afine_heads.f32',
               'afine.meta', 'afine_goldens.tsv'):
        f.write(f'sha256 {sha(os.path.join(OUT, fn))} {fn}\n')
print('done afine min_size', min_size)
