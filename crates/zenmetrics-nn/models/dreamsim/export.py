# DreamSim ensemble export (upstream ssundaram21/dreamsim, MIT).
#
# `python export.py <ladder_dir> <out_dir> <ckpt_dir>`
#
# The ensemble is three LoRA-finetuned ViT-B/16 extractors — DINO (cls
# token, 768d), OpenAI CLIP and OpenCLIP (each `visual @ proj`, 512d) —
# whose embeddings are concatenated to 1792d, mean-centred, L2-normed and
# cosine-compared: score = 1 - cos(e_a, e_b) on upstream's 224x224
# bicubic-resized input. LoRA (r=16, alpha=1, target `qkv`) is folded into
# the ViT weights with peft's merge_and_unload, verified below to give the
# identical embedding.
#
# Written files (in <out_dir>):
#  dreamsim_trunk.onnx     x01 [1,3,224,224] -> taps [1,768,1,1], [1,512,1,1],
#                    [1,512,1,1]; each branch normalises internally with its
#                    own stats (ImageNet for dino, CLIP for both clip nets).
#  dreamsim_full.onnx      (r01, d01) -> distance scalar, the whole pipeline.
#  dreamsim.meta           `channels 768,512,512`, `min_size 1`.
#  dreamsim_goldens.tsv    upstream distance on the ladder (CPU, fp32), under the
#                    upstream preprocess (PIL bicubic resize to 224x224).
#  dreamsim_export.meta    versions and sha256 of every output.
import hashlib, os, sys, platform
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image

LADDER, OUT, CKPT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
torch.set_grad_enabled(False)

from dreamsim import dreamsim  # pip package; checkpoint layout below
model, preprocess = dreamsim(pretrained=True, cache_dir=CKPT, device='cpu',
                             dreamsim_type='ensemble')
peft_score = model(preprocess(Image.new('RGB', (96, 96), (120, 60, 30))),
                   preprocess(Image.new('RGB', (96, 96), (200, 180, 150)))).item()
merged = model.merge_and_unload()          # base PerceptualModel, LoRA folded
merged_score = merged(preprocess(Image.new('RGB', (96, 96), (120, 60, 30))),
                      preprocess(Image.new('RGB', (96, 96), (200, 180, 150)))).item()
assert abs(merged_score - peft_score) < 1e-6, (merged_score, peft_score)

ext_d, ext_c, ext_o = (merged.extractor_list[i] for i in range(3))
IN_MEAN = (0.485, 0.456, 0.406); IN_STD = (0.229, 0.224, 0.225)
CL_MEAN = (0.48145466, 0.4578275, 0.40821073); CL_STD = (0.26862954, 0.26130258, 0.27577711)


def dino_cls(x):
    """extract_descriptors(layer=11) CLS: last block output, PRE final LN."""
    v = ext_d.model
    x = v.prepare_tokens(x)
    for blk in v.blocks:
        x = blk(x)
    return x[:, 0]                                # [B, 768]


def clip_embed(ext, x):
    """'embedding' feat: forward (normed CLS) @ proj."""
    return ext.model(x) @ ext.proj                # [B, 512]


def embeds(x01):
    """Per-branch normalisation then the three taps."""
    def norm(mean, std):
        t = x01 - torch.tensor(mean).view(1, 3, 1, 1)
        return t / torch.tensor(std).view(1, 3, 1, 1)
    e_d = dino_cls(norm(IN_MEAN, IN_STD))
    e_c = clip_embed(ext_c, norm(CL_MEAN, CL_STD))
    e_o = clip_embed(ext_o, norm(CL_MEAN, CL_STD))
    return torch.cat([e_d, e_c, e_o], dim=-1)     # [B, 1792]


def distance(a, b):
    """concat -> mean-centre -> L2-norm -> 1 - cosine (upstream forward)."""
    ea, eb = embeds(a), embeds(b)
    ea = ea - ea.mean(dim=-1, keepdim=True)
    eb = eb - eb.mean(dim=-1, keepdim=True)
    ea = ea / ea.norm(dim=-1, keepdim=True)
    eb = eb / eb.norm(dim=-1, keepdim=True)
    return 1 - (ea * eb).sum(dim=-1)


class Trunk(torch.nn.Module):
    def forward(self, x01):
        def norm(mean, std):
            return (x01 - torch.tensor(mean).view(1, 3, 1, 1)) / \
                torch.tensor(std).view(1, 3, 1, 1)
        e_d = dino_cls(norm(IN_MEAN, IN_STD))
        e_c = clip_embed(ext_c, norm(CL_MEAN, CL_STD))
        e_o = clip_embed(ext_o, norm(CL_MEAN, CL_STD))
        return (e_d.reshape(1, -1, 1, 1), e_c.reshape(1, -1, 1, 1),
                e_o.reshape(1, -1, 1, 1))


class Full(torch.nn.Module):
    def forward(self, r01, d01):
        return distance(r01, d01).reshape(())


x = torch.rand(1, 3, 224, 224)
trunk, full = Trunk().eval(), Full().eval()
g_d, g_c, g_o = trunk(x)
# trunk taps must equal the merged model's own embed
e = embeds(x)
assert float((torch.cat([g_d.flatten(), g_c.flatten(), g_o.flatten()]) - e).abs().max()) < 1e-6
# full graph == merged model distance
a = torch.rand(1, 3, 224, 224); b = torch.rand(1, 3, 224, 224)
assert abs(float(full(a, b)) - float(merged(a, b))) < 1e-6

for name, mod, args, inputs, outs in (
    ('dreamsim_trunk', trunk, (x,), ['x'], ['dino', 'clip', 'openclip']),
    ('dreamsim_full', full, (x, x.clone()), ['r', 'd'], ['dreamsim']),
):
    torch.onnx.export(mod, args, os.path.join(OUT, name + '.onnx'),
                      opset_version=17, dynamo=False, do_constant_folding=True,
                      input_names=inputs, output_names=outs)

# goldens: upstream distance on the ladder, under upstream's own preprocess
# (PIL RGB -> Resize(224,224, BICUBIC) -> ToTensor). Ladder is flat
# <src>_<size>_{ref,dist}.png; the TSV convention is `image\tsize\t<col>`.
rows = []
for f in sorted(os.listdir(LADDER)):
    if not f.endswith('_dist.png'):
        continue
    stem = f[:-9]
    ref_p = os.path.join(LADDER, stem + '_ref.png')
    dis_p = os.path.join(LADDER, f)
    src, size = stem.rsplit('_', 1)
    r, d = preprocess(Image.open(ref_p)), preprocess(Image.open(dis_p))
    rows.append((src, size, f'{float(merged(r, d)):.9e}'))

with open(os.path.join(OUT, 'dreamsim_goldens.tsv'), 'w') as f:
    f.write('image\tsize\tdreamsim\n')
    for r in rows:
        f.write('\t'.join(r) + '\n')

with open(os.path.join(OUT, 'dreamsim.meta'), 'w') as f:
    f.write('channels 768,512,512\nmin_size 1\n')

meta = {'python': platform.python_version(), 'torch': torch.__version__}
with open(os.path.join(OUT, 'dreamsim_export.meta'), 'w') as f:
    f.write(f'# dreamsim export {meta}\n')
    for n in sorted(os.listdir(OUT)):
        if n.endswith(('.onnx', '.f32', '.meta', '.tsv')):
            h = hashlib.sha256(open(os.path.join(OUT, n), 'rb').read()).hexdigest()
            f.write(f'{h}  {n}\n')
print('wrote', len(rows), 'goldens; peft vs merged delta:',
      abs(merged_score - peft_score))
