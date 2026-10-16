"""Raw little-endian f32 dump of the LPIPS-vgg trunk for the burn spike:
scaling shift (3), scale (3), then 13 x (conv weight [co,ci,3,3], bias [co]) in
VGG16 order, then the 5 lin vectors. Also raw RGB8 of the ladder pairs."""
import sys, os, numpy as np, torch, lpips
from PIL import Image
out, ladder = sys.argv[1], sys.argv[2]
m = lpips.LPIPS(net='vgg', version='0.1', verbose=False).eval()
parts = [m.scaling_layer.shift.reshape(-1), m.scaling_layer.scale.reshape(-1)]
convs = [l for s in range(1, 6) for l in getattr(m.net, f'slice{s}') if isinstance(l, torch.nn.Conv2d)]
assert len(convs) == 13
for c in convs:
    parts += [c.weight.reshape(-1), c.bias.reshape(-1)]
parts += [l.model[-1].weight.reshape(-1) for l in m.lins]
arr = torch.cat([p.detach().float() for p in parts]).numpy().astype('<f4')
arr.tofile(os.path.join(out, 'lpips_vgg_all.f32'))
print('floats', arr.size, [tuple(c.weight.shape) for c in convs][:3])
for name in sorted(os.listdir(ladder)):
    if name.endswith('.png'):
        a = np.asarray(Image.open(os.path.join(ladder, name)).convert('RGB'))
        a.tofile(os.path.join(out, name[:-4] + '.rgb'))
