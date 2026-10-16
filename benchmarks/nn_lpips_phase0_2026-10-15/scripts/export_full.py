"""Whole LPIPS-vgg (upstream lpips.LPIPS forward: scaling, trunk x2, head) as one
ONNX graph r,d -> scalar, for engines that should keep everything on device."""
import sys, os, torch, lpips
out = sys.argv[1]
m = lpips.LPIPS(net='vgg', version='0.1', verbose=False).eval()
class Full(torch.nn.Module):
    def __init__(s): super().__init__(); s.m = m
    def forward(s, r, d): return s.m(r, d).reshape(())
x = torch.rand(1, 3, 64, 64) * 2 - 1
torch.onnx.export(Full().eval(), (x, x.clone()), os.path.join(out, 'lpips_vgg_full.onnx'), opset_version=17, dynamo=False,
                  input_names=['r', 'd'], output_names=['lpips'], dynamic_axes={'r': {2: 'h', 3: 'w'}, 'd': {2: 'h', 3: 'w'}})
print('ok')
