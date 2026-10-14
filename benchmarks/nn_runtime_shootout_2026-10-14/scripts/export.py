# Export random-init torchvision backbones (timing only) to ONNX.
import torch, torchvision, sys
out = sys.argv[1]
torch.manual_seed(0)
class VggLpips(torch.nn.Module):
    # LPIPS-vgg trunk: vgg16.features through relu5_3 (index 29), returns the 5 tapped relus
    def __init__(s):
        super().__init__(); s.f = torchvision.models.vgg16().features[:30].eval()
    def forward(s, x):
        outs = []
        for i, l in enumerate(s.f):
            x = l(x)
            if i in (3, 8, 15, 22, 29): outs.append(x)
        return tuple(outs)
class R50(torch.nn.Module):
    def __init__(s):
        super().__init__(); m = torchvision.models.resnet50().eval(); m.fc = torch.nn.Identity(); s.m = m
    def forward(s, x): return s.m(x)
for name, mod, hw in [("vgg16lpips_256", VggLpips(), 256), ("vgg16lpips_512", VggLpips(), 512), ("resnet50_224", R50(), 224), ("resnet50_384", R50(), 384)]:
    mod.eval()
    x = torch.rand(1, 3, hw, hw)
    torch.onnx.export(mod, (x,), f"{out}/{name}.onnx", opset_version=17, dynamo=False, do_constant_folding=True)
    print("exported", name)
