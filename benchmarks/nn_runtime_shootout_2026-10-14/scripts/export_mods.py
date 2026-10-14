import torch, torchvision
class VggLpips(torch.nn.Module):
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
def build(name):
    torch.manual_seed(0)
    return VggLpips() if name.startswith("vgg") else R50()
