# Execution compatibility only. Official model source is not inspected.
import torch
import numpy as np
torch.set_num_threads(16)
np.float = float
np.int = int
_load = torch.load
def load(*args, **kwargs):
    kwargs.setdefault('map_location', 'cpu')
    kwargs.setdefault('weights_only', False)
    return _load(*args, **kwargs)
torch.load = load
torch.nn.Module.cuda = lambda self, *args, **kwargs: self
torch.Tensor.cuda = lambda self, *args, **kwargs: self
torch.cuda.set_device = lambda *args, **kwargs: None
_item = torch.Tensor.__getitem__
def item(self, key):
    if self.ndim == 0 and isinstance(key, int) and key == 0:
        return self
    return _item(self, key)
torch.Tensor.__getitem__ = item
_state = torch.nn.Module.load_state_dict
def state(self, incoming, *args, **kwargs):
    targets = self.state_dict()
    incoming = incoming.copy()
    for key, value in incoming.items():
        if key in targets and value.shape != targets[key].shape and value.numel() == targets[key].numel():
            incoming[key] = value.reshape(targets[key].shape)
    return _state(self, incoming, *args, **kwargs)
torch.nn.Module.load_state_dict = state
# Restore the inference intent of legacy Variable(..., volatile=True).
torch.set_grad_enabled(False)
# Explicit black-box configuration probe for the demo's missing global.
import os
if 'PIEAPP_LEGACY_NUM_PATCHES' in os.environ:
    import builtins
    builtins.num_patches = int(os.environ['PIEAPP_LEGACY_NUM_PATCHES'])
_view = torch.Tensor.view
def view(self, *shape):
    def integral(x):
        return int(x) if isinstance(x, float) and x.is_integer() else x
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(integral(x) for x in shape[0])
    else:
        shape = tuple(integral(x) for x in shape)
    return _view(self, *shape)
torch.Tensor.view = view
# Match the contiguous device copy in the original CUDA execution path.
torch.Tensor.cuda = lambda self, *args, **kwargs: self.contiguous()
