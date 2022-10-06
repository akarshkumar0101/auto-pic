import torch
import numpy as np

def to_np(a):
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    elif isinstance(a, np.ndarray):
        return a
    else:
        return np.array(a)
    
def count_params(net):
    return np.sum([p.numel() for p in net.parameters()], dtype=int)

"""
alpha= 0.0 will take the mean.
alpha= 0.2 will be very close to max.
alpha=-0.2 will be very close to min.
"""
def smooth_max(x, alpha, dim=-1):
    # unstable version:
    # return (x*(alpha*x).exp()).sum()/((alpha*x).exp()).sum()
    return ((alpha*x).softmax(dim=dim)*x).sum(dim=dim)

def single_batch_map(fn, x, instance_shape, **kwargs):
    shape = tuple(x.shape)
    assert shape[-len(instance_shape):] == instance_shape
    bs = shape[:-len(instance_shape)]
    x = x.view(-1, *instance_shape) # flatten batch indices
    # print(x.shape)
    x = fn(x, **kwargs)
    # print(x.shape)
    x = x.view(*bs, *x.shape[1:]) # unflatten batch indieces
    return x