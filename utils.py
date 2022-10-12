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
    """
    Runs a single-dim batch fn with a multi-dim batch input.
    For example a function may only take (B, H, W), but what if you wanted to run with input (B1, B2, B3, H, W).
    It takes care of flattening the batch indices and unflatting them after.
    """
    shape = tuple(x.shape)
    assert shape[-len(instance_shape):] == instance_shape
    bs = shape[:-len(instance_shape)]
    x = x.reshape(-1, *instance_shape) # flatten batch indices
    # print(x.shape)
    x = fn(x, **kwargs)
    # print(x.shape)
    x = x.reshape(*bs, *x.shape[1:]) # unflatten batch indieces
    return x

def get_multibatch_fn(fn, ndims_instance):
    """
    Returns a multi-dim batch version of the single-dim batch fn.
    For example a function may only take (B, H, W), but what if you wanted to run with input (B1, B2, B3, H, W).
    It gives a function which automatically handles flattening the batch indices and unflatting them after.
    
    ndims_instance is the number of dimension in one instance (no batching).
    """
    def fn_multibatch(x, **kwargs):
        shape_batch, shape_instance = x.shape[:-ndims_instance], x.shape[-ndims_instance:]
        x = x.reshape(-1, *shape_instance) # flatten batch indices
        x = fn(x, **kwargs)
        x = x.reshape(*shape_batch, *x.shape[1:]) # unflatten batch indices
        return x
    return fn_multibatch


# def inverse_sigmoid():
    # x = ln(y/(1-y))