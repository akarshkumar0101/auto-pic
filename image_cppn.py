import numpy as np
import torch
from torch import nn

import matplotlib.pyplot as plt

from collections import OrderedDict

class ImageCPPN(nn.Module):
    """
    n_batches are implemented using the `groups` feature of torch's Conv2D.
    This is *much* faster and more efficient than having multiple ImageCPPNs.
    """
    def __init__(self, n_hidden=24, n_layers=8, n_out_channels=3,
                 activation=torch.relu, activation_final=torch.sigmoid,
                 normalization='coordinate', residual=True,
                 features=['x', 'y', 'r'], dim_latent=0, n_batches=1):
        """
        two suggested configurations:
            - activation=torch.relu, normalization='coordinate', residual=True
            - activation=torch.tanh, normalization='coordinate', residual=False
        
        normalization can either be 'layer', 'layeraffine' or 'coordinate'
        
        Works for fitting images:
                 # activation=torch.tanh, n_batches=len(targets), normalization='coordinate', residual=False)
                 
                 
                 
        Some code to get started:
        
        cppn_params = dict(n_hidden=100, n_layers=20, activation=torch.tanh, normalization='coordinate', residual=False)
        dofs = ImageCPPN(n_batches=8, **cppn_params).to(device)
        imgs = dofs.generate_image((64, 64)) # returns tensor of shape 8, 3, 64, 64 (batch, 3, h, w)
        
        Some code w/ latent vector:
        
        latents = torch.randn(2, 10)
        cppn_params = dict(n_hidden=100, n_layers=20, activation=torch.tanh, normalization='coordinate', residual=False)
        dofs = ImageCPPN(n_batches=8, **cppn_params, dim_latent=10)
        imgs = torch.stack([dofs.generate_image((64, 64), latent) for latent in latents], dim=1) # 8, 2, 3, 64, 64 (batch, n_latents, 3, h, w)
        
        """
        super().__init__()
        
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.n_out_channels = n_out_channels
        self.activation = activation
        self.activation_final = activation_final
        self.normalization = normalization
        self.residual = residual
        self.features = features
        self.dim_latent = dim_latent
        self.n_batches = n_batches
        
        n_units = self.n_batches*np.array([len(self.features)+dim_latent] + [n_hidden]*n_layers + [self.n_out_channels])
        self.layers = nn.ModuleList([nn.Conv2d(n_in, n_out, kernel_size=1, groups=self.n_batches) 
                                     for n_in, n_out in zip(n_units[:-1], n_units[1:])])
        
        if self.normalization is not None and 'layer' in self.normalization:
            self.layers_norm = nn.ModuleList([nn.GroupNorm(n_batches, n_hidden*n_batches, affine='affine' in self.normalization)
                                              for _ in self.layers[:-1]])
        
    def normalize(self, lx, layer_idx):
    #     if self.normalization is not None:
    #         if 'coordinate' in self.normalization:
    #             dim = (-1, -2)
    #         elif 'layer' in self.normalization:
    #             dim = (-3)
    #         if 'divmean' in self.normalization:
    #             lx = lx/lx.mean(dim=dim, keepdim=True) # WHY TF DOES THIS WORK THE BEST????? with tanh at least
    #         elif 'divstd' in self.normalization:
    #             lx = lx/lx.std(dim=dim, keepdim=True)
    #         elif 'submean' in self.normalization:
    #             lx = lx - lx.mean(dim=dim, keepdim=True)
    #         else:
    #             lx = (lx - lx.mean(dim=dim, keepdim=True))/lx.std(dim=dim, keepdim=True)
        if isinstance(self.normalization, nn.ModuleList):
            lx = self.layers_norm[layer_idx](lx)
        elif self.normalization=='coordinate':
            lx = nn.functional.layer_norm(lx, normalized_shape=lx.shape[-2:])
        return lx
        
    def forward(self, x):
        for layer_idx, layer in enumerate(self.layers[:-1]):
            lx = layer(x)
            lx = self.activation(lx)
            if layer_idx>5:
                lx = self.normalize(lx, layer_idx)
            x = x+lx if self.residual and x.shape==lx.shape else lx
            
        x = self.layers[-1](x)
        x = self.activation_final(x)
        return x

    def generate_image(self, img_size=(224, 224), latent=None):
        h, w = img_size
        x = generate_input(img_size, self.n_batches, self.features, latent, device=self.layers[0].weight.device)
        assert (latent is None and self.dim_latent==0) or (latent.ndim==1 and len(latent)==self.dim_latent)
        # x is (1, n_batches*n_features, h, w) 
        x = self(x) # (1, n_batches*n_out_channels, h, w)
        x = x.reshape(self.n_batches, self.n_out_channels, *img_size) # (n_batches, n_out_channels, h, w)
        return x
    
    # def generate_image_incorrect(self, img_size=(224, 224)):
    #     x = self.generate_input(img_size) # (1, n_batches*n_features, h, w) 
    #     x = self(x) # (1, n_batches*n_out_channels, h, w)
    #     x = x.reshape(self.n_out_channels, self.n_batches, *img_size).permute(1, 0, 2, 3) # (n_batches, n_out_channels, h, w)
    #     return x
    
    def get_instance_state_dict(self, instance=None):
        """
        If instance is None:
            returns a list of state dicts corresponding to each CPPN in the batch.
        if instance==0:
            returns a state dict for CPPN at index 0.
        """
        data = OrderedDict([(key, value.chunk(self.n_batches, dim=0)) for key, value in self.state_dict().items()])
        state_dicts = [OrderedDict([(key, value[i]) for key, value in data.items()]) for i in range(self.n_batches)]
        return state_dicts if instance is None else state_dicts[instance]
    
    def load_instance_state_dict(self, state_dicts, instances=None):
        """
        state_dicts should be a dictionary or list of dictionaries corresponding to different CPPNs
        If instances is None:
            loads each of the state dicts into each of the CPPNs
        if instance==0:
            loads the input state dict into CPPN #0.
        """
        data = OrderedDict([(key, value.chunk(self.n_batches, dim=0)) for key, value in self.state_dict().items()])
        if not isinstance(state_dicts, list):
            state_dicts = [state_dicts]
        if instances is None:
            instances = np.arange(len(state_dicts))
        if isinstance(instances, int):
            instances = [instances]
        for i, state_dict in zip(instances, state_dicts):
            for key in data:
                data[key][i].data[...] = state_dict[key]
    
    def get_instance_cppn(self, instance=None):
        """
        Get a fully functional batch_size=1 ImageCPPN from this batched ImageCPPN at index=instance.
        """
        state_dicts = self.get_instance_state_dict(instance)
        
        ImageCPPN(self.n_hidden, self.n_layers, self.n_out_channels,
                  self.activation, self.activation_final,
                  self.normalization, self.residual,
                  self.features, n_batches=1
                 )

# class BatchImageCPPN(nn.Module):
#     def __init__(self, cppns=None, n_batch=1, **kwargs):
#         super().__init__()
#         if cppns is None:
#             self.cppns = nn.ModuleList([ImageCPPN(**kwargs) for _ in range(n_batch)])
#         else:
#             self.cppns = cppns

#     def forward(self, x):
#         return torch.cat([cppn(x) for cppn in self.cppns], dim=0)

#     def generate_image(self, img_size=(224, 224)):
#         x = self.cppns[0].generate_input(img_size)
        # return self(x)

# cache = None
# def generate_input(img_size=(224, 224), n_batches=1, features=['x', 'y', 'r'], device='cpu'):
#     inputs = (img_size, n_batches, features, device)
#     if cache is not None and inputs in cache:
#         x =  cache[inputs]
#     else:
#         h, w = img_size
#         y, x = torch.linspace(-1, 1, h, device=device), torch.linspace(-1, 1, w, device=device)
#         y, x = torch.meshgrid(y, x, indexing='ij')

#         r = (x.pow(2)+y.pow(2)).sqrt()
#         theta = torch.atan2(x, y)
#         absy, absx = y.abs(), x.abs()
#         feature2data = {'y': y, 'x': x, 'r': r, 'theta': theta, 'absy': absy, 'absx': absx}
#         x = torch.stack([feature2data[key] for key in features], dim=0)[None] # (1, n_features, h, w)

#         # normalize
#         x = (x-x.mean(dim=(-1, -2), keepdim=True))/x.std(dim=(-1, -2), keepdim=True)
        
#         x = x.repeat(1, n_batches, 1, 1) # (1, n_batches*n_features, h, w)
#         x = x.to(device)
        
#         if cache is not None:
#             cache[inputs] = x

#     return x

def generate_input(img_size=(224, 224), n_batches=1, features=['x', 'y', 'r'], latent=None, device='cpu'):
    h, w = img_size
    y, x = torch.linspace(-1, 1, h, device=device), torch.linspace(-1, 1, w, device=device)
    y, x = torch.meshgrid(y, x, indexing='ij')

    r = (x.pow(2)+y.pow(2)).sqrt()
    theta = torch.atan2(x, y)
    absy, absx = y.abs(), x.abs()
    feature2data = {'y': y, 'x': x, 'r': r, 'theta': theta, 'absy': absy, 'absx': absx}
    x = torch.stack([feature2data[key] for key in features], dim=0)[None] # (1, n_features, h, w)

    # normalize
    x = (x-x.mean(dim=(-1, -2), keepdim=True))/x.std(dim=(-1, -2), keepdim=True)
    
    if latent is not None:
        latent = latent[None, :, None, None].repeat(1, 1, h, w)
        x = torch.cat([x, latent], dim=-3)
    
    x = x.repeat(1, n_batches, 1, 1) # (1, n_batches*n_features, h, w)
    x = x.to(device)

    return x


class PixelDOFs(nn.Module):
    def __init__(self, n_out_channels=3, n_batches=3, img_size=(224, 224)):
        super().__init__()
        self.dofs = nn.Parameter(1e-1*torch.randn(n_batches, n_out_channels, *img_size))

    def forward(self):
        return self.dofs.sigmoid()

    def generate_image(self, img_size=None):
        return self()



if __name__=='__main__':
    import wandb
    
    for x in range(10):
        run = wandb.init(reinit=True)
        for y in range (100):
            wandb.log({"metric": x+y})
        run.finish()