
import numpy as np
import torch
from torch import nn

import matplotlib.pyplot as plt

def composite_activation(x):
    x = torch.atan(x)
    # Coefficients computed by:
    #   def rms(x):
    #     return np.sqrt((x*x).mean())
    #   a = np.arctan(np.random.normal(0.0, 1.0, 10**6))
    #   print(rms(a), rms(a*a))
    # print(x.shape)
    x = torch.concat([x/0.67, (x*x)/0.6], dim=1)
    # print(x.shape)
    # print()
    return x

# class CompositeActivation(nn.Module):
    # def forward():

class ImageCPPN(nn.Module):
    def __init__(self, n_hidden=24, n_layers=8, n_channels=3, activation=composite_activation):
        super().__init__()
        
        n_features = 2
        
        n_units = [n_features] + [n_hidden]*n_layers + [n_channels]
        n_ins = np.array(n_units[:-1])
        n_outs = np.array(n_units[1:])
        
        n_ins[1:] *= 2
        # print(n_ins, n_outs)
        self.layers = nn.ModuleList([nn.Conv2d(n_in, n_out, kernel_size=1) 
                                     for n_in, n_out in zip(n_ins, n_outs)])
        # print(self.layers)
        
        self.activation = activation
        
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        x = self.layers[-1](x)
        x = x.sigmoid()
        return x
    
    def generate_image(self, img_size):
        a = 3.0**0.5  # std(coord_range) == 1.0
        h, w = img_size
        
        y, x = torch.linspace(-a, a, h), torch.linspace(-a, a, w)
        y, x = torch.meshgrid(y, x, indexing='ij')
        
        # r = (x.pow(2)+y.pow(2)).sqrt()
        # theta = torch.atan2(x, y)
        
        # ay, ax = y.abs(), x.abs()
        
        # plt.colorbar()
        
        
        # x = tf.expand_dims(tf.stack([x, y], -1), 0)  # add batch dimension
        x = torch.stack([x, y], dim=0)[None]
        
        # plt.imshow(x[0, 1])
        # plt.colorbar()
        x = x.to(self.layers[0].weight.device)
        return self(x)