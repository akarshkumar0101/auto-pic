
from distutils.ccompiler import gen_lib_options
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
    def __init__(self, n_hidden=24, n_layers=8, n_channels=3, activation=torch.relu, features=['x', 'y', 'r'], cache_coordinates=True):
        super().__init__()
        
        self.features = features
        self.idx_features_to_use = []
        feature2idx = {'y': 0, 'x': 1, 'r': 2, 'theta': 3, 'ay': 4, 'ax': 5}
        for feature in self.features:
            self.idx_features_to_use.append(feature2idx[feature])

        self.cache_coordinates = {} if cache_coordinates else None
        
        n_units = [len(self.idx_features_to_use)] + [n_hidden]*n_layers + [n_channels]
        n_ins = np.array(n_units[:-1])
        n_outs = np.array(n_units[1:])
        
        # print(n_ins, n_outs)
        self.layers = nn.ModuleList([nn.Conv2d(n_in, n_out, kernel_size=1) 
                                     for n_in, n_out in zip(n_ins, n_outs)])
        # print(self.layers)
        
        self.activation = activation
        
    def forward(self, x):
        x = x[:, self.idx_features_to_use, :, :]

        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        x = self.layers[-1](x)
        x = x.sigmoid()
        return x

    def generate_input(self, img_size=(224, 224)):
        device = self.layers[0].weight.device

        if self.cache_coordinates is not None and img_size in self.cache_coordinates:
            return self.cache_coordinates[img_size].to(device)

        h, w = img_size
        y, x = torch.linspace(0, 1, h, device=device), torch.linspace(0, 1, w, device=device)
        y, x = torch.meshgrid(y, x, indexing='ij')
        
        r = (x.pow(2)+y.pow(2)).sqrt()
        theta = torch.atan2(x, y)
        ay, ax = y.abs(), x.abs()
        
        data = [y, x, r, theta, ay, ax]
        x = torch.stack(data, dim=0)[None] # 1, 6, h, w
        # normalize
        x = (x-x.mean(dim=(-1, -2), keepdim=True))/x.std(dim=(-1, -2), keepdim=True)

        if self.cache_coordinates is not None:
            self.cache_coordinates[img_size] = x

        return x

    def generate_image(self, img_size=(224, 224)):
        x = self.generate_input(img_size)
        return self(x)


class BatchImageCPPN(nn.Module):
    def __init__(self, n_batch=1, n_hidden=24, n_layers=8, n_channels=3, activation=torch.relu, give_radius=False, cache_coordinates=True):
        super().__init__()
        self.cppns = nn.ModuleList([ImageCPPN(n_hidden, n_layers, n_channels, activation, give_radius, cache_coordinates) for _ in range(n_batch)])

    def forward(self, x):
        return torch.cat([cppn(x) for cppn in self.cppns], dim=0)

    def generate_image(self, img_size=(224, 224)):
        x = self.cppns[0].generate_input(img_size)
        return self(x)

        