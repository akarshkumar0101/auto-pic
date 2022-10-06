
import matplotlib.pyplot as plt
from tqdm import tqdm

import numpy as np
import torch
from torchvision import transforms
import clip

from torch import nn

from image_cppn import ImageCPPN, BatchImageCPPN

augment_trans = transforms.Compose([
    transforms.RandomPerspective(distortion_scale=0.5, p=1, fill=1),
    transforms.RandomResizedCrop(224, scale=(0.7,0.9)),
])

mean = [0.48145466, 0.4578275, 0.40821073]
std = [0.26862954, 0.26130258, 0.27577711]

def augment_img(img, n_augments, augment='crops', keep_original=False):
    imgs = []

    if keep_original:
        imgs.append(img)
    
    if augment=='noise':
        while(len(imgs)<n_augments):
            imgs.append(img+1e-2*torch.randn_like(img))
    if augment=='crops':
        while(len(imgs)<n_augments):
            imgs.append(augment_trans(img))
    
    # return torch.stack(imgs, dim=0)
    return torch.stack(imgs, dim=1)

class ImageDOFs(nn.Module):
    def __init__(self, n_batch=1, n_channels=3, img_size=(224, 224)):
        super().__init__()
        self.dofs = nn.Parameter(1e-1*torch.randn(n_batch, n_channels, *img_size))

    def forward(self):
        return self.dofs.sigmoid()

    def generate_image(self, img_size=None):
        return self()

def optimize_imgs(clip_model, texts, imgs='pixel', n_iterations=100, img_size=(224, 224), lr=3e-2, device=None, callback=None):
    """
    clip_model should be should be the model returned by ```model, preprocess = clip.load("ViT-B/32", device=device)```

    texts should be List[str]
    imgs should be one of
        - 'pixel'
        - 'cppn'
        - torch.Tensor of shape (len(texts), 3, 224, 224)
        - BatchImageCPPN

    n_iterations should be int

    img_size should be (h, w)

    callback is called every iteration like 
    ```callback(i_iteration=i_iteration, texts=texts, imgs=imgs, imgs_augments=imgs_augments, imgs_features=imgs_features, texts_features=texts_features, dots=dots, loss=loss)```
    
    """

    clip_model = clip_model.to(device)

    with torch.no_grad():
        texts_tokens = clip.tokenize(texts).to(device)
        texts_features = clip_model.encode_text(texts_tokens).detach() # (batch, embed_dim)

    if imgs is None or imgs=='pixel':
        dofs = ImageDOFs(len(texts), 3, img_size).to(device)
    elif imgs=='cppn':
        dofs = BatchImageCPPN(len(texts), n_hidden=24, n_layers=8, n_channels=3, activation=torch.relu, cache_coordinates=True).to(device)
    elif isinstance(imgs, torch.Tensor):
        raise NotImplementedError
    elif isinstance(imgs, BatchImageCPPN):
        dofs = imgs.to(device)

    opt = torch.optim.Adam(dofs.parameters(), lr=lr, weight_decay=1e-6)

    for i_iteration in tqdm(range(n_iterations)):
        imgs = dofs.generate_image()

        imgs_augments = augment_img(imgs, 3, augment='crops') # (n_augs, batch, 3, h, w)
        shape_aug = imgs_augments.shape
        clip_input = imgs_augments.view(-1, *shape_aug[-3:]) *2.-1. # go from [0, 1] to [-1, 1]
        imgs_features = clip_model.encode_image(clip_input)
        imgs_features = imgs_features.view(*shape_aug[:2], -1) # (n_augs, batch, embed_dim)
        
        dots = torch.cosine_similarity(imgs_features, texts_features, dim=-1)
        loss = -dots.mean()
        
        opt.zero_grad()
        loss.backward()
        opt.step()

        if callback is not None:
            callback(i_iteration=i_iteration, texts=texts, imgs=imgs, imgs_augments=imgs_augments, 
                     imgs_features=imgs_features, texts_features=texts_features, dots=dots, loss=loss)

