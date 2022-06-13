
import matplotlib.pyplot as plt
from tqdm import tqdm

import numpy as np
import torch
from torchvision import transforms
import clip

augment_trans = transforms.Compose([
    transforms.RandomPerspective(distortion_scale=0.5, p=1, fill=1),
    transforms.RandomResizedCrop(224, scale=(0.7,0.9)),
])

mean = [0.48145466, 0.4578275, 0.40821073]
std = [0.26862954, 0.26130258, 0.27577711]

def augment_img(img, n_augments, augment='noise'):
    imgs = [img]
    
    if augment=='noise':
        for i in range(n_augments):
            imgs.append(img+1e-2*torch.randn_like(img))
    if augment=='crops':
        for i in range(n_augments):
            imgs.append(augment_trans(img))
    
    return torch.stack(imgs, dim=0)

def imshow(img, mean=mean, std=std):
    """
    img.shape should be (3, h, w) on any device, torch tensor
    """
    img = img.detach().cpu()
    mean = torch.tensor(mean)
    std = torch.tensor(std)
    img = img.permute(1, 2, 0)
    img = img*std+mean
    img = img.clamp(0, 1).numpy()
    plt.imshow(img)

def optimize_imgs(model, texts, n_iterations=100, imgs=None, img_size=None, device='cpu', show_every_k=None):

    model = model.to(device)

    with torch.no_grad():
        texts_tokens = clip.tokenize(texts).to(device)
        texts_features = model.encode_text(texts_tokens)

    target = texts_features.detach()

    if imgs is None:
        imgs = 1e-2*torch.randn(len(target), 3, 224, 224)
    imgs = imgs.clone().to(device).requires_grad_()

    opt = torch.optim.Adam([imgs], lr=3e-2, weight_decay=1e-6)

    for i in tqdm(range(n_iterations)):
        imgs_augments = augment_img(imgs, 3, augment='crops')
        shape_aug = imgs_augments.shape
        imgs_features = model.encode_image(imgs_augments.view(-1, *shape_aug[-3:]))
        imgs_features = imgs_features.view(*shape_aug[:2], -1)
        
        dots = torch.cosine_similarity(imgs_features, target, dim=-1)
        loss = -dots.mean()
        
        opt.zero_grad()
        loss.backward()
        opt.step()

        if show_every_k is not None and i%show_every_k==0:
            dots_img = dots.mean(dim=0).tolist()
            plt.figure(figsize=(5*len(imgs), 5))
            for i_img, img in enumerate(imgs):
                plt.subplot(1, len(imgs), i_img+1)
                plt.title(texts[i_img] + f', match: {dots_img[i_img]:.4f}')
                imshow(img)
            plt.show()


