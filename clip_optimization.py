import random

import numpy as np

import torch
from torch import nn
from torchvision import transforms

import matplotlib.pyplot as plt

import clip
import lpips

import wandb

from tqdm import tqdm

from image_cppn import ImageCPPN, PixelDOFs
import utils

trans_aug = transforms.Compose([
    transforms.RandomPerspective(distortion_scale=0.5, p=1., fill=0.),
    transforms.RandomResizedCrop(224, scale=(0.7,0.9)),
])
trans_aug = utils.get_multibatch_fn(trans_aug, 3)

trans_resize = transforms.Resize((224, 224))
trans_resize = utils.get_multibatch_fn(trans_resize, 3)

def augment_img(img, n_augments, augment='crops', keep_original=False):
    """
    img is of shape (b, 3, h, w)
    returns img of shape (b, n_augs, 3, h, w)
    """
    imgs = []
    if keep_original:
        imgs.append(img)
    if augment=='noise':
        while(len(imgs)<n_augments):
            imgs.append(img+1e-2*torch.randn_like(img))
    if augment=='crops':
        while(len(imgs)<n_augments):
            imgs.append(trans_aug(img))
    return torch.stack(imgs, dim=1)

# TODO add normalization preprocess trans before CLIP

cppn_params = dict(n_hidden=100, n_layers=20, activation=torch.tanh, normalization='coordinate', residual=False)
# cppn_params = dict(n_hidden=100, n_layers=20, activation=torch.relu, normalization='coordinate', residual=True)

def run_optimization(rep, lr, weight_decay, n_steps, n_augs, img_size, device, use_wandb, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if use_wandb: wandb.init()
    
    img_size = (img_size, img_size)
    
    net_clip, preprocess = clip.load("RN50", device=device)
    net_lpips = lpips.LPIPS(net='alex').to(device)

    # texts = ["mountains with a waterfall", "cheetah eating a banana", "unicorn and rainbow"]
    # texts = ["A red apple", "A green tree", "The bright sun"]
    texts = ["A red dog", "A blue tree"]
    tokens_text = clip.tokenize(texts)
    
    if rep=='cppn':
        dofs = ImageCPPN(n_batches=len(texts), **cppn_params)
    else:
        dofs = PixelDOFs(n_batches=len(targets), img_size=img_size)

    tokens_text, dofs = tokens_text.to(device), dofs.to(device)
    net_clip, net_lpips = net_clip.to(device).eval(), net_lpips.to(device).eval()
    with torch.no_grad():
        features_texts = net_clip.encode_text(tokens_text).detach() # (b, embed_dim)
        
    opt = torch.optim.Adam(dofs.parameters(), lr=lr, weight_decay=weight_decay)
    
    pbar = tqdm(range(n_steps))
    for i_step in pbar:
        imgs = dofs.generate_image(img_size)
        imgs.retain_grad()
        imgs_resize = trans_resize(imgs)
        imgs_aug = augment_img(imgs_resize, n_augs, augment='crops', keep_original=True) # (b, n_augs, 3, h, w)
        features_imgs = utils.single_batch_map(net_clip.encode_image, imgs_aug, imgs_aug.shape[-3:]) # (b, n_augs, embed_dim)
        dots = torch.cosine_similarity(features_imgs, features_texts[:, None], dim=-1)
        loss = -dots.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        pbar.set_postfix(loss=loss.item())
        
        # print(net_clip.visual.conv1.weight.grad.abs().mean().item())
        
        data_wandb = dict(loss=loss.item(), grad=imgs.grad.abs().mean().item())
        if use_wandb and i_step%(n_steps//20)==0:
            # imgs.shape is b, 3, 224, 224
            # imgs_aug.shape is b, n_aug, 3, 224, 224
            fig, axs = plt.subplots(2, len(imgs_aug), figsize=(3*len(imgs_aug), 3*2))
            for i in range(len(imgs_aug)):
                axs[0, i].imshow(imgs[i].permute(1, 2, 0).detach().cpu().numpy())
                axs[1, i].imshow(imgs_resize[i].permute(1, 2, 0).detach().cpu().numpy())
            plt.tight_layout()
            data_wandb['img'] = fig
            
        if use_wandb: wandb.log(data_wandb)
        plt.clf()

    # def callback(i_iteration, texts, imgs, imgs_aug, imgs_features, texts_features, dots, loss):
    #     if i_iteration%200==0:
    #         dots_img = dots.mean(dim=-1).tolist()
    #         plt.figure(figsize=(5*len(imgs), 5))
    #         for i_img, img in enumerate(imgs):
    #             plt.subplot(1, len(imgs), i_img+1)
    #             plt.title(texts[i_img] + f', match: {dots_img[i_img]:.4f}')
    #             plt.imshow(to_np(img.permute(1, 2, 0)))
    #         plt.show()
    if use_wandb: wandb.finish()

import argparse

parser = argparse.ArgumentParser(description='Run CLIP Optimization')
parser.add_argument('--rep', type=str, default='cppn')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--n_steps', type=int, default=1000)
parser.add_argument('--n_augs', type=int, default=8)
parser.add_argument('--img_size', type=int, default=32)

parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--use_wandb', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--seed', type=int, default=0)

def main():
    args = parser.parse_args()
    print(args)
    
    run_optimization(**vars(args))

if __name__=='__main__':
    main()
    
    
    