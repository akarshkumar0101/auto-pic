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
trans_norm = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
trans_norm = utils.get_multibatch_fn(trans_norm, 3)

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

cppn_params = dict(n_hidden=100, n_layers=20, activation=torch.tanh, normalization='coordinate', residual=False)
# cppn_params = dict(n_hidden=100, n_layers=20, activation=torch.relu, normalization='coordinate', residual=True)

def calc_pairwise_lpips_dist(net_lpips, imgs):
    """
    imgs.shape should be (b, L, 3, h, w)
    returns a tensor of (b, L, L) corresponding to pairwise distances in L
    """
    b, L = imgs.shape[:2]
    ans = torch.zeros(b, L, L, dtype=imgs.dtype, device=imgs.device)
    for i in range(L):
        for j in range(L):
            if i>j:
                ans[:, i, j] = net_lpips(imgs[:, i], imgs[:, j])[:, 0, 0, 0]
    ans = ans+ans.transpose(-1, -2) # make it symmetric
    return ans
    

def run_optimization(rep, lr, weight_decay, n_steps, n_augs, img_size, n_latents, coef_clip, coef_lpips, device, use_wandb, seed, config):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if use_wandb: wandb.init(config=config)
    
    img_size = (img_size, img_size)
    
    net_clip, preprocess = clip.load("RN50", device=device)
    net_lpips = lpips.LPIPS(net='alex').to(device)

    # texts = ["mountains with a waterfall", "cheetah eating a banana", "unicorn and rainbow"]
    # texts = ["A red apple", "A green tree", "The bright sun"]
    texts = ["A red dog", "A blue tree"]
    tokens_text = clip.tokenize(texts)
    bs = len(texts)
    
    latents = torch.randn(n_latents, 10).to(device)
    
    if rep=='cppn':
        dofs = ImageCPPN(n_batches=len(texts), **cppn_params, dim_latent=(10 if n_latents>0 else 0))
    else:
        dofs = PixelDOFs(n_batches=len(targets), img_size=img_size)

    tokens_text, dofs = tokens_text.to(device), dofs.to(device)
    net_clip, net_lpips = net_clip.to(device).eval(), net_lpips.to(device).eval()
    with torch.no_grad():
        features_texts = net_clip.encode_text(tokens_text).detach() # (b, embed_dim)
        
    opt = torch.optim.Adam(dofs.parameters(), lr=lr, weight_decay=weight_decay)
    
    encode_image = utils.get_multibatch_fn(net_clip.encode_image, 3)
    
    pbar = tqdm(range(n_steps))
    for i_step in pbar:
        imgs = torch.stack([dofs.generate_image(img_size, latent) for latent in latents], dim=1) # b, l, 3, h, w
        # imgs = dofs.generate_image(img_size) # b, 3, h, w
        imgs.retain_grad()
        imgs_resize = trans_resize(imgs) # b, l, 3, h, w
        imgs_aug = augment_img(imgs_resize, n_augs, augment='crops', keep_original=True) # (b, n_augs, l, 3, h, w)
        # features_imgs = utils.single_batch_map(net_clip.encode_image, imgs_aug, imgs_aug.shape[-3:]) # (b, n_augs, embed_dim)
        features_imgs = encode_image(trans_norm(imgs_aug)) # (b, n_augs, l, embed_dim)
        
        dots_clip = torch.cosine_similarity(features_imgs, features_texts[:, None, None], dim=-1) # (b, n_augs, l)
        dist_lpips = calc_pairwise_lpips_dist(net_lpips, trans_norm(imgs_resize)) # (b, l, l)
        loss_clip, loss_lpips = -dots_clip.mean(), -dist_lpips.mean()
        loss = coef_clip*loss_clip + coef_lpips*loss_lpips
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        # print(net_clip.visual.conv1.weight.grad.abs().mean().item())
        data_wandb = dict(loss=loss.item(), loss_clip=loss_clip.item(), loss_lpips=loss_lpips.item(), grad=imgs.grad.abs().mean().item())
        if use_wandb and i_step%(n_steps//200)==0:
            # imgs_resize.shape is # b, l, 3, h, w
            # fig, axs = plt.subplots(n_latents, bs, figsize=(3*bs, 3*n_latents))
            fig, axs = plt.subplots(bs, n_latents, figsize=(3*n_latents, 3*bs))
            for i in range(bs):
                for j in range(n_latents):
                    axs[i, j].imshow(imgs_resize[i, j].permute(1, 2, 0).detach().cpu().numpy())
                    if j==0:
                        axs[i, j].set_ylabel(texts[i])
                    if i==bs-1:
                        axs[i, j].set_xlabel(f'Latent vec #{j}')
            plt.tight_layout()
            data_wandb['img'] = fig
            
        pbar.set_postfix(**{k: v for k, v in data_wandb.items() if isinstance(v, float)})
        
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
parser.add_argument('--n_augs', type=int, default=4)
parser.add_argument('--img_size', type=int, default=32)
parser.add_argument('--n_latents', type=int, default=1)
parser.add_argument('--coef_clip', type=float, default=1)
parser.add_argument('--coef_lpips', type=float, default=0)

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--device', type=str, default='cuda:1')
parser.add_argument('--use_wandb', action=argparse.BooleanOptionalAction, default=False)

def main():
    args = parser.parse_args()
    print(args)
    
    run_optimization(**vars(args), config=vars(args))

if __name__=='__main__':
    main()
    
    
    