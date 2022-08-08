import os
import random
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import open3d as o3d

import numpy as np
import torch
torch.no_grad()
from torchvision.utils import save_image, make_grid

from draw import draw, draw_attention_open3d


os.environ['DISPLAY'] = ':0.0'


save_dir = 'images_attn'
experiment_name = 'shapenet15k-tooth/camera-ready'
summary_name = os.path.join('../checkpoints/gen/', experiment_name, 'summary.pth')
summary_train_name = os.path.join('../checkpoints/gen/', experiment_name, 'summary_train_recon.pth')

imgdir = os.path.join(save_dir, experiment_name)
imgdir_recon = os.path.join(imgdir, 'recon')
imgdir_gt = os.path.join(imgdir, 'gt')
imgdir_gen = os.path.join(imgdir, 'gen')
imgdir_gt_train = os.path.join(imgdir, 'gt_train')

os.makedirs(save_dir, exist_ok=True)
os.makedirs(imgdir_gt, exist_ok=True)
os.makedirs(imgdir_recon, exist_ok=True)
os.makedirs(imgdir_gen, exist_ok=True)
os.makedirs(imgdir_gt_train, exist_ok=True)


# validation and generated
summary = torch.load(summary_name)
for k, v in summary.items():
    try:
        print(f"{k}: {v.shape}")
    except AttributeError:
        print(f"{k}: {len(v)}")
len_att = len(summary['dec_att'])


# train
summary_train = torch.load(summary_train_name)
for k, v in summary_train.items():
    try:
        print(f"{k}: {v.shape}")
    except AttributeError:
        print(f"{k}: {len(v)}")
len_att_train = len(summary_train['dec_att'])


recon_targets = list(range(len(summary['gt_mask'])))[:]
gen_targets = list(range(len(summary['smp_mask'])))[:]

gt = summary['gt_set'][recon_targets]
gt_mask = summary['gt_mask'][recon_targets]

recon = summary['recon_set'][recon_targets]
recon_mask = summary['recon_mask'][recon_targets]

dec_att = [summary['dec_att'][l][:, :, recon_targets] for l in range(len_att)]
enc_att = [summary['enc_att'][l][:, :, recon_targets] for l in range(len_att)]

gen = summary['smp_set'][gen_targets]
gen_mask = summary['smp_mask'][gen_targets]
gen_att = [summary['smp_att'][l][:, :, gen_targets] for l in range(len_att)]


recon_targets_train = list(range(len(summary_train['gt_mask'])))[:]

gt_train = summary_train['gt_set'][recon_targets_train]
gt_mask_train = summary_train['gt_mask'][recon_targets_train]
enc_att_train = [summary_train['enc_att'][l][:, :, recon_targets_train] for l in range(len_att_train)]


def attention_selector(gt, gt_mask, att, lidx=0, projection=0, selected_heads=None, palette_permutation=None):
    if selected_heads is not None:
        att = [a[:, selected_heads].view(a.size(0), len(selected_heads), a.size(2), a.size(3), a.size(4)) for a in att]

    print(len(att), lidx, projection)
    print(len(att[lidx]))
    qwe = att[lidx][projection]
     
    return draw_attention_open3d(gt, gt_mask, qwe, color_opt='gist_rainbow', size=10, palette_permutation=palette_permutation)
    #return draw_attention_open3d(gt, gt_mask, att[lidx][projection], color_opt='gist_rainbow', size=10, palette_permutation=palette_permutation)

print("starting....")
#for topdown in tqdm(range(2,5)):
for topdown in range(2,5):
    print("start...")
    for projection in [0]:
        if topdown == 2:
            print("topdown == 2")
            selected_heads = [2]
            palette_permutation = [1, 0]
        elif topdown == 3:
            print("topdown == 3")
            selected_heads = [0]
            palette_permutation = [1, 2, 3, 0]
        elif topdown == 4:
            print("topdown == 4")
            selected_heads = [2]
            palette_permutation = None
        else:
            print("topdown else")
            selected_heads = list(range(enc_att[0].size(1)))
            palette_permutation = None
        print(f"HEAD: {selected_heads}, COLOR: {palette_permutation}")
        gt_imgs = attention_selector(gt, gt_mask, enc_att, len(enc_att) - 1 - topdown, projection,
                                     selected_heads=selected_heads, palette_permutation=palette_permutation)
        for head_idx in range(len(selected_heads)):
            for idx in range(len(recon_targets)):
                data_idx = recon_targets[idx]
                try:
                    pos_min = torch.nonzero(gt_imgs[idx][head_idx].mean(0) != 1).min(0)[0]
                    pos_max = torch.nonzero(gt_imgs[idx][head_idx].mean(0) != 1).max(0)[0]
                    gt_img = gt_imgs[idx][head_idx][:, pos_min[0]:pos_max[0]+1, pos_min[1]:pos_max[1]+1]
                except RuntimeError:
                    gt_img = gt_imgs[idx][head_idx]
                head = selected_heads[head_idx]
                save_image(gt_img, os.path.join(imgdir_gt, f'{topdown}_{projection}_{head}_{data_idx}.png'))
        del gt_imgs
print('gt DONE')


for topdown in tqdm(range(2, 5)):
    for projection in [1]:
        if topdown == 2:
            selected_heads = [1]
            palette_permutation = [0, 1]
        elif topdown == 3:
            selected_heads = [0, 2]
            palette_permutation = [0, 1, 2, 3]
        elif topdown == 4:
            selected_heads = [0, 1, 3]
            palette_permutation = None
        else:
            selected_heads = list(range(gen_att[0].size(1)))
            palette_permutation = None
        print(f"HEAD: {selected_heads}, COLOR: {palette_permutation}")
        gen_imgs = attention_selector(gen, gen_mask, gen_att, topdown, projection, selected_heads, palette_permutation)
        for head_idx in range(len(selected_heads)):
            for idx in range(len(gen_targets)):
                data_idx = gen_targets[idx]
                try:
                    pos_min = torch.nonzero(gen_imgs[idx][head_idx].mean(0) != 1).min(0)[0]
                    pos_max = torch.nonzero(gen_imgs[idx][head_idx].mean(0) != 1).max(0)[0]
                    gen_img = gen_imgs[idx][head_idx][:, pos_min[0]:pos_max[0]+1, pos_min[1]:pos_max[1]+1]
                except RuntimeError:
                    gen_img = gen_imgs[idx][head_idx]
                head = selected_heads[head_idx]
                save_image(gen_img.float(), os.path.join(imgdir_gen, f'{topdown}_{projection}_{head}_{data_idx}.png'))
        del gen_imgs
print('gen DONE')


for topdown in tqdm(range(2, 5)):
    for projection in [0]:
        if topdown == 2:
            selected_heads = [2]
            palette_permutation = [1, 0]
        elif topdown == 3:
            selected_heads = [0]
            palette_permutation = [1, 2, 3, 0]
        elif topdown == 4:
            selected_heads = [2]
            palette_permutation = None
        else:
            selected_heads = list(range(enc_att_train[0].size(1)))
            palette_permutation = None
        print(f"HEAD: {selected_heads}, COLOR: {palette_permutation}")
        gt_imgs = attention_selector(gt_train, gt_mask_train, enc_att_train, len(enc_att_train) - 1 - topdown, projection,
                                     selected_heads=selected_heads, palette_permutation=palette_permutation)
        for head_idx in range(len(selected_heads)):
            for idx in range(len(recon_targets_train)):
                data_idx = recon_targets_train[idx]
                try:
                    pos_min = torch.nonzero(gt_imgs[idx][head_idx].mean(0) != 1).min(0)[0]
                    pos_max = torch.nonzero(gt_imgs[idx][head_idx].mean(0) != 1).max(0)[0]
                    gt_img = gt_imgs[idx][head_idx][:, pos_min[0]:pos_max[0]+1, pos_min[1]:pos_max[1]+1]
                except RuntimeError:
                    gt_img = gt_imgs[idx][head_idx]
                head = selected_heads[head_idx]
                save_image(gt_img, os.path.join(imgdir_gt_train, f'{topdown}_{projection}_{head}_{data_idx}.png'))
        del gt_imgs
print('gt DONE')
