import os
from pathlib import Path
from collections import OrderedDict, defaultdict

import torch
import numpy as np
from tqdm import tqdm

from args import get_args
from datasets import get_datasets
from models.networks import SetVAE


def get_train_loader(args):
    train_dataset, _, train_loader, _ = get_datasets(args)
    if args.resume_dataset_mean is not None and args.resume_dataset_std is not None:
        mean = np.load(args.resume_dataset_mean)
        std = np.load(args.resume_dataset_std)
        train_dataset.renormalize(mean, std)
    loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=train_loader.collate_fn,
        num_workers=0, pin_memory=True, drop_last=False)
    return loader


def get_test_loader(args):
    _, val_dataset, _, val_loader = get_datasets(args)
    if args.resume_dataset_mean is not None and args.resume_dataset_std is not None:
        mean = np.load(args.resume_dataset_mean)
        std = np.load(args.resume_dataset_std)
        val_dataset.renormalize(mean, std)
    loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=val_loader.collate_fn,
        num_workers=0, pin_memory=True, drop_last=False)
    return loader


def collate(result):
    print(result.keys(), len(result.keys()))
    # Concat summary
    for k, v in result.items():
        print("current key:", k)
        if 'set' in k or 'mask' in k or k in ['std', 'mean']:
            if type(v[0]) != torch.Tensor:
                result[k] = torch.tensor(v)
            else:
                result[k] = torch.cat(v, 0)
        elif 'att' in k:
            # OUTER LOOP, Z, Tensor
            inner = len(v[0])
            outer = len(v)
            lst = list()
            for i in range(inner):
                pass
                # 2, HEAD, BATCH, CARD, IND
                #lst.append(torch.cat([v[j][i] for j in range(outer)], 2))
            '''
                print("2.3")
                # 2, HEAD, BATCH, CARD, IND
                #lst.append(torch.cat([v[j][i] for j in range(outer)], 2))
                # MY CHANGE:
                my_shape = None
                print("b")
                print("outer:", outer)
                for j in range(outer):
                    if my_shape == None:
                        my_shape = v[j][i].shape

                    a = v[j][i].shape

                    if a[0] != my_shape[0] or a[1] != my_shape[1] or a[3] != my_shape[3] or a[4] != my_shape[4]:
                        print("my_shape, v[j][i]:", my_shape, v[j][i].shape, j, i)
                b = [v[j][i] for j in range(outer)]
                print("hi")
                torch.cat([v[j][i] for j in range(outer//10)], 2)
                print("hi again")
                c = torch.cat(b[0:outer//2], 2)
                print("hihi")
                d = torch.cat(b[outer//2:], 2)
                print("xd")
                torch.cat([c, d], 2)
                print("len(b):", len(b))
                a = torch.cat(b, 2)
                print("apppend")
                lst.append(a)
            '''

            print("2.4")
            result[k] = lst
        elif k in ['posteriors', 'priors']:
            # OUTER LOOP, Z, Tensor
            inner = len(v[0])
            outer = len(v)
            lst = list()
            for i in range(inner):
                lst.append(torch.cat([v[j][i] for j in range(outer)], 0))
            result[k] = lst
        else:
            result[k] = v
    return result


def recon(model, args, data):
    idx_b, gt, gt_mask = data['idx'], data['set'], data['set_mask']
    gt = gt.cuda()
    gt_mask = gt_mask.to(gt.device)

    output = model(gt, gt_mask)
    enc_att, dec_att = output['alphas']
    # Batch, Cardinality, n_mixtures
    enc_att = [torch.stack(a, 0).cpu() for a in enc_att]
    dec_att = [torch.stack(a, 0).cpu() for a in dec_att]

    posteriors = [z.cpu() for z, _, _ in output['posteriors']]

    # TODO: attention to cpu

    result = {
        'recon_set': output['set'].cpu(),
        'recon_mask': output['set_mask'].cpu(),
        'posteriors': posteriors,
        'dec_att': dec_att,
        'enc_att': enc_att,
    }
    return result


def sample(model, args, data):
    gt_c = data['cardinality']
    gt_c = gt_c.cuda()

    output = model.sample(gt_c)
    priors = [z.cpu() for z, _, _ in output['priors']]
    smp_att = [torch.stack(a, 0).cpu() for a in output['alphas']]
    result = {
        'smp_set': output['set'].cpu(),
        'smp_mask': output['set_mask'].cpu(),
        'smp_att': smp_att,
        'priors': priors,
    }
    return result


def train_recon(model, args):
    print("train recon 0")
    loader = get_train_loader(args)
    save_dir = os.path.dirname(args.resume_checkpoint)
    print("train recon")

    summary = defaultdict(list)
    for idx, data in enumerate(tqdm(loader)):
        if idx % 100 == 0:
            print(idx, data.keys())
        gt_result = {
            'gt_set': data['set'],
            'gt_mask': data['set_mask'],
            'mean': data['mean'],
            'std': data['std'],
            'sid': data['sid'],
            'mid': data['mid'],
            'cardinality': data['cardinality'],
        }
        result = recon(model, args, data)
        result.update(gt_result)

        for k, v in result.items():
            summary[k].append(v)

        del gt_result, result

    print("before collat in train recon")
    summary = collate(summary)
    print("after")
    summary_name = Path(save_dir) / f"summary_train_recon.pth"
    torch.save(summary, summary_name)
    print(summary_name)


def sample_and_recon(model, args):
    all_sample, all_sample_mask = None, None
    all_ref, all_ref_mask = None, None

    loader = get_test_loader(args)
    save_dir = os.path.dirname(args.resume_checkpoint)
    print("1")
    summary = dict()
    for idx, data in enumerate(tqdm(loader)):
        gt_result = {
            'gt_set': data['set'],
            'gt_mask': data['set_mask'],
            'mean': data['mean'],
            'std': data['std'],
            'sid': data['sid'],
            'mid': data['mid'],
            'cardinality': data['cardinality'],
        }
        result = dict()
        # sample needs : Set, Mask, Smp_att, prior
        smp_result = sample(model, args, data)

        # recon needs : set, mask, enc_att, dec_att, posterior, gt_set, gt_mask
        recon_result = recon(model, args, data)
        result.update(smp_result)
        result.update(recon_result)
        result.update(gt_result)
        if len(summary.keys()) == 0:
            for k in result.keys():
                summary[k] = []
        for k, v in result.items():
            summary[k].append(v)
    print("2")
    summary = collate(summary)
    print("3")
    summary_name = Path(save_dir) / f"summary.pth"
    torch.save(summary, summary_name)
    print("Saving summary to:", summary_name)


def main(args):
    model = SetVAE(args)
    model = model.cuda()

    save_dir = Path("checkpoints") / args.log_name
    args.resume_checkpoint = os.path.join(save_dir, f'checkpoint-{args.epochs - 1}.pt')
    print("Resume Path:%s" % args.resume_checkpoint)
    checkpoint = torch.load(args.resume_checkpoint)

    try:
        model.load_state_dict(checkpoint['model'])
    except RuntimeError:
        print("Load failed, trying compatibility matching")
        ckpt = checkpoint['model']
        updated_ckpt = OrderedDict()
        for k, v in ckpt.items():
            k = k.split('.')
            k[0] = f"{k[0]}.module"
            k = '.'.join(k)
            updated_ckpt.update({k: v})
        model.load_state_dict(updated_ckpt)
        print("Load success")
    
    print("ASD")
    model.eval()
    with torch.no_grad():
        print("QWE")
        sample_and_recon(model, args)
        print("ZXC")
        train_recon(model, args)


if __name__ == '__main__':
    args = get_args()
    args.batch_size = 4
    print(args)
    main(args)
