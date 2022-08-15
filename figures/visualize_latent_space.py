from pathlib import Path
import os
import numpy as np

from datasets import get_datasets
from models.networks import SetVAE
from args import get_args

import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


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


def reduce_dimensionality(x, method_name, dimensions=3):
    assert method_name in ["tsne", "umap", "pca"], "method_name got: {0} expected: either 'tsne', 'umap' or 'pca'".format(method_name)
    if method_name == "tsne":
        print("x:", x.shape)
        x = x.reshape(x.shape[1], x.shape[2])
        return TSNE(n_components=dimensions).fit_transform(x)
    elif method_name == "umap":
        return None
    elif method_name == "pca":
        return None


def visualize_latent_variable(latent_variables):
    reduced_latent_variables = reduce_dimensionality(latent_variables, "tsne", 2)
    print("reduced:", reduced_latent_variables, reduced_latent_variables.shape)

    plt.scatter(reduced_latent_variables[:, 0], reduced_latent_variables[:, 1])
    plt.savefig("my_plot.pdf")


def encode_data_with_model(model, data):
    gt, gt_mask = data['set'], data['set_mask']
    gt = gt.cuda()
    gt_mask = gt_mask.to(gt.device)

    return model.bottom_up(gt, gt_mask)


def main(args):
    model = SetVAE(args)
    model = model.cuda()

    save_dir = Path("checkpoints") / args.log_name
    args.resume_checkpoint = os.path.join(save_dir, f'checkpoint-{args.epochs - 1}.pt')
    checkpoint = torch.load(args.resume_checkpoint)

    model.load_state_dict(checkpoint['model'])
    model.eval()

    args.batch_size = 1
    loader = get_test_loader(args)

    data = next(iter(loader))

    encoding = encode_data_with_model(model, data)
    features = encoding["features"]
    alphas = encoding["alphas"]
    print(len(features))
    print(type(features[0]))
    print(len(alphas))
    print(type(alphas[0]))
    
    features = [feature.cpu().detach().numpy() for feature in features]

    for i in range(7):
        print(features[i].shape)

    visualize_latent_variable(features[0])


if __name__ == "__main__":
    args = get_args()
    main(args)
