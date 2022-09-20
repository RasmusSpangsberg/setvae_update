from pathlib import Path
import os
import numpy as np

from datasets import get_datasets
from models.networks import SetVAE
from args import get_args

import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import umap
import pandas as pd
import colorsys
from pyntcloud import PyntCloud


def get_train_loader(args):
    train_dataset, val_dataset, train_loader, val_loader = get_datasets(args)
    if args.resume_dataset_mean is not None and args.resume_dataset_std is not None:
        mean = np.load(args.resume_dataset_mean)
        std = np.load(args.resume_dataset_std)
        val_dataset.renormalize(mean, std)

    loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=train_loader.collate_fn,
        num_workers=0, pin_memory=True, drop_last=False)
    return loader


def reduce_dimensionality(x, method_name, dimensions=2):
    assert method_name in ["tsne", "umap", "pca"], "method_name got: {0} expected: either 'tsne', 'umap' or 'pca'".format(method_name)

    x = x.reshape(-1, x.shape[1] * x.shape[2])

    if method_name == "tsne":
        return TSNE(n_components=dimensions).fit_transform(x)
    elif method_name == "pca":
        return PCA(n_components=dimensions).fit_transform(x)
    elif method_name == "umap":
        return umap.UMAP(n_components=dimensions).fit_transform(x)


def legend_without_duplicate_labels(figure):
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    figure.legend(by_label.values(), by_label.keys(), loc='lower right')


def visualize_latent_variable(latent_variables, unns, axis=None, reduction_method_name="pca", dimensions=2):
    assert dimensions in [2, 3], f"dimensions has to be either 2 or 3, got {dimensions}"
    color_mapping = plt.cm.get_cmap("hsv", 32)  # 32 different unn

    reduced_latent_variables = reduce_dimensionality(latent_variables, reduction_method_name, dimensions)

    teeth_types = [["1", "2", "3", "14", "15", "16", "17", "18", "19", "32", "31", "30"], ["4", "5", "12", "13", "20", "21", "29", "28"], ["6", "11", "22", "27"], ["7", "8", "9", "10",  "26", "25", "24", "23"]]
    teeth_types_label = ['Molar', 'Pre-molar', "Canine", "Incisor"]

    # used for visualizing 3 dimensions
    data_frames = []

    for unn in set(unns):
        matched = [unn == x for x in unns]

        xs = reduced_latent_variables[matched, :]
        if dimensions == 2:
            axis.scatter(xs[:, 0], xs[:, 1], label=unn, s=0.1)
        elif dimensions == 3:
            h, s, v = color_mapping(int(unn))[:-1]
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            r = int(r * 255)
            g = int(g * 255)
            b = int(b * 255)

            points = [{"x": x[0], "y": x[1], "z": x[2], "red": r, "green": g, "blue": b} for x in xs]

            data_frame = pd.DataFrame(points)
            data_frames.append(data_frame)

    if dimensions == 3:
        return data_frames


def encode_data_with_model(model, data):
    # ground truth
    gt, gt_mask = data['set'], data['set_mask']
    gt = gt.cuda()
    gt_mask = gt_mask.to(gt.device)

    return model.bottom_up(gt, gt_mask)


def main(args):
    args.seed = 42
    model = SetVAE(args)
    model = model.cuda()

    model_dir = Path("checkpoints") / args.log_name
    args.resume_checkpoint = os.path.join(model_dir, f'checkpoint-{args.epochs - 1}.pt')
    checkpoint = torch.load(args.resume_checkpoint)

    model.load_state_dict(checkpoint['model'])
    model.eval()

    args.batch_size = 128
    loader = get_train_loader(args)

    features = []
    unns = []
    for data in loader:
        encoding = encode_data_with_model(model, data)
        features_ = encoding["features"]
        unns += data["unn"]

        if len(features) == 0:
            features = [feature.cpu().detach().numpy() for feature in features_]
        else:
            for j in range(len(features)):
                features[j] = np.concatenate((features[j], features_[j].cpu().detach().numpy()))

    for reduction_method_name in ["umap", "pca", "tsne"]:
        figures_path = f"latent_space_viz/{reduction_method_name}"

        if not os.path.exists(figures_path):
            os.makedirs(figures_path)

        for i, feature in enumerate(features):
            # 2d plots
            figure, axis = plt.subplots(3, 2, figsize=(12, 10))
            x = i % 3
            y = int(i > 2)

            visualize_latent_variable(feature, unns, axis=axis[x, y], dimensions=2, reduction_method_name=reduction_method_name)

            if x == 0 and y == 0:
                lgnd = figure.legend()
                for x in range(len(lgnd.legendHandles)):
                    lgnd.legendHandles[x]._sizes = [30]

            figure.savefig(f"latent_space_viz/{reduction_method_name}/2d.pdf")

            # 3d plots
            data_frames = visualize_latent_variable(feature, unns, dimensions=3, reduction_method_name=reduction_method_name)

            data_frame = pd.concat(data_frames, ignore_index=True)
            data_frame[["red", "green", "blue"]] = data_frame[["red", "green", "blue"]].astype(np.uint8)

            pc = PyntCloud(data_frame)
            pc.to_file(f"latent_space_viz/{reduction_method_name}/layer{i}.ply")

        '''
        # 2d plots
        k = 0
        figure, axis = plt.subplots(3, 2, figsize=(12, 10))
        
        for i in range(3):
            for j in range(2):
                visualize_latent_variable(features[k], unns, axis=axis[i, j], dimensions=2, reduction_method_name=reduction_method_name)

                k += 1
                if i == 0 and j == 0:
                    lgnd = figure.legend()
                    for x in range(len(lgnd.legendHandles)):
                        lgnd.legendHandles[x]._sizes = [30]

        figure.savefig(f"latent_space_viz/{reduction_method_name}.pdf")
    
        # 3d plots
        for i, feature in enumerate(features):
            data_frames = visualize_latent_variable(feature, unns, dimensions=3, reduction_method_name=reduction_method_name)

            data_frame = pd.concat(data_frames, ignore_index=True)
            data_frame[["red", "green", "blue"]] = data_frame[["red", "green", "blue"]].astype(np.uint8)

            pc = PyntCloud(data_frame)
            pc.to_file(f"latent_space_viz/{reduction_method_name}_{i}.ply")
        '''


if __name__ == "__main__":
    args = get_args()
    main(args)

