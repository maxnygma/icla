import os
import wandb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import openood.utils.comm as comm
from openood.postprocessors import BasePostprocessor
from openood.utils import Config


def to_np(x):
    return x.data.cpu().numpy()


# (PROLY FOR DELETION)
def compute_embeddings_plots(embeddings, labels):
    tsne = TSNE(n_components=2, random_state=0)
    embeddings_2d = tsne.fit_transform(embeddings)
   
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in np.linspace(0, 1, len(set(labels)))]

    for i, c in zip(sorted(set(labels)), colors):
        plt.scatter(embeddings_2d[labels == i, 0], embeddings_2d[labels == i, 1], c=c,
                    label=i)

    plt.savefig('checkpoints/embeddings.png')
    plt.close()



class BaseEvaluator:
    def __init__(self, config: Config):
        self.config = config
        if self.config.recorder.name == 'wandb':
            wandb.init(dir=self.config.output_dir,
                       project=self.config.recorder.project,
                       name=self.config.recorder.experiment,
                       group=self.config.recorder.group or None)

    def eval_acc(self,
                 net: nn.Module,
                 data_loader: DataLoader,
                 postprocessor: BasePostprocessor = None,
                 epoch_idx: int = -1,
                 visualize_embeddings=False): # (PROLY FOR DELETION OR COMMENT - vis + MCCS)
        net.eval()

        # (PROLY FOR DELETION)
        if visualize_embeddings:
            feature_list = []
            label_list = []

        loss_avg = 0.0
        correct = 0
        with torch.no_grad():
            for batch in tqdm(data_loader,
                              desc='Eval: ',
                              position=0,
                              leave=True):
                # prepare data
                data = batch['data'].cuda()
                target = batch['label'].cuda()

                # forward
                # (PROLY FOR DELETION)
                if not visualize_embeddings:
                    output = net(data)
                    loss = F.cross_entropy(output, target)
                else:
                    output, feature = net(data, return_feature=True)
                    loss = F.cross_entropy(output, target)

                    feature_list.append(feature.detach().cpu())
                    label_list.append(target.detach().cpu())

                # accuracy
                pred = output.data.max(1)[1]
                correct += pred.eq(target.data).sum().item()

                # test loss average
                loss_avg += float(loss.data)

        loss = loss_avg / len(data_loader)
        acc = correct / len(data_loader.dataset)

        # (PROLY FOR DELETION)
        if visualize_embeddings:
            print('Visualizing embeddings...')

            feature_list = torch.stack(feature_list[:-1], dim=0).view(-1, feature_list[0].size()[-1])
            label_list = torch.stack(label_list[:-1], dim=0).view(-1)

            # Cosine Similarity
            np_labels = label_list.cpu().detach().numpy()
            sim_by_class = []
            for i in sorted(set(np_labels)):
                total_sim = 0
                for j in sorted(set(np_labels)):
                    if i == j:
                        continue

                    norm_x = feature_list[np_labels == i] / feature_list[np_labels == i].norm(dim=1)[:, None]
                    norm_y = feature_list[np_labels == j] / feature_list[np_labels == j].norm(dim=1)[:, None]

                    sim = torch.mean(torch.mm(norm_x, norm_y.transpose(0, 1)))
                    total_sim += sim

                total_sim /= (len(set(np_labels)) - 1)
                print(f'Class {i}: {total_sim}')

                sim_by_class.append(total_sim)
            print(f'Max by class: {max(sim_by_class)}')
            print(f'Mean embeddings cos sim: {torch.tensor(sim_by_class).mean()}, Variance of cos sim: {torch.tensor(sim_by_class).var()}')

            compute_embeddings_plots(feature_list.numpy(), label_list.numpy())

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = self.save_metrics(loss)
        metrics['acc'] = self.save_metrics(acc)
        return metrics

    def extract(self, net: nn.Module, data_loader: DataLoader):
        net.eval()
        feat_list, label_list = [], []

        with torch.no_grad():
            for batch in tqdm(data_loader,
                              desc='Feature Extracting: ',
                              position=0,
                              leave=True):
                data = batch['data'].cuda()
                label = batch['label']

                _, feat = net(data, return_feature=True)
                feat_list.extend(to_np(feat))
                label_list.extend(to_np(label))

        feat_list = np.array(feat_list)
        label_list = np.array(label_list)

        save_dir = self.config.output_dir
        os.makedirs(save_dir, exist_ok=True)
        np.savez(os.path.join(save_dir, 'feature'),
                 feat_list=feat_list,
                 label_list=label_list)

    def save_metrics(self, value):
        all_values = comm.gather(value)
        temp = 0
        for i in all_values:
            temp = temp + i
        # total_value = np.add([x for x in all_values])s

        return temp
