from main import VaDE, MyDataset, my_collate_fn

import pickle
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE

import os

def latent_plot(z_stacked, y_stacked, c_stacked, save_dir, debug):
    tsne = TSNE(n_components=2,random_state=42)
    tsne_result = tsne.fit_transform(z_stacked)
    
    plt.figure()
    dock_path = os.path.join(save_dir, f"dock_{debug}.png")
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=y_stacked, cmap='Blues_r', s=1)
    plt.colorbar(scatter, label="Docking Score")
    plt.title("t-SNE Visualization")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.savefig(dock_path)
    plt.close()

    plt.figure()
    c_max = int(max(c_stacked))
    cluster_path = os.path.join(save_dir, f"cluster_{debug}.png")
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=c_stacked, cmap='tab10', s=1)
    plt.colorbar(scatter, ticks = list(range(0, c_max + 1)), label="Cluster")
    plt.title("t-SNE Visualization")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.savefig(cluster_path)
    plt.close()
    return

if __name__ == "__main__":    
    """ parameters """
    # load data
    train_data_ratio = 0.8

    # load model
    k = 10
    debug = '10_0'

    hid_dim = 10
    input_dim= 1024
    inter_dims= [1000,1000,2000]
    cuda = True
    num_workers = 2

    batch_size = 128

    best_model_path= f"./save/model/{debug}.pt"
    save_dir = f"./save/cluster/"

    # load data
    data_file = "./aichem_2024_final_data.pkl"

    with open(data_file, 'rb') as f:
        raw_data = pickle.load(f)

    """ Load """
    # Data
    inputs = raw_data['train']['input']
    labels = raw_data['train']['label']

    Ntot = 30000

    Ntrain = int(Ntot * train_data_ratio)
    train_set = MyDataset(inputs[:Ntrain], labels[:Ntrain])
    valid_set = MyDataset(inputs[Ntrain:Ntot], labels[Ntrain:Ntot])
    print(f"length of train, valid set: {len(train_set), len(valid_set)}")

    # define data loader which batches dataset
    train_loader = DataLoader(train_set, batch_size, shuffle=True,
                            num_workers=num_workers, collate_fn=my_collate_fn)
    valid_loader = DataLoader(valid_set, batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=my_collate_fn)
    
    # Model
    args = {'nClusters':k, 'hid_dim':hid_dim, 'input_dim': input_dim, 'inter_dims': inter_dims, 'cuda': cuda, 'debug': debug}
    vade = VaDE(args)
    if args['cuda']:
        vade=vade.cuda()
        vade=nn.DataParallel(vade,device_ids=range(1)) # GPU 병렬처리. Batch를 4개로 나눠서 처리

    vade.load_state_dict(torch.load(best_model_path, weights_only=True))

    with torch.no_grad():
        z_stacked = torch.empty((0, args['hid_dim'])).cuda()
        y_stacked = torch.empty(0)
        c_stacked = torch.empty(0)
        for batch in train_loader:
            x, y = batch['x1'], batch['y']
            if args['cuda']:
                x=x.cuda()
            z, _ = vade.module.encoder(x)
            c = torch.tensor(vade.module.predict(x))
            
            c_stacked = torch.cat((c_stacked, c), dim=0)
            z_stacked = torch.cat((z_stacked, z), dim=0)
            y_stacked = torch.cat((y_stacked, y), dim=0)

    c_stacked = np.array(c_stacked)
    y_stacked = np.array(y_stacked)

    print(z_stacked.shape)
    print(len(c_stacked), len(y_stacked))
    z_stacked = z_stacked.cpu().detach().numpy()
    latent_plot(z_stacked, y_stacked, c_stacked, save_dir, debug)