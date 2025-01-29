import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem

import pickle
import os
import itertools
import numpy as np

from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt

import random

# Dataset
class MyDataset(Dataset):
    def __init__(
        self,
        pathway_list: list[tuple[str, list[str]]],
        label_list: list[float],
    ):
        super().__init__()
        self.smiles: list[str] = [smi for smi, traj in pathway_list]
        self.trajs: list[list[str]] = [traj for smi, traj in pathway_list]
        self.inputs: list[tuple[str, list[str]]] = pathway_list
        self.labels: list[float] = label_list

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int):
        '''see the data structure here!
        pathway = [<smiles1>, 'click|amide', <smiles2>, 'click|amide', <smiles3>, ... <smilesN>]
        label = scalar value
        '''
        label: float = self.labels[idx]
        smi: str = self.smiles[idx]
        pathway: list[str] = self.trajs[idx]

        block_smi_list: list[str] = list(pathway[0::2])
        reaction_list: list[str] = list(pathway[1::2])
        assert len(block_smi_list) == len(reaction_list) + 1
        assert set(reaction_list) <= {'click', 'amide'}

        fpgen = AllChem.GetMorganGenerator(radius=2, fpSize=1024)

        mol = Chem.MolFromSmiles(smi)
        fp = fpgen.GetFingerprint(mol)
        x1 = torch.as_tensor(fp, dtype=torch.float)

        sample = {'x1': x1, 'y': label}
        return sample

def my_collate_fn(batch: list[dict]):
    x1 = [data['x1'] for data in batch]
    y = [data['y'] for data in batch]
    return {
        'x1': torch.stack(x1, dim=0),
        'y': torch.tensor(y, dtype=torch.float),
    }

# Model
def block(in_c,out_c):
    layers=[
        nn.Linear(in_c,out_c),
        nn.ReLU(True)
    ]
    return layers

class Encoder(nn.Module):
    '''
    [mu_til, log sigma_til^2] = g(x; phi)
    g: q(z|x) = N(z; mu_til, sigma_til^2 I)
    '''
    def __init__(self,input_dim, inter_dims, hid_dim):
        super(Encoder,self).__init__()

        self.encoder=nn.Sequential(
            *block(input_dim,inter_dims[0]), 
            *block(inter_dims[0],inter_dims[1]),
            *block(inter_dims[1],inter_dims[2]),
        )
        self.mu_l=nn.Linear(inter_dims[-1],hid_dim) # mu_til from x
        self.log_sigma2_l=nn.Linear(inter_dims[-1],hid_dim) # log sigma_til^2 from x
    
    def forward(self, x):
        '''
        x: [bs, 784]
        '''
        e=self.encoder(x) # [bs, 2048] -> [bs, 4000]

        mu=self.mu_l(e) # [bs, 4000] -> [bs, 10]
        log_sigma2=self.log_sigma2_l(e) # [bs, 4000] -> [bs, 10]

        return mu,log_sigma2

class Decoder(nn.Module):
    '''
    f: p(x|z) = N(x; mu_x, sigma_x^2 I)
    '''
    def __init__(self,input_dim, inter_dims, hid_dim):
        super(Decoder,self).__init__()

        self.decoder=nn.Sequential(
            *block(hid_dim,inter_dims[-1]),
            *block(inter_dims[-1],inter_dims[-2]),
            *block(inter_dims[-2],inter_dims[-3]),
            nn.Linear(inter_dims[-3],input_dim),
            nn.Sigmoid() 
        )

    def forward(self, z):
        '''
        z : [bs, 10]
        '''
        x_pro=self.decoder(z) # [bs, 10] -> [bs, 784]

        return x_pro


class VaDE(nn.Module):
    def __init__(self,args):
        super(VaDE,self).__init__()
        self.encoder=Encoder(args['input_dim'], args['inter_dims'], args['hid_dim']) # g
        self.decoder=Decoder(args['input_dim'], args['inter_dims'], args['hid_dim']) # f

        self.pi_=nn.Parameter(torch.FloatTensor(args['nClusters'],).fill_(1)/args['nClusters'],requires_grad=True) 
        self.mu_c=nn.Parameter(torch.FloatTensor(args['nClusters'],args['hid_dim']).fill_(0),requires_grad=True)
        self.log_sigma2_c=nn.Parameter(torch.FloatTensor(args['nClusters'],args['hid_dim']).fill_(0),requires_grad=True) 

        self.args=args
    
    def pre_train(self, dataloader, pre_epoch=10):

        if  not os.path.exists(f'./pretrain_model_{self.args['debug']}.pk'):
            Loss=nn.BCELoss() # reconstruction loss
            opti=Adam(itertools.chain(self.encoder.parameters(),self.decoder.parameters())) 

            print('Pretraining......')
            epoch_bar=tqdm(range(pre_epoch))

            for _ in epoch_bar:
                L=0 
                for batch in dataloader: 
                    x = batch['x1']
                    if self.args['cuda']:
                        x=x.cuda()

                    z,_=self.encoder(x) # g(x; phi) = [mu_til, log sigma_til^2]
                    x_=self.decoder(z) # f(z; theta) = [mu_x]

                    loss=Loss(x_,x) # add for all x dim

                    L+=loss.detach().cpu().numpy()

                    opti.zero_grad()
                    loss.backward()
                    opti.step()

                epoch_bar.write('L2={:.4f}'.format(L/len(dataloader))) # avg loss per data

            self.encoder.log_sigma2_l.load_state_dict(self.encoder.mu_l.state_dict())

            ''' GMM initialization '''
            Z = [] # mu til
            with torch.no_grad():
                for batch in dataloader:
                    x = batch['x1']
                    if self.args['cuda']:
                        x = x.cuda()

                    z1, z2 = self.encoder(x) # g(x; phi) = [mu_til, log sigma_til^2]
                    assert F.mse_loss(z1, z2) == 0 
                    Z.append(z1) 

            Z = torch.cat(Z, 0).detach().cpu().numpy()

            gmm = GaussianMixture(n_components=self.args['nClusters'], covariance_type='diag')
            pre = gmm.fit_predict(Z) # trained by EM algorithm

            self.pi_.data = torch.from_numpy(gmm.weights_).cuda().float()
            self.mu_c.data = torch.from_numpy(gmm.means_).cuda().float()
            self.log_sigma2_c.data = torch.log(torch.from_numpy(gmm.covariances_).cuda().float())

            torch.save(self.state_dict(), f'./pretrain_model_{self.args['debug']}.pk')

        else:

            self.load_state_dict(torch.load(f'./pretrain_model_{self.args['debug']}.pk', weights_only=True))

    def gaussian_pdfs_log(self,x,mus,log_sigma2s):
        # log N(z|mus, sigma)
        G=[]
        for c in range(self.args['nClusters']):
            G.append(self.gaussian_pdf_log(x,mus[c:c+1,:],log_sigma2s[c:c+1,:]).view(-1,1))
        return torch.cat(G,1)

    def ELBO_Loss(self,x,L=1):
        ''' 
        x : [bs, 784]
        L : num monte carlo (default=1) 
        '''
        
        det=1e-10 # for numerical stability

        ''' 1. log p(x|z) reconstriction loss '''
        L_rec=0 

        z_mu, z_sigma2_log = self.encoder(x) # g(x;phi) = [mu_til, log sigma_til^2] # variational inference
        for l in range(L):

            z = torch.randn_like(z_mu)*torch.exp(z_sigma2_log/2) + z_mu + det # z ~ N(z; mu_til, sigma_til^2) reparam 
            x_pro = self.decoder(z) # [bs, hid_dim] -> [bs, 784]
            L_rec += F.binary_cross_entropy(x_pro,x) 
        
        L_rec /= L # average for monte carlo sampling L
        L_rec = - L_rec * x.size(1) 

        ''' 5-1. q(c|x) ~ p(c|z) clustering related term '''
        pi=self.pi_ # pi_k [k]
        log_sigma2_c=self.log_sigma2_c # log sigma_c^2 [k, hid_dim]
        mu_c=self.mu_c # mu_c [k, hid_dim]

        z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu # p(z|x) ~ p(z|c) # [bs, hid_dim]

        yita_c = torch.exp(torch.log(pi.unsqueeze(0)) + self.gaussian_pdfs_log(z,mu_c,log_sigma2_c)) + det 
        # exp [ log pi_c [1, k] + log N(z|mu_c, sigma_c) [bs, k] ] # [bs, k] 
        yita_c = yita_c / (yita_c.sum(1).view(-1,1)) # [bs, k] / [bs, 1]
        # q(c|x) ~ p(c|z) ~ normalized p(c)p(z|c) # [bs, k]

        ''' 2. log  p(z|c) '''
        # log p(z|c)
        L_2 = - 0.5 * torch.mean ( # average for data
            torch.sum( # sum for all cluster
                yita_c * torch.sum( # sum for all hid_dim
                    log_sigma2_c.unsqueeze(0) + torch.exp(z_sigma2_log.unsqueeze(1)-log_sigma2_c.unsqueeze(0)) + (z_mu.unsqueeze(1)-mu_c.unsqueeze(0)).pow(2)/torch.exp(log_sigma2_c.unsqueeze(0))
                , 2)
            , 1)
        )
        # [1, k, hid_dim] + ([bs, 1,  hid_dim] - [1, k, hid_dim]) + ([bs, 1, hid_dim] - [1, k, hid_dim])^2 / [1, k, hid_dim] 

        ''' 3. log p(c),  5-2. log q(c|x) => log p(c) - log q(c|x) '''
        L_3m5 = torch.mean( # average for data
            torch.sum( # sum for all cluster
                yita_c * torch.log(pi.unsqueeze(0)/(yita_c))
            , 1)
        )
        # [1, k]/ [bs, k]

        ''' 4. log q(z|x) '''
        L_4 = - 0.5 * torch.mean( # for all data
            torch.sum( # for all dimension
                1 + z_sigma2_log
            , 1)
        )

        L = L_rec + L_2 + L_3m5 - L_4
        Loss = - L

        return Loss # scalar
    
    def predict(self,x):
        z_mu, z_sigma2_log = self.encoder(x) # g(x; phi) = [mu_til, log sigma_til^2]
        z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu # reparam z
        
        # get gmm
        pi = self.pi_ 
        log_sigma2_c = self.log_sigma2_c
        mu_c = self.mu_c
        
        # q(c|x)
        yita_c = torch.exp(torch.log(pi.unsqueeze(0)) + self.gaussian_pdfs_log(z,mu_c,log_sigma2_c))
        # exp [ log pi_c [1, k] + log N(z|mu_c, sigma_c) [bs, k] ] # [bs, k] 
        yita=yita_c.detach().cpu().numpy() 
        return np.argmax(yita,axis=1)
        
    @staticmethod
    def gaussian_pdf_log(x,mu,log_sigma2):
        return -0.5*(torch.sum(np.log(np.pi*2)+log_sigma2+(x-mu).pow(2)/torch.exp(log_sigma2),1))
    

if __name__ == "__main__":    
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    """ parameters """
    # load model
    k = 10
    hid_dim = 10
    input_dim= 1024
    inter_dims= [1000,1000,2000]
    cuda = True
    num_workers = 2
    debug_num = k
    # save
    model_path = f"./save/model/best_{debug_num}.pt" 
    plot_path = f"./save/loss_history/loss_history_{debug_num}.png"
    save_epoch = [0, 10, 50, 100, 200, 500, 999]
    # load data
    train_data_ratio = 0.8

    # hyperparameter
    pre_epoch= 5
    lr= 2e-3
    step_size= 10
    gamma= 0.95
    n_epoch= 1000
    n_monte_carlo= 1
    batch_size = 128

    """ Load Data """
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
    args = {'nClusters':k, 'hid_dim':hid_dim, 'input_dim': input_dim, 'inter_dims': inter_dims, 'cuda': cuda, 'debug': debug_num}
    vade = VaDE(args)
    if args['cuda']:
        vade=vade.cuda()
        vade=nn.DataParallel(vade,device_ids=range(1)) 

    """ Train """
    # pretrain
    vade.module.pre_train(train_loader,pre_epoch=pre_epoch)

    opti=Adam(vade.parameters(),lr=lr)
    lr_s=StepLR(opti,step_size=step_size,gamma=gamma)

    epoch_bar=tqdm(range(n_epoch))

    # train
    train_loss_history = []
    best_loss = float('inf')

    for epoch in epoch_bar:
        L=0
        for batch in train_loader:
            x = batch['x1']
            if args['cuda']:
                x=x.cuda()

            loss=vade.module.ELBO_Loss(x, n_monte_carlo)

            opti.zero_grad()
            loss.backward()
            opti.step()

            L+=loss.detach().cpu().numpy()
        lr_s.step()

        train_loss_history.append(L/len(train_loader))
        
        tloss = L/len(train_loader)
        print('loss',tloss,epoch)
        print('lr',lr_s.get_lr()[0],epoch)
        print('Loss={:.4f},LR={:.4f}'.format(tloss,lr_s.get_lr()[0]))

        if tloss < best_loss:
            best_loss = tloss
            torch.save(vade.state_dict(), model_path)

        if epoch in save_epoch:
            torch.save(vade.state_dict(), f'./save/model/{debug_num}_{epoch}.pt')
        

    if True: # loss history
        plt.plot(train_loss_history, label='Train Loss')
        plt.ylim(0, 200)
        plt.legend(loc='upper right')
        plt.savefig(plot_path)