
import os
import time
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import pandas as pd
import numpy as np
import anndata as ad
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from .loss import *
from .dataset import *
from .utils import *


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import spmm  # For sparse matrix multiplication
from torch_geometric.utils import sparse_softmax  # For sparse softmax

class GATE(nn.Module):
    def __init__(self, hidden_dims, alpha=0.8, nonlinear=True, weight_decay=0.0001):
        super(GATE, self).__init__()
        self.n_layers = len(hidden_dims) - 1
        self.alpha = alpha
        self.nonlinear = nonlinear
        self.weight_decay = weight_decay

        # Initialize weights
        self.W, self.v, self.prune_v = self.define_weights(hidden_dims)
        self.C = {}
        self.prune_C = {}

    def forward(self, A, prune_A, X):
        # Encoder
        H = X
        for layer in range(self.n_layers):
            H = self._encoder(A, prune_A, H, layer)
            if self.nonlinear:
                if layer != self.n_layers-1:
                    H = F.elu(H)
        # Final node representations
        self.H = H

        # Decoder
        for layer in range(self.n_layers - 1, -1, -1):
            H = self._decoder(H, layer)
            if self.nonlinear:
                if layer != 0:
                    H = F.elu(H)
        X_ = H
        
        # The reconstruction loss of node features
        features_loss = torch.sqrt(torch.sum(torch.pow(X - X_, 2)))

        # Weight decay loss
        weight_decay_loss = 0
        for layer in range(self.n_layers):
            weight_decay_loss += torch.sum(self.W[layer] ** 2) * self.weight_decay

        # Total loss
        self.loss = features_loss + weight_decay_loss

        if self.alpha == 0:
            self.Att_l = self.C
        else:
            self.Att_l = {'C': self.C, 'prune_C': self.prune_C}

        return self.loss, self.H, self.Att_l, X_

    def _encoder(self, A, prune_A, H, layer):
        H = torch.matmul(H, self.W[layer])
        if layer == self.n_layers - 1:
            return H
        self.C[layer] = self.graph_attention_layer(A, H, self.v[layer], layer)
        if self.alpha == 0:
            return spmm(self.C[layer], H)
        else:
            self.prune_C[layer] = self.graph_attention_layer(prune_A, H, self.prune_v[layer], layer)
            return (1 - self.alpha) * spmm(self.C[layer], H) + self.alpha * spmm(self.prune_C[layer], H)

    def _decoder(self, H, layer):
        H = torch.matmul(H, self.W[layer].T)
        if layer == 0:
            return H
        if self.alpha == 0:
            return spmm(self.C[layer-1], H)
        else:
            return (1 - self.alpha) * spmm(self.C[layer-1], H) + self.alpha * spmm(self.prune_C[layer-1], H)

    def define_weights(self, hidden_dims):
        W = nn.ModuleDict()
        for i in range(self.n_layers):
            W[i] = nn.Parameter(torch.randn(hidden_dims[i], hidden_dims[i+1]))

        v = {}
        for i in range(self.n_layers - 1):
            v[i] = {}
            v[i][0] = nn.Parameter(torch.randn(hidden_dims[i+1], 1))
            v[i][1] = nn.Parameter(torch.randn(hidden_dims[i+1], 1))

        if self.alpha == 0:
            return W, v, None
        
        prune_v = {}
        for i in range(self.n_layers - 1):
            prune_v[i] = {}
            prune_v[i][0] = nn.Parameter(torch.randn(hidden_dims[i+1], 1))
            prune_v[i][1] = nn.Parameter(torch.randn(hidden_dims[i+1], 1))

        return W, v, prune_v

    def graph_attention_layer(self, A, M, v, layer):
        # Calculate attention scores using matrix multiplication
        f1 = torch.matmul(M, v[0])
        f1 = A * f1
        f2 = torch.matmul(M, v[1])
        f2 = A * f2.T
        logits = f1 + f2

        # Apply sparse softmax
        unnormalized_attentions = torch.sparse.FloatTensor(logits.indices(), logits.values(), logits.size())
        attentions = sparse_softmax(unnormalized_attentions)

        return attentions
   


def GAT_adata(
        adata,
        data_type,
        experiment='generation',
        down_ratio=0.5,
        coord_sf=77,
        sec_name='section',
        select_section=[1, 3, 5, 6, 8],
        gap=0.05,
        expand_time=5,
        rad_off=1,
        train_epoch=2000,
        seed=1234,
        batch_size=512,
        learning_rate=1e-3,
        w_recon=0.1,
        w_w=0.1,
        w_l1=0.1,
        step_size=500,
        gamma=1,
        relu=True,
        device='cpu',
        path1='/Users/tinaguo/Desktop/FinalProject/'):
   
    # set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    # Set the device for the computation
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    if data_type == '10x' and experiment not in ['generation', 'recovery']:
        raise ValueError("Experiments designed for 10x Visium data are only 'generation' and 'recovery'.")
    elif data_type == 'ST_KTH' and experiment != 'generation':
        raise ValueError("Experiment designed for Spatial Transcriptomics data is only 'generation'.")
    elif data_type == 'Slide-seq' and experiment != '3d_model':
        raise ValueError("Experiment designed for Slide-seq data is only '3d_model'.")

    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    if not os.path.isdir(path1):
        os.mkdir(path1)

    # Preparation
    if experiment=='generation' and data_type=='10x':
        coor_df, fill_coor_df = generation_coord_10x(adata)
        used_gene, normed_data = get_data(adata, experiment=experiment)
    elif experiment=='recovery' and data_type=='10x':
        coor_df, fill_coor_df, sample_index, sample_barcode = recovery_coord(adata, down_ratio=down_ratio, path1=path1)
        used_gene, normed_data, adata_sample = get_data(adata, experiment=experiment, sample_index=sample_index, sample_barcode=sample_barcode, path1=path1)
    elif experiment=='generation' and data_type=='ST_KTH':
        coor_df, fill_coor_df = generation_coord_ST(adata)
        used_gene, normed_data = get_data(adata, experiment=experiment)
    elif experiment=='3d_model' and data_type=='Slide-seq':
        used_gene, normed_data = get_data(adata, experiment=experiment, sec_name=sec_name, select_section=select_section)
        coor_df, fill_coor_df, new_coor_df, all_coor_df = Slide_seq_coord_3d(
            adata,sec_name=sec_name,select_section=select_section,gap=gap)
    elif experiment=='higher_res' and data_type=='arbitrary':
        coor_df, fill_coor_df = generate_coord_random(adata, expand_time=expand_time, rad_off=rad_off)
        used_gene, normed_data = get_data(adata, experiment=experiment)


    if experiment == 'recovery':
        normed_coor_df = coor_df.copy()
        normed_coor_df = normed_coor_df / coord_sf

        normed_fill_coor_df = fill_coor_df.copy()
        normed_fill_coor_df = normed_fill_coor_df / coord_sf

        normed_coor_df.to_csv(path1+"/coord.txt", header=0, index=0)
        normed_fill_coor_df.to_csv(path1+"/fill_coord.txt", header=0, index=0)

    if experiment == 'generation' or experiment == 'higher_res':
        normed_coor_df = coor_df.copy()
        normed_coor_df = normed_coor_df / coord_sf
        X_dim = 2
    elif experiment == '3d_model' and data_type == 'Slide-seq':
        normed_coor_df = coor_df.copy()
        normed_coor_df.iloc[:, range(2)] = normed_coor_df.iloc[:, range(2)] / coord_sf
        X_dim = 3

    transformed_dataset = MyDataset(normed_data=normed_data, coor_df=normed_coor_df, transform=transforms.Compose([ToTensor()]))
    train_loader = DataLoader(transformed_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)

    # Training process
    gene_number = normed_data.shape[0]
    encoder, decoder = GATE._encoder(gene_number, X_dim), GATE._decoder(gene_number, X_dim)

    encoder.train()
    decoder.train()

    encoder, decoder = encoder.to(device), decoder.to(device)

    enc_optim = optim.Adam(encoder.parameters(), lr=learning_rate)
    dec_optim = optim.Adam(decoder.parameters(), lr=learning_rate)

    enc_sche = optim.lr_scheduler.StepLR(enc_optim, step_size=step_size, gamma=gamma)
    dec_sche = optim.lr_scheduler.StepLR(dec_optim, step_size=step_size, gamma=gamma)


    with tqdm(range(train_epoch), total=train_epoch, desc='Epochs') as epoch:
        for j in epoch:

            train_loss = []
            train_lc_loss = []
            train_re_loss = []

            for xdata, xlabel in train_loader:
                xdata = xdata.to(torch.float32)
                xlabel = xlabel.to(torch.float32)

                enc_optim.zero_grad()
                dec_optim.zero_grad()

                xdata, xlabel, = Variable(xdata.to(device)), Variable(xlabel.to(device))

                latent = encoder(xdata, relu)
                latent = latent.view(-1, X_dim)
                xlabel = xlabel.float().to(device)
                latent_loss = loss1(latent, xlabel) + w_w * sliced_wasserstein_distance(latent, xlabel, 1000, device=device)
                xrecon = decoder(latent, relu)
                recon_loss = loss2(xrecon, xdata) + w_l1 * loss1(xrecon, xdata)

                total_loss = 0.1 * latent_loss + 0.1 * w_recon * recon_loss

                total_loss.backward()

                enc_optim.step()
                dec_optim.step()

                enc_sche.step()
                dec_sche.step()

                train_lc_loss.append(latent_loss.item())
                train_re_loss.append(recon_loss.item())
                train_loss.append(total_loss.item())

            epoch_info = 'latent_loss: %.5f, recon_loss: %.5f, total_loss: %.5f' % \
                         (torch.mean(torch.FloatTensor(train_lc_loss)),
                          torch.mean(torch.FloatTensor(train_re_loss)),
                          torch.mean(torch.FloatTensor(train_loss)))
            epoch.set_postfix_str(epoch_info)

    torch.save(encoder, save_path+'/encoder.pth')
    torch.save(decoder, save_path+'/decoder.pth')

    encoder.eval()
    decoder.eval()

    # Get generated or recovered data
    if experiment=='generation' or experiment=='recovery' or experiment=='higher_res':
        normed_fill_coor_df = fill_coor_df.copy()
        normed_fill_coor_df = normed_fill_coor_df / coord_sf
        normed_fill_coor_df = torch.from_numpy(np.array(normed_fill_coor_df))
        normed_fill_coor_df = normed_fill_coor_df.to(torch.float32)
        normed_fill_coor_df = Variable(normed_fill_coor_df.to(device))
        generate_profile = decoder(normed_fill_coor_df, relu)
        generate_profile = generate_profile.cpu().detach().numpy()

        if not relu:
            generate_profile = np.clip(generate_profile, a_min=0, a_max=None)

        if experiment=='recovery':
            np.savetxt(save_path+"/fill_data.txt", generate_profile)

        adata_new = sc.AnnData(generate_profile)
        adata_new.obsm["coord"] = fill_coor_df.to_numpy()
        adata_new.var.index = used_gene

        adata.write(save_path + '/original_data.h5ad')

        if experiment=='generation' or experiment=='higher_res':
            adata_new.write(save_path + '/generated_data.h5ad')
            return adata_new
        elif experiment=='recovery' and data_type=='10x':
            adata_sample.write(save_path + '/sampled_data.h5ad')
            adata_new.obs = adata.obs
            adata_new.write(save_path + '/recovered_data.h5ad')
            return adata_sample, adata_new

    elif experiment=='3d_model' and data_type=='Slide-seq':

        # recovered data
        normed_fill_coor_df = fill_coor_df.copy()
        normed_fill_coor_df.iloc[:, range(2)] = normed_fill_coor_df.iloc[:, range(2)] / coord_sf
        normed_fill_coor_df = torch.from_numpy(np.array(normed_fill_coor_df))
        normed_fill_coor_df = normed_fill_coor_df.to(torch.float32)
        normed_fill_coor_df = Variable(normed_fill_coor_df.to(device))
        generate_profile = decoder(normed_fill_coor_df, relu)
        generate_profile = generate_profile.cpu().detach().numpy()

        if not relu:
            generate_profile = np.clip(generate_profile, a_min=0, a_max=None)

        adata_new = sc.AnnData(generate_profile)
        adata_new.obsm["coord"] = fill_coor_df.to_numpy()
        adata_new.var.index = used_gene
        adata_new.obs = adata.obs

        # generated data
        normed_new_coor_df = new_coor_df.copy()
        normed_new_coor_df.iloc[:, range(2)] = normed_new_coor_df.iloc[:, range(2)] / coord_sf
        normed_new_coor_df = torch.from_numpy(np.array(normed_new_coor_df))
        normed_new_coor_df = normed_new_coor_df.to(torch.float32)
        normed_new_coor_df = Variable(normed_new_coor_df.to(device))
        generate_profile = decoder(normed_new_coor_df, relu)
        generate_profile = generate_profile.cpu().detach().numpy()

        if not relu:
            generate_profile = np.clip(generate_profile, a_min=0, a_max=None)

        adata_simu = sc.AnnData(generate_profile)
        adata_simu.obsm["coord"] = new_coor_df.to_numpy()
        adata_simu.var.index = used_gene
        new_coor_df.index = adata_simu.obs.index
        new_coor_df.columns = ["xcoord", "ycoord", "zcoord"]
        adata_simu.obs = new_coor_df

        # all data
        #adata_all = ad.concat([adata_new, adata_simu], merge="same")

        return adata_new, adata_simu