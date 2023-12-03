import os
import yaml
import shutil
import sys
import time
import warnings
import numpy as np
from random import sample
from sklearn import metrics
from datetime import datetime
from torch_sparse import SparseTensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch_scatter import scatter
from model.dimenet_ori import DimeNetPlusPlusWrap
#from matformer.models.pyg_att import Matformer

OFFSET_LIST = [
    [-1, -1, -1],
    [-1, -1, 0],
    [-1, -1, 1],
    [-1, 0, -1],
    [-1, 0, 0],
    [-1, 0, 1],
    [-1, 1, -1],
    [-1, 1, 0],
    [-1, 1, 1],
    [0, -1, -1],
    [0, -1, 0],
    [0, -1, 1],
    [0, 0, -1],
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, -1],
    [0, 1, 0],
    [0, 1, 1],
    [1, -1, -1],
    [1, -1, 0],
    [1, -1, 1],
    [1, 0, -1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, -1],
    [1, 1, 0],
    [1, 1, 1],
]

EPSILON = 1e-5

def min_distance_sqr_pbc(cart_coords1, cart_coords2, lengths, angles,
                         num_atoms, device, return_vector=False,
                         return_to_jimages=False):
    """Compute the pbc distance between atoms in cart_coords1 and cart_coords2.
    This function assumes that cart_coords1 and cart_coords2 have the same number of atoms
    in each data point.
    returns:
        basic return:
            min_atom_distance_sqr: (N_atoms, )
        return_vector == True:
            min_atom_distance_vector: vector pointing from cart_coords1 to cart_coords2, (N_atoms, 3)
        return_to_jimages == True:
            to_jimages: (N_atoms, 3), position of cart_coord2 relative to cart_coord1 in pbc
    """
    batch_size = len(num_atoms)

    # Get the positions for each atom
    pos1 = cart_coords1
    pos2 = cart_coords2

    unit_cell = torch.tensor(OFFSET_LIST, device=device).float()
    num_cells = len(unit_cell)
    unit_cell_per_atom = unit_cell.view(1, num_cells, 3).repeat(
        len(cart_coords2), 1, 1
    )
    unit_cell = torch.transpose(unit_cell, 0, 1)
    unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(
        batch_size, -1, -1
    )

    # lattice matrix
    lattice = lattice_params_to_matrix_torch(lengths, angles)

    # Compute the x, y, z positional offsets for each cell in each image
    data_cell = torch.transpose(lattice, 1, 2)
    pbc_offsets = torch.bmm(data_cell, unit_cell_batch)
    pbc_offsets_per_atom = torch.repeat_interleave(
        pbc_offsets, num_atoms, dim=0
    )

    # Expand the positions and indices for the 9 cells
    pos1 = pos1.view(-1, 3, 1).expand(-1, -1, num_cells)
    pos2 = pos2.view(-1, 3, 1).expand(-1, -1, num_cells)
    # Add the PBC offsets for the second atom
    pos2 = pos2 + pbc_offsets_per_atom

    # Compute the vector between atoms
    # shape (num_atom_squared_sum, 3, 27)
    atom_distance_vector = pos1 - pos2
    atom_distance_sqr = torch.sum(atom_distance_vector ** 2, dim=1)

    min_atom_distance_sqr, min_indices = atom_distance_sqr.min(dim=-1)

    return_list = [min_atom_distance_sqr]

    if return_vector:
        min_indices = min_indices[:, None, None].repeat([1, 3, 1])

        min_atom_distance_vector = torch.gather(
            atom_distance_vector, 2, min_indices).squeeze(-1)

        return_list.append(min_atom_distance_vector)

    if return_to_jimages:
        to_jimages = unit_cell.T[min_indices].long()
        return_list.append(to_jimages)

    return return_list[0] if len(return_list) == 1 else return_list

def build_mlp(in_dim, hidden_dim, fc_num_layers, out_dim):
    mods = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for i in range(fc_num_layers-1):
        mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    mods += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*mods)


def frac_to_cart_coords(
    frac_coords,
    lengths,
    angles,
    num_atoms,
):
    lattice = lattice_params_to_matrix_torch(lengths, angles)
    lattice_nodes = torch.repeat_interleave(lattice, num_atoms, dim=0)
    pos = torch.einsum('bi,bij->bj', frac_coords, lattice_nodes)  # cart coords

    return pos


def lattice_params_to_matrix_torch(lengths, angles):
    """Batched torch version to compute lattice matrix from params.

    lengths: torch.Tensor of shape (N, 3), unit A
    angles: torch.Tensor of shape (N, 3), unit degree
    """
    angles_r = torch.deg2rad(angles)
    coses = torch.cos(angles_r)
    sins = torch.sin(angles_r)

    val = (coses[:, 0] * coses[:, 1] - coses[:, 2]) / (sins[:, 0] * sins[:, 1])
    # Sometimes rounding errors result in values slightly > 1.
    val = torch.clamp(val, -1., 1.)
    gamma_star = torch.arccos(val)

    vector_a = torch.stack([
        lengths[:, 0] * sins[:, 1],
        torch.zeros(lengths.size(0), device=lengths.device),
        lengths[:, 0] * coses[:, 1]], dim=1)
    vector_b = torch.stack([
        -lengths[:, 1] * sins[:, 0] * torch.cos(gamma_star),
        lengths[:, 1] * sins[:, 0] * torch.sin(gamma_star),
        lengths[:, 1] * coses[:, 0]], dim=1)
    vector_c = torch.stack([
        torch.zeros(lengths.size(0), device=lengths.device),
        torch.zeros(lengths.size(0), device=lengths.device),
        lengths[:, 2]], dim=1)

    return torch.stack([vector_a, vector_b, vector_c], dim=1)

def cart_to_frac_coords(
    cart_coords,
    lengths,
    angles,
    num_atoms,
):
    lattice = lattice_params_to_matrix_torch(lengths, angles)
    # use pinv in case the predicted lattice is not rank 3
    inv_lattice = torch.linalg.pinv(lattice)
    inv_lattice_nodes = torch.repeat_interleave(inv_lattice, num_atoms, dim=0)
    frac_coords = torch.einsum('bi,bij->bj', cart_coords, inv_lattice_nodes)
    return (frac_coords % 1.)




class finetune_ENDE(nn.Module):
    def __init__(self, cutoff, neibor):
        super(finetune_ENDE, self).__init__()
        self.latent_dim = 256
        self.hidden_dim = 256
        self.fc_num_layers = 4
        self.fc_graph_lyers = 3
        self.type_sigma_begin = 5.
        self.type_sigma_end=0.01
        self.num_noise_level = 50
        self.device =  'cuda'
        self.encoder = DimeNetPlusPlusWrap(cutoff=cutoff, max_num_neighbors=neibor)#MaskedAutoencoderViT()
        #self.output_embedding = nn.Linear(1,self.num_targets)
        #self.decoder = DimeNetPlusPlusWrapd()
        #self.Masked_graph = Masked_graph()
        
        self.emb_dim = 64#300
        #self.emb_linear = nn.Sequential(
        #                    nn.Linear(128, self.emb_dim),
        #                    nn.BatchNorm1d(self.emb_dim),
        #                    nn.SiLU(),
        #                                                    )
        self.MaskedAutoencoder1 = nn.Sequential(nn.Linear(3,32), nn.LeakyReLU(inplace=True),nn.Linear(32, 8))#MaskedAutoencoderViT() 
        self.num_targets = self.emb_dim*2+16
        self.JK = "last"
        self.gnn_type = "gin"
        self.num_layer = 5
        self.dropout_ratio=0
        NUM_NODE_ATTR = 119
        # self.model = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.dropout_ratio, gnn_type = self.gnn_type)
        self.fc_out_emb_linear = nn.Linear(128, self.emb_dim)
        self.fc_out_c = nn.Linear(128, 64)
        self.fc_out_emb_linear1 = nn.Linear(128, self.emb_dim*2)
        self.output_embedding_extend = nn.Sequential(nn.Linear(1,self.num_targets), nn.LeakyReLU(inplace=True),nn.Linear(self.num_targets, self.num_targets))
        self.input_embedding_extend = nn.Sequential(nn.Linear(self.num_targets, self.num_targets), nn.LeakyReLU(inplace=True),nn.Linear(self.num_targets, self.num_targets))
        self.input_embedding_extend2 = nn.Sequential(nn.Linear(self.num_targets, self.num_targets), nn.LeakyReLU(inplace=True),nn.Linear(self.num_targets, self.num_targets))
        self.fc_out = nn.Sequential(
                nn.Linear(self.num_targets*2, self.num_targets//2),
                #nn.Softplus(),
                nn.LeakyReLU(inplace=True),
                nn.Linear(self.num_targets//2, self.num_targets//4),
                nn.LeakyReLU(inplace=True),
                nn.Linear(self.num_targets//4, self.num_targets//8),
                nn.LeakyReLU(inplace=True),
                nn.Linear(self.num_targets//8, 1)
            )
        self.fc_out_extend = nn.Sequential(
                nn.Linear(self.num_targets, self.num_targets//2),
                #nn.Softplus(),
                nn.LeakyReLU(inplace=True),
                nn.Linear(self.num_targets//2, self.num_targets//4),
                nn.LeakyReLU(inplace=True),
                nn.Linear(self.num_targets//4, self.num_targets//8),
                nn.LeakyReLU(inplace=True),
                nn.Linear(self.num_targets//8, 1)
            )
        self.fc_force = nn.Sequential(
                nn.Linear(self.num_targets, self.num_targets//2),
                #nn.Softplus(),
                nn.LeakyReLU(inplace=True),
                nn.Linear(self.num_targets//2, 1))
        
        


    

    def get_samples(self, x_vector, y_vector, K=20, eta=0.6):
        K=2
        eta = 0.8
        bias_vector = y_vector - x_vector
        #import ipdb
        #ipdb.set_trace()
        W_r = (torch.abs(bias_vector)-torch.min(torch.abs(bias_vector), 1,True)[0])/(torch.max(torch.abs(bias_vector),1, True)[0] - torch.min(torch.abs(bias_vector), 1,True)[0])
        R = []
        omega = torch.normal(0, W_r) 
        sample = x_vector + torch.mul(omega, bias_vector) 
        R.append(sample) 
        for i in range(1, K): 
            chain = [torch.unsqueeze(item, 1) for item in R[:i]] 
            average_omega = torch.mean(torch.cat(chain, axis=1), axis=1) 
            omega = eta * torch.normal(0, W_r) + (1.0 - eta) * torch.normal(average_omega, 1.0) 
            sample = x_vector + torch.mul(omega, bias_vector) 
            R.append(sample) 
        return R

                
    def get_ctl_loss(self, src_embedding,  trg_embedding, dynamic_coefficient):
       # dynamic_coefficient = 0.7
        batch_size = src_embedding.shape[0] 
        def get_ctl_logits(query, keys): 
            #import ipdb
            #ipdb.set_trace()
            query = query.unsqueeze(1) 
            keys = keys.unsqueeze(1)
            # expand_query: [batch_size, batch_size, hidden_size] 
            # expand_keys: [batch_size, batch_size, hidden_size] 
            # the current ref is the positive key, while others in the training batch are negative ones 
            expand_query = query.repeat(1, batch_size, 1) 
            expand_keys = keys.permute(1,0,2).repeat(batch_size, 1,1)
            # distances between queries and positive keys 
            d_pos = torch.sqrt(torch.sum(torch.pow(query - keys, 2.0), -1)) # [batch_size, 1] 
            d_pos = d_pos.repeat(1, batch_size) # [batch_size, batch_size] 
            d_neg = torch.sqrt(torch.sum(torch.pow(expand_query - expand_keys, 2.0), -1)) # [batch_size, batch_size] 
            lambda_coefficient = (d_pos / d_neg) ** dynamic_coefficient 
            
            hardness_masks = torch.gt(d_neg, d_pos).float() 
            hard_keys =(expand_query + torch.unsqueeze(lambda_coefficient, 2) * (expand_keys - expand_query)) * torch.unsqueeze(hardness_masks, 2) + expand_keys * torch.unsqueeze(1.0 - hardness_masks, 2) 
            logits = torch.bmm(query, hard_keys.permute(0,2,1))#, transpose_b=True) # [batch_size, 1, batch_size] 
            return logits 
       
        logits_src_trg = get_ctl_logits(src_embedding, trg_embedding) 
        logits_trg_src = get_ctl_logits(trg_embedding, src_embedding) + torch.unsqueeze(torch.eye(batch_size).cuda() * -1e9, 1) 
        logits = torch.cat([logits_src_trg, logits_trg_src], axis=2) # [batch_size, 1, 2*batch_size] 
        labels = torch.unsqueeze(torch.range(0,batch_size-1, dtype=torch.int32), 1) 
        #import ipdb
        #ipdb.set_trace()
        labels = F.one_hot(torch.arange(0, batch_size), num_classes=batch_size*2).cuda().unsqueeze(1).float() 
        #cross_entropy_fn = tf.nn.softmax_cross_entropy_with_logits 
        loss = torch.mean(self.softmax_cross_entropy_with_logits(labels, logits))
        return loss

    def softmax_cross_entropy_with_logits(self, labels, logits, dim=-1):
        return  -torch.sum(F.log_softmax(logits, dim=dim) * labels, dim=dim)



    def forward(self, batch_gt, batch_gt1, batch_output, mode, dynamic_coefficient=0.9):
        batch_gt = batch_gt.cuda()
        hidden_atom, hidden, egde_ij, edge_attr, x_att_, x_emb, rbf = self.encoder(batch_gt)
        
        hidden_atom_1, hidden_1, egde_ij_1, edge_attr_1, x_att_1, x_emb_1, rbf_1 = self.encoder(batch_gt1)

        
        lattice_coord = batch_gt.scaled_lattice_tensor
        latent1_coord = self.MaskedAutoencoder1(lattice_coord)
        
        lattice_coord1 = batch_gt1.scaled_lattice_tensor
        latent1_coord1 = self.MaskedAutoencoder1(lattice_coord1)

        
        hidden_atom_mean =  scatter(hidden_atom, batch_gt.batch, dim=0, reduce='mean')
        hidden_atom_max =  scatter(hidden_atom, batch_gt.batch, dim=0, reduce='max')
        hidden_atom = torch.concat([hidden_atom_mean, hidden_atom_max], -1)
        hidden_loss = torch.concat([hidden_atom, latent1_coord.view(latent1_coord.shape[0],-1)], dim=-1)
        

        hidden_atom_mean_1 =  scatter(hidden_atom_1, batch_gt1.batch, dim=0, reduce='mean')
        hidden_atom_max_1 =  scatter(hidden_atom_1, batch_gt1.batch, dim=0, reduce='max')
        hidden_atom_1 = torch.concat([hidden_atom_mean_1, hidden_atom_max_1], -1)
        hidden_loss_1 = torch.concat([hidden_atom_1, latent1_coord1.view(latent1_coord1.shape[0],-1)], dim=-1)



        if mode  == 'train' or mode == 'val': 
            hidden1_loss = self.input_embedding_extend(hidden_loss)
            hidden2_loss = self.input_embedding_extend(hidden_loss_1)

            hidden_output = self.output_embedding_extend(batch_output.unsqueeze(-1))
            loss = self.get_ctl_loss(torch.concat([hidden1_loss,hidden2_loss],-1), torch.concat([hidden_output,hidden_output],-1), dynamic_coefficient=dynamic_coefficient)
            
            d_choose1 = torch.sqrt(torch.sum(torch.pow(hidden_output - hidden1_loss, 2.0), -1))
            d_choose2 = torch.sqrt(torch.sum(torch.pow(hidden_output - hidden2_loss, 2.0), -1))
            d_hid1 = d_choose1.unsqueeze(1).repeat(1,hidden1_loss.size(1))
            d_hid2 = d_choose2.unsqueeze(1).repeat(1,hidden2_loss.size(1))
            x_sample1 = self.get_samples(hidden1_loss, hidden_output)
            x_sample_tensor1 = torch.stack(x_sample1,1)#.view(-1,hidden1_loss.shape[-1])
            x_sample2 = self.get_samples(hidden2_loss, hidden_output)
            x_sample_tensor2 = torch.stack(x_sample2,1)#.view(-1,hidden2_loss.shape[-1])
            x_size = torch.stack(x_sample1,1).size(1)
            d_sample1 = d_hid1.unsqueeze(1).repeat(1, x_size, 1)
            d_sample2 = d_hid2.unsqueeze(1).repeat(1, x_size, 1)
            x_sample = torch.where(d_sample1<=d_sample2, x_sample_tensor1, x_sample_tensor2)
            x_sample = x_sample.view(-1,hidden2_loss.shape[-1])
            
            energy = torch.where(d_choose1<=d_choose2,batch_gt.energy,batch_gt1.energy).float()
            loss_force = F.mse_loss(self.fc_force(x_sample), energy.unsqueeze(1).repeat(1,x_size).view(-1))
            

            pred_target = self.fc_out(torch.concat([hidden_loss, hidden_loss_1],-1)).view(-1)
            pred_extend = self.fc_out_extend(x_sample).view(-1)
            pred = torch.cat([pred_target, pred_extend], 0)
            
            target_extend = batch_output.unsqueeze(1).repeat(1,x_size).view(-1)
            target =  torch.cat([batch_output, target_extend], 0)
            test_output = pred_target
            return pred, target, loss, test_output, loss_force
        else:
            pred_target1 = self.fc_out(torch.concat([hidden_loss,hidden_loss_1],-1)).view(-1)
            test_output = pred_target1#torch.where((d_choose1_1+d_choose1_2)*0.5 >=(d_choose2_1+d_choose2_2), pred_target2, pred_target1)
            target = 0
            pred_target = test_output
            loss = 0
            loss_force=0
            return pred_target, target, loss, test_output, loss_force
        
