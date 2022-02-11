import torch
from SDE_CG import layers
from torch_scatter import scatter_add
import numpy as np
import torch.nn as nn
from .SDE_builder import GaussianFourierProjection


class ScoreNet(torch.nn.Module):
    def __init__(self,config, marginal_prob_std, hidden_dim=256,device='cuda'):
        super().__init__()
        self.config = config
        self.hidden_dim = hidden_dim
        self.bond_emb = torch.nn.Embedding(100, self.hidden_dim)
        self.atom_emb = torch.nn.Embedding(100, self.hidden_dim)
        self.input_mlp = layers.MultiLayerPerceptron(1, [self.hidden_dim, self.hidden_dim], activation=self.config.model.mlp_act)
        self.output_mlp = layers.MultiLayerPerceptron(2*self.hidden_dim, [hidden_dim, hidden_dim//2,1], activation=config.model.gnn_act)
        self.model = layers.GraphIsomorphismNetwork(hidden_dim=self.hidden_dim, \
                                                    num_convs=self.config.model.num_convs, \
                                                    activation=self.config.model.gnn_act, \
                                                    readout="sum", short_cut=self.config.model.short_cut, \
                                                    concat_hidden=self.config.model.concat_hidden)
        self.model = self.model.to(device)
        self.t_embed = nn.Sequential(GaussianFourierProjection(embed_dim=hidden_dim),
                                  nn.Linear(hidden_dim, hidden_dim))
        self.dense1 = nn.Linear(hidden_dim, 1)
        self.marginal_prob_std = marginal_prob_std
        self.device = device
        
    @torch.no_grad()
    def get_score(self,data,d,t):
        t_embedding = self.t_embed(t)
        t_embedding = self.dense1(t_embedding)
        node2graph = data.batch
        edge2graph = node2graph[data.edge_index[0]]
        atom_attr = self.atom_emb(data.node_type)  # (atom_dim, hidden_dim)
        bond_attr = self.bond_emb(data.edge_type)  # (edge_dim, hidden_dim)
        d_emb = self.input_mlp(d)  # (edge_dim, hidden_dim) =[58234, 256]
        d_emb += t_embedding[edge2graph]
        bond_attr = d_emb * bond_attr  # (edge_dim, hidden_dim) =[58234, 256]
        output = self.model(data, atom_attr, bond_attr)
        h_row, h_col = output["node_feature"][data.edge_index[0]], output["node_feature"][data.edge_index[1]]
        distance_feature = torch.cat([h_row * h_col, bond_attr], dim=-1)  # (edge_dim, 2*hidden_dim) =[58234, 512]
        scores = self.output_mlp(distance_feature)  # (edge_dim, 1) = [58234, 1]
        scores = scores.view(-1)
        scores = scores / self.marginal_prob_std(t[edge2graph]).to(self.device)
        return scores
    
    
    def forward(self, data, t):
        '''
        Score_Net function, which is constructed by two MLP blocks, two embedding layers for atom and bond attribution
        and an embedding layer for time.
        input: data sturcture(we need node_type, bond_type, edge_length, batch, edge_index, atom_feature)
        '''
        node2graph = data.batch
        edge2graph = node2graph[data.edge_index[0]]
        t_embedding = self.t_embed(t)             #(batch_dim, hidden_dim) = (128, 256)
        t_embedding = self.dense1(t_embedding)    #(batch_dim, 1) = (128,1)
        d = data.edge_length                      #(edge_dim, 1)
        atom_attr = self.atom_emb(data.node_type) #(atom_dim, hidden_dim)
        bond_attr = self.bond_emb(data.edge_type) #(edge_dim, hidden_dim)
        d_emb = self.input_mlp(d)                 #(edge_dim, hidden_dim) =[58234, 256]
        d_emb += t_embedding[edge2graph]          #(edge_dim, hidden_dim) =[58234, 256]
        bond_attr = d_emb * bond_attr             #(edge_dim, hidden_dim) =[58234, 256]
        output = self.model(data, atom_attr, bond_attr)
        h_row, h_col = output["node_feature"][data.edge_index[0]], output["node_feature"][data.edge_index[1]]
        distance_feature = torch.cat([h_row*h_col, bond_attr], dim=-1)  #(edge_dim, 2*hidden_dim) =[58234, 512]
        scores = self.output_mlp(distance_feature)   #(edge_dim, 1) = [58234, 1]
        scores = scores.view(-1)                    #(edge_dim)
        scores = scores / self.marginal_prob_std(t[edge2graph]).to(self.device)     #(edge_dim)
        return scores
