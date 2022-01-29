import torch
from SDE_CG import layers
from torch_scatter import scatter_add
import numpy as np

class ScoreNet_discretized(torch.nn.Module):
    def __init__(self, config,device):
        super(ScoreNet_discretized, self).__init__()
        self.config = config
        self.anneal_power = self.config.train.anneal_power
        self.hidden_dim = self.config.model.hidden_dim
        self.order = self.config.model.order
        self.noise_type = self.config.model.noise_type

        self.node_emb = torch.nn.Embedding(100, self.hidden_dim)
        self.edge_emb = torch.nn.Embedding(100, self.hidden_dim)
        self.input_mlp = layers.MultiLayerPerceptron(1, [self.hidden_dim, self.hidden_dim], activation=self.config.model.mlp_act)
        self.output_mlp = layers.MultiLayerPerceptron(2 * self.hidden_dim, \
                                [self.hidden_dim, self.hidden_dim // 2, 1], activation=self.config.model.mlp_act)

        self.model = layers.GraphIsomorphismNetwork(hidden_dim=self.hidden_dim, \
                                 num_convs=self.config.model.num_convs, \
                                 activation=self.config.model.gnn_act, \
                                 readout="sum", short_cut=self.config.model.short_cut, \
                                 concat_hidden=self.config.model.concat_hidden)
        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
                               self.config.model.num_noise_level)), dtype=torch.float32)
        self.sigmas = torch.nn.Parameter(sigmas, requires_grad=False) # (num_noise_level)
        self.device = device

    def noise_generator(self,data,noise_type,device):
        node2graph = data.batch
        edge2graph = node2graph[data.edge_index[0]]
        if noise_type == 'symmetry':
            num_nodes = scatter_add(torch.ones(data.num_nodes, dtype=torch.long, device=self.device), node2graph) # (num_graph)
            num_cum_nodes = num_nodes.cumsum(0) # (num_graph)
            node_offset = num_cum_nodes - num_nodes # (num_graph)
            edge_offset = node_offset[edge2graph] # (num_edge)

            num_nodes_square = num_nodes**2 # (num_graph)
            num_nodes_square_cumsum = num_nodes_square.cumsum(-1) # (num_graph)
            edge_start = num_nodes_square_cumsum - num_nodes_square # (num_graph)
            edge_start = edge_start[edge2graph]

            all_len = num_nodes_square_cumsum[-1]

            node_index = data.edge_index.t() - edge_offset.unsqueeze(-1)
            #node_in, node_out = node_index.t()
            node_large = node_index.max(dim=-1)[0]
            node_small = node_index.min(dim=-1)[0]
            undirected_edge_id = node_large * (node_large + 1) + node_small + edge_start
            symm_noise = torch.Tensor(all_len.detach().cpu().numpy(), device='cpu').normal_()
            symm_noise = symm_noise.to(self.device)
            d_noise = symm_noise[undirected_edge_id].unsqueeze(-1) # (num_edge, 1)
        elif noise_type == 'rand':
            d = data.edge_length
            d_noise = torch.randn_like(d)
        return d_noise

    def forward(self,data):
        node2graph = data.batch
        edge2graph = node2graph[data.edge_index[0]]
        noise_level = torch.randint(0, self.sigmas.size(0), (data.num_graphs,), device=self.device) #(num_graph)
        used_sigmas = self.sigmas[noise_level] #(num_graph)
        edge_sigmas = used_sigmas[edge2graph].unsqueeze(-1) #(num_edge, 1)

        d = data.edge_length
        d_noise = self.noise_generator(data,self.noise_type, self.device)
        perturbed_d = d + d_noise

        #estimate scores
        atom_attr = self.node_emb(data.node_type)
        bond_attr = self.edge_emb(data.edge_type)
        d_emb = self.input_mlp(perturbed_d)
        bond_attr = d_emb * bond_attr
        #GNN model
        output = self.model(data, atom_attr, bond_attr)
        h_row, h_col = output["node_feature"][data.edge_index[0]], output["node_feature"][data.edge_index[1]]
        distance_feature =torch.cat([h_row*h_col, bond_attr], dim=-1)
        scores = self.output_mlp(distance_feature)
        scores = scores * (1. / edge_sigmas)

        target = -1 / (edge_sigmas **2) * d_noise
        return (scores, target, edge_sigmas)
