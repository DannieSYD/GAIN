import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter
from torch_geometric.nn import global_add_pool
from torch_geometric.nn.models import schnet
from torch_geometric.nn import radius_graph, MessagePassing
from torch_geometric.data import Data, Batch

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


class ConfNetDSS(nn.Module):

    def __init__(self, config):
        super(ConfNetDSS, self).__init__()
        self.device = config.device
        self.cutoff = config.cutoff
        hidden_dim = config.hidden_dim
        num_gaussians = 50
        self.distance_expansion = schnet.GaussianSmearing(0.0, self.cutoff, 
                                                          num_gaussians)
        self.atom_encoder = AtomEncoder(emb_dim=config.hidden_dim)

        ### set the initial virtual node embedding to 0.
        self.virtual_node = config.virtual_node
        if self.virtual_node:
            self.virtualnode_embedding = torch.nn.Embedding(1, config.hidden_dim)
            torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        self.dss_layers = nn.ModuleList()
        for _ in range(config.num_gnn_layers):
            self.dss_layers.append(DSSConf(hidden_channels=config.hidden_dim,
                                           num_gaussians=num_gaussians, 
                                           num_filters=config.num_filters, 
                                           cutoff=self.cutoff, 
                                           use_conf=config.use_conf,
                                           use_graph=config.use_graph,
                                           residual=config.residual))

        if self.virtual_node:
            self.mlp_virtualnode = torch.nn.Sequential(
                    torch.nn.Linear(hidden_dim, hidden_dim), 
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, hidden_dim), 
                    torch.nn.ReLU())

        self.out_layer = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),nn.Linear(hidden_dim, config.num_tasks))


    @staticmethod
    def unbatch_first_element(batch_obj):
        num_nodes_per_graph = batch_obj.batch.bincount()[0]
        graph_data = Data()

        for key, value in batch_obj:
            if key == 'edge_index':
                edge_index = value[:, value[0] < num_nodes_per_graph]
                graph_data.edge_index = edge_index
            elif key == 'edge_attr':
                edge_mask = batch_obj.edge_index[0] < num_nodes_per_graph
                if value.ndim == 1:
                    edge_attr = value[edge_mask]
                else:
                    edge_attr = value[edge_mask, :]
                graph_data.edge_attr = edge_attr
            elif torch.is_tensor(value) and value.size(0) == batch_obj.num_nodes:
                graph_data[key] = value[:num_nodes_per_graph]
            else:
                graph_data[key] = value

        return graph_data

    @staticmethod
    def get_mol_idx(data_list):
        molecule_index = [[i] * (d.batch.max().item() + 1) for i, d in enumerate(data_list)]
        molecule_index = sum(molecule_index, [])
        return torch.Tensor(molecule_index).long()

    @staticmethod
    def get_conf_node_idx(data_list):
        node_count = 0
        conf_node_idx = []
        for i, d in enumerate(data_list):
            num_confs = d.batch.max().item() + 1
            num_nodes = d.batch.shape[0] // num_confs
            conf_node_idx.extend(
                (torch.arange(num_nodes) + node_count).repeat(num_confs))
            node_count += num_nodes
        return torch.Tensor(conf_node_idx).long()

    def forward(self, data_ori):
        data = data_ori[0].to(self.device)
        data_list = data.to_data_list()
        data_first_list = [self.unbatch_first_element(data) for data in data_list]
        data_topo = Batch.from_data_list(data_first_list)
        part_molecule_indices = self.get_mol_idx(data_list).to(self.device)
        conf_node_idx = self.get_conf_node_idx(data_list).to(self.device)

        data.pos_batch = data.batch
        data.conf_node_batch = conf_node_idx
        data.conf_batch = part_molecule_indices
        x = self.atom_encoder(data_topo.x)
        x = x[data.conf_node_batch]
        edge_index_conf = radius_graph(data.pos, r=self.cutoff, 
                                       batch=data.pos_batch)
        row, col = edge_index_conf
        edge_weight_conf = (data.pos[row] - data.pos[col]).norm(dim=-1)
        edge_attr_conf = self.distance_expansion(edge_weight_conf)

        edge_index_graph = data_topo.edge_index
        edge_attr_graph = data_topo.edge_attr

        if self.virtual_node:
            virtualnode_embedding = self.virtualnode_embedding(torch.zeros(data.pos_batch[-1].item() + 1).to(edge_index_graph.dtype).to(edge_index_graph.device))

        for i, layer in enumerate(self.dss_layers):
            if self.virtual_node:
                x = x + virtualnode_embedding[data.pos_batch]
                if i < len(self.dss_layers) - 1:
                    virtualnode_embedding_temp = global_add_pool(x, data.pos_batch) + virtualnode_embedding
                    virtualnode_embedding = self.mlp_virtualnode(virtualnode_embedding_temp)

            # print(x.shape, x)
            # print(data.conf_node_batch.shape, data.conf_node_batch)
            # exit(123)
            x = layer(x, data.conf_node_batch,
                      edge_index_conf, edge_weight_conf, edge_attr_conf, 
                      edge_index_graph, edge_attr_graph)
            x = F.relu(x)
        
        x = scatter(x, data.pos_batch, dim=0, reduce="max")
        x = scatter(x, data.conf_batch, dim=0, reduce="max")
        x = self.out_layer(x)

        if self.training:
            return x
        else:
            # At inference time, relu is applied to output to ensure positivity
            return torch.clamp(x, min=0, max=50)


class DSSConf(nn.Module):

    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff,
                 use_conf, use_graph, residual):
        super(DSSConf, self).__init__()
        self.use_conf = use_conf
        self.use_graph = use_graph
        self.residual = residual
        if self.use_conf:
            self.mlp = nn.Sequential(
                    nn.Linear(num_gaussians, num_filters),
                    nn.ReLU(),
                    nn.Linear(num_filters, num_filters),
            )
            self.cf_conv = schnet.CFConv(hidden_channels, hidden_channels, 
                                         num_filters, self.mlp, cutoff)
            self.lin = nn.Linear(hidden_channels, hidden_channels)
        if self.use_graph:
            self.gin_conv = GINConv(emb_dim=hidden_channels)
            self.bn = nn.BatchNorm1d(hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        if self.use_conf:
            self.cf_conv.reset_parameters()

    def forward(self, x, conf_node_batch,
                edge_index_conf, edge_weight_conf, edge_attr_conf, 
                edge_index_graph, edge_attr_graph):
        if self.residual:
            out = x
        else:
            out = 0
        if self.use_conf:
            h = self.cf_conv(x, edge_index_conf, edge_weight_conf, edge_attr_conf)
            h = F.relu(h)
            h = self.lin(h)
            out = out + h
        if self.use_graph:
            x_agg = scatter(x, conf_node_batch, dim=0, reduce="max")
            x_agg = self.gin_conv(x_agg, edge_index_graph, edge_attr_graph)
            x_agg = self.bn(x_agg)

            out = out + x_agg[conf_node_batch]
        return out


class GINConv(MessagePassing):

    def __init__(self, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConv, self).__init__(aggr = "add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        
        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr):

        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)
        
    def update(self, aggr_out):
        return aggr_out
