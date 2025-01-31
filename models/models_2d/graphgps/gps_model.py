import torch
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.data import Data, Batch
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.models.layer import (new_layer_config,
                                                   BatchNorm1dNode)
from torch_geometric.graphgym.register import register_network
from models.models_2d.encoders import AtomEncoder, BondEncoder
from .layer.gps_layer import GPSLayer


class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
    """
    def __init__(self, dim_in):
        super(FeatureEncoder, self).__init__()
        self.dim_in = dim_in
        if cfg.dataset.node_encoder:
            # Encode integer node features via nn.Embeddings
            NodeEncoder = register.node_encoder_dict[
                cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.gnn.dim_inner)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_inner, -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))
            # Update dim_in to reflect the new dimension of the node features
            self.dim_in = cfg.gnn.dim_inner
        if cfg.dataset.edge_encoder:
            # Hard-limit max edge dim for PNA.
            if 'PNA' in cfg.gt.layer_type:
                cfg.gnn.dim_edge = min(128, cfg.gnn.dim_inner)
            else:
                cfg.gnn.dim_edge = cfg.gnn.dim_inner
            # Encode integer edge features via nn.Embeddings
            EdgeEncoder = register.edge_encoder_dict[
                cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(cfg.gnn.dim_edge)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_edge, -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


@register_network('GPSModel')
class GPSModel(torch.nn.Module):
    """General-Powerful-Scalable graph transformer.
    https://arxiv.org/abs/2205.12454
    Rampasek, L., Galkin, M., Dwivedi, V. P., Luu, A. T., Wolf, G., & Beaini, D.
    Recipe for a general, powerful, scalable graph transformer. (NeurIPS 2022)
    """

    def __init__(self, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.atom_encoder = AtomEncoder(emb_dim=hidden_dim)
        self.bond_encoder = BondEncoder(emb_dim=hidden_dim)

        local_gnn_type = 'GINE'
        global_model_type = 'Transformer'

        self.layers = []
        for _ in range(num_layers):
            self.layers.append(GPSLayer(
                dim_h=hidden_dim,
                local_gnn_type=local_gnn_type,
                global_model_type=global_model_type,
                num_heads=4,
                dropout=0.0,
                layer_norm=False,
                batch_norm=True
            ))

        # GNNHead = register.head_dict[cfg.gnn.head]
        # self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=output_dim)

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

    def preprocess(self, data_list):
        data_first_list = [self.unbatch_first_element(data) for data in data_list]
        batch = Batch.from_data_list(data_first_list)
        x, edge_index, edge_attr, batch = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        x = self.atom_encoder(x)
        edge_attr = self.bond_encoder(edge_attr)

        return {'node_feat': x, 'edge_index': edge_index, 'edge_attr': edge_attr}, x

    def block_call(self, i, x, data_dict, *args, **kwargs):
        edge_index = data_dict['edge_index']
        edge_attr = data_dict['edge_attr']
        batch = Data(x=x, edge_index=edge_index, edge_attr=edge_attr).to(x.device)
        block = self.layers[i].to(x.device)
        batch = block(batch)
        return batch.x

    # def post_process(self, batch):
    #     return self.post_mp(batch)


# @register_network('GPSModel')
# class GPSModel(torch.nn.Module):
#     """General-Powerful-Scalable graph transformer.
#     https://arxiv.org/abs/2205.12454
#     Rampasek, L., Galkin, M., Dwivedi, V. P., Luu, A. T., Wolf, G., & Beaini, D.
#     Recipe for a general, powerful, scalable graph transformer. (NeurIPS 2022)
#     """
#
#     def __init__(self, dim_in, dim_out):
#         super().__init__()
#         self.encoder = FeatureEncoder(dim_in)
#         dim_in = self.encoder.dim_in
#
#         if cfg.gnn.layers_pre_mp > 0:
#             self.pre_mp = GNNPreMP(
#                 dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
#             dim_in = cfg.gnn.dim_inner
#
#         if not cfg.gt.dim_hidden == cfg.gnn.dim_inner == dim_in:
#             raise ValueError(
#                 f"The inner and hidden dims must match: "
#                 f"embed_dim={cfg.gt.dim_hidden} dim_inner={cfg.gnn.dim_inner} "
#                 f"dim_in={dim_in}"
#             )
#
#         try:
#             local_gnn_type, global_model_type = cfg.gt.layer_type.split('+')
#         except:
#             raise ValueError(f"Unexpected layer type: {cfg.gt.layer_type}")
#         layers = []
#         for _ in range(cfg.gt.layers):
#             layers.append(GPSLayer(
#                 dim_h=cfg.gt.dim_hidden,
#                 local_gnn_type=local_gnn_type,
#                 global_model_type=global_model_type,
#                 num_heads=cfg.gt.n_heads,
#                 act=cfg.gnn.act,
#                 pna_degrees=cfg.gt.pna_degrees,
#                 equivstable_pe=cfg.posenc_EquivStableLapPE.enable,
#                 dropout=cfg.gt.dropout,
#                 attn_dropout=cfg.gt.attn_dropout,
#                 layer_norm=cfg.gt.layer_norm,
#                 batch_norm=cfg.gt.batch_norm,
#                 bigbird_cfg=cfg.gt.bigbird,
#                 log_attn_weights=cfg.train.mode == 'log-attn-weights',
#             ))
#         self.layers = torch.nn.Sequential(*layers)
#
#         GNNHead = register.head_dict[cfg.gnn.head]
#         self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)
#
#     def forward(self, batch):
#         for module in self.children():
#             batch = module(batch)
#         return batch
