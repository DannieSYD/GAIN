import torch
from torch import Tensor
from torch_geometric.typing import OptTensor
from torch_scatter import scatter

from typing import Callable, Optional, Tuple, Union
from torch_geometric.nn.resolver import activation_resolver
from models.models_3d.dimenet import BesselBasisLayer, SphericalBasisLayer, EmbeddingBlock, OutputBlock, InteractionBlock, triplets, OutputPPBlock, InteractionPPBlock
from torch_geometric.nn import radius_graph


class DimeNet(torch.nn.Module):
    r"""The directional message passing neural network (DimeNet) from the
    `"Directional Message Passing for Molecular Graphs"
    <https://arxiv.org/abs/2003.03123>`_ paper.
    DimeNet transforms messages based on the angle between them in a
    rotation-equivariant fashion.
    .. note::
        For an example of using a pretrained DimeNet variant, see
        `examples/qm9_pretrained_dimenet.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        qm9_pretrained_dimenet.py>`_.
    Args:
        hidden_channels (int): Hidden embedding size.
        out_channels (int): Size of each output sample.
        num_blocks (int): Number of building blocks.
        num_bilinear (int): Size of the bilinear layer tensor.
        num_spherical (int): Number of spherical harmonics.
        num_radial (int): Number of radial basis functions.
        cutoff (float, optional): Cutoff distance for interatomic
            interactions. (default: :obj:`5.0`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            collect for each node within the :attr:`cutoff` distance.
            (default: :obj:`32`)
        envelope_exponent (int, optional): Shape of the smooth cutoff.
            (default: :obj:`5`)
        num_before_skip (int, optional): Number of residual layers in the
            interaction blocks before the skip connection. (default: :obj:`1`)
        num_after_skip (int, optional): Number of residual layers in the
            interaction blocks after the skip connection. (default: :obj:`2`)
        num_output_layers (int, optional): Number of linear layers for the
            output blocks. (default: :obj:`3`)
        act (str or Callable, optional): The activation function.
            (default: :obj:`"swish"`)
    """

    def __init__(
        self,
        max_atomic_num: int,
        hidden_channels: int,
        out_channels: int,
        num_blocks: int,
        num_bilinear: int,
        num_spherical: int,
        num_radial,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        envelope_exponent: int = 5,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_output_layers: int = 3,
        act: Union[str, Callable] = 'swish',
    ):
        super().__init__()

        if num_spherical < 2:
            raise ValueError("num_spherical should be greater than 1")

        act = activation_resolver(act)

        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.num_blocks = num_blocks

        self.rbf = BesselBasisLayer(num_radial, cutoff, envelope_exponent)
        self.sbf = SphericalBasisLayer(num_spherical, num_radial, cutoff,
                                       envelope_exponent)

        self.emb = EmbeddingBlock(max_atomic_num, num_radial, hidden_channels, act)

        self.output_blocks = torch.nn.ModuleList([
            OutputBlock(num_radial, hidden_channels, out_channels,
                        num_output_layers, act) for _ in range(num_blocks + 1)
        ])

        self.interaction_blocks = torch.nn.ModuleList([
            InteractionBlock(hidden_channels, num_bilinear, num_spherical,
                             num_radial, num_before_skip, num_after_skip, act)
            for _ in range(num_blocks)
        ])
        self.blocks = torch.nn.ModuleList()
        self.out_blocks = torch.nn.ModuleList()
        self.build_blocks()

    def reset_parameters(self):
        self.rbf.reset_parameters()
        self.emb.reset_parameters()
        for out in self.output_blocks:
            out.reset_parameters()
        for interaction in self.interaction_blocks:
            interaction.reset_parameters()

    def build_blocks(self):
        for i in range(self.num_blocks):
            block = self.interaction_blocks[i]
            out_block = self.output_blocks[i+1]
            self.blocks.append(block)
            self.out_blocks.append(out_block)

    def preprocess(self, data):
        z, pos, batch = data.x[:, 0], data.pos, data.batch
        edge_index = radius_graph(
            pos, r=self.cutoff, batch=batch, max_num_neighbors=self.max_num_neighbors)
        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(
            edge_index, num_nodes=z.size(0))

        # Calculate distances.
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

        # Calculate angles.
        pos_i = pos[idx_i]
        pos_ji, pos_ki = pos[idx_j] - pos_i, pos[idx_k] - pos_i
        a = (pos_ji * pos_ki).sum(dim=-1)
        b = torch.cross(pos_ji, pos_ki).norm(dim=-1)
        angle = torch.atan2(b, a)

        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)

        # Embedding block.
        x = self.emb(z, rbf, i, j)
        P = self.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))

        return {
            'i': i,
            'rbf': rbf,
            'sbf': sbf,
            'idx_kj': idx_kj,
            'idx_ji': idx_ji,
            'pos': pos,
            'edge_index': edge_index,
        }, x, P

    def block_call(self, i, x, P, data_dict, batch):
        interaction_block = self.interaction_blocks[i]
        output_block = self.output_blocks[i+1]
        ii = data_dict['i']
        rbf = data_dict['rbf']
        sbf = data_dict['sbf']
        idx_kj = data_dict['idx_kj']
        idx_ji = data_dict['idx_ji']
        pos = data_dict['pos']
        x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
        P = P + output_block(x, rbf, ii, num_nodes=pos.size(0))
        return x, P

    def postprocess(self, P, batch):
        return P.sum(dim=0) if batch is None else scatter(P, batch, dim=0)



    # def forward(
    #     self,
    #     z: Tensor,
    #     pos: Tensor,
    #     batch: OptTensor = None,
    # ) -> Tensor:
    #     """"""
    #     edge_index = radius_graph(pos, r=self.cutoff, batch=batch,
    #                               max_num_neighbors=self.max_num_neighbors)
    #
    #     i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(
    #         edge_index, num_nodes=z.size(0))
    #
    #     # Calculate distances.
    #     dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()
    #
    #     # Calculate angles.
    #     pos_i = pos[idx_i]
    #     pos_ji, pos_ki = pos[idx_j] - pos_i, pos[idx_k] - pos_i
    #     a = (pos_ji * pos_ki).sum(dim=-1)
    #     b = torch.cross(pos_ji, pos_ki).norm(dim=-1)
    #     angle = torch.atan2(b, a)
    #
    #     rbf = self.rbf(dist)
    #     sbf = self.sbf(dist, angle, idx_kj)
    #
    #     # Embedding block.
    #     x = self.emb(z, rbf, i, j)
    #     P = self.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))
    #
    #     # Interaction blocks.
    #     for interaction_block, output_block in zip(self.interaction_blocks,
    #                                                self.output_blocks[1:]):
    #         x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
    #         P = P + output_block(x, rbf, i, num_nodes=pos.size(0))
    #
    #     return P.sum(dim=0) if batch is None else scatter(P, batch, dim=0)


class DimeNetPlusPlus(DimeNet):
    r"""The DimeNet++ from the `"Fast and Uncertainty-Aware
    Directional Message Passing for Non-Equilibrium Molecules"
    <https://arxiv.org/abs/2011.14115>`_ paper.
    :class:`DimeNetPlusPlus` is an upgrade to the :class:`DimeNet` model with
    8x faster and 10% more accurate than :class:`DimeNet`.
    Args:
        hidden_channels (int): Hidden embedding size.
        out_channels (int): Size of each output sample.
        num_blocks (int): Number of building blocks.
        int_emb_size (int): Size of embedding in the interaction block.
        basis_emb_size (int): Size of basis embedding in the interaction block.
        out_emb_channels (int): Size of embedding in the output block.
        num_spherical (int): Number of spherical harmonics.
        num_radial (int): Number of radial basis functions.
        cutoff: (float, optional): Cutoff distance for interatomic
            interactions. (default: :obj:`5.0`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            collect for each node within the :attr:`cutoff` distance.
            (default: :obj:`32`)
        envelope_exponent (int, optional): Shape of the smooth cutoff.
            (default: :obj:`5`)
        num_before_skip: (int, optional): Number of residual layers in the
            interaction blocks before the skip connection. (default: :obj:`1`)
        num_after_skip: (int, optional): Number of residual layers in the
            interaction blocks after the skip connection. (default: :obj:`2`)
        num_output_layers: (int, optional): Number of linear layers for the
            output blocks. (default: :obj:`3`)
        act: (str or Callable, optional): The activation funtion.
            (default: :obj:`"swish"`)
    """

    def __init__(
            self,
            max_atomic_num: int,
            hidden_channels: int,
            out_channels: int,
            num_blocks: int,
            int_emb_size: int,
            basis_emb_size: int,
            out_emb_channels: int,
            num_spherical: int,
            num_radial: int,
            cutoff: float = 5.0,
            max_num_neighbors: int = 32,
            envelope_exponent: int = 5,
            num_before_skip: int = 1,
            num_after_skip: int = 2,
            num_output_layers: int = 3,
            act: Union[str, Callable] = 'swish',
    ):
        act = activation_resolver(act)

        super().__init__(
            max_atomic_num=max_atomic_num,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_blocks=num_blocks,
            num_bilinear=1,
            num_spherical=num_spherical,
            num_radial=num_radial,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            envelope_exponent=envelope_exponent,
            num_before_skip=num_before_skip,
            num_after_skip=num_after_skip,
            num_output_layers=num_output_layers,
            act=act,
        )

        # We are re-using the RBF, SBF and embedding layers of `DimeNet` and
        # redefine output_block and interaction_block in DimeNet++.
        # Hence, it is to be noted that in the above initalization, the
        # variable `num_bilinear` does not have any purpose as it is used
        # solely in the `OutputBlock` of DimeNet:
        self.output_blocks = torch.nn.ModuleList([
            OutputPPBlock(num_radial, hidden_channels, out_emb_channels,
                          out_channels, num_output_layers, act)
            for _ in range(num_blocks + 1)
        ])

        self.interaction_blocks = torch.nn.ModuleList([
            InteractionPPBlock(hidden_channels, int_emb_size, basis_emb_size,
                               num_spherical, num_radial, num_before_skip,
                               num_after_skip, act) for _ in range(num_blocks)
        ])

        self.reset_parameters()
